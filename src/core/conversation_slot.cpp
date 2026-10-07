// src/core/conversation_slot.cpp - see include/strata/core/conversation_slot.hpp.
#include "strata/core/conversation_slot.hpp"
#include "conversation_checked.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::core {
namespace {

bool fail(std::string& error, const std::string& message) {
    error = "conversation slot: " + message;
    return false;
}

// Two passes over the same carve: a null base sizes it, a real one hands out the pointers.
struct Carve {
    uint8_t* base = nullptr;
    size_t used = 0;
    template<class T> T* take(size_t bytes) {
        if (!bytes) return nullptr;
        used = (used + 255) & ~size_t(255);
        T* p = base ? reinterpret_cast<T*>(base + used) : nullptr;
        used += bytes;
        return p;
    }
};

// The mirror of one QSA state: its flags and extents, and the five snapshot arrays (pools() in
// conversation_snapshot.cpp) in their own memory - VRAM for a resident layer, pinned RAM for a streamed one.
bool mirror_state(const QsaState& a, const ModelGeometry& g, bool index, const ConversationStateSizes& z,
                  Carve& dev, Carve& host, QsaState& m, std::string& error) {
    std::array<ConversationRange, 5> cap;
    if (!conversation_kv_ranges(a, g, a.max_cells, index, cap, error)) return false;
    m = QsaState{};
    m.kv_int8 = a.kv_int8; m.kv_q4 = a.kv_q4; m.kv_hybrid = a.kv_hybrid; m.kv_mode = a.kv_mode;
    m.kv_rot = a.kv_rot;   // part of the snapshot format (#293): the mirror must read as the session's
    m.n_slots = a.n_slots; m.n_pages = a.n_pages; m.max_cells = a.max_cells;
    m.idx_pooled_rows = a.idx_pooled_rows;
    Carve& kv = a.kv_mode != 0 ? host : dev;
    void* p[4];
    for (int i = 0; i < 4; ++i) p[i] = kv.take<uint8_t>(cap[(size_t) i].bytes);
    auto& h = m.host;
    if (a.kv_hybrid) {
        if (a.kv_mode != 0) { h.k_q = (int8_t*) p[0]; h.v_q4 = (uint8_t*) p[1]; h.k_scale = (uint16_t*) p[2]; }
        else { m.k_q = (int8_t*) p[0]; m.v_q4 = (uint8_t*) p[1]; m.k_scale = (uint16_t*) p[2]; }
    } else if (a.kv_q4) {
        if (a.kv_mode != 0) { h.k_q4 = (uint8_t*) p[0]; h.v_q4 = (uint8_t*) p[1]; }
        else { m.k_q4 = (uint8_t*) p[0]; m.v_q4 = (uint8_t*) p[1]; }
    } else if (a.kv_int8) {
        if (a.kv_mode != 0) {
            h.k_q = (int8_t*) p[0]; h.v_q = (int8_t*) p[1]; h.k_scale = (uint16_t*) p[2]; h.v_scale = (uint16_t*) p[3];
        } else {
            m.k_q = (int8_t*) p[0]; m.v_q = (int8_t*) p[1]; m.k_scale = (uint16_t*) p[2]; m.v_scale = (uint16_t*) p[3];
        }
    } else {
        if (a.kv_mode != 0) { h.k_pool = (uint16_t*) p[0]; h.v_pool = (uint16_t*) p[1]; }
        else { m.k_pool = (uint16_t*) p[0]; m.v_pool = (uint16_t*) p[1]; }
    }
    if (index) {
        m.idx_tail = dev.take<float>(z.tail);
        m.idx_dead = dev.take<float>(z.dead);
        m.idx_block_pos = dev.take<int32_t>(z.block_pos);
        m.idx_pooled = dev.take<float>((size_t) a.idx_pooled_rows * (size_t) g.idx_key_dim * sizeof(float));
    }
    return true;
}

bool build(const SessionState& a, const QsaState* draft, const ModelGeometry& g, Carve& dev, Carve& host,
           ConversationSlot& slot, std::string& error) {
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, a, z, error)) return false;
    if (a.qsa_alloc > 0 && a.qsa_states == nullptr) return fail(error, "the session has no QSA states");
    SessionState& m = slot.ss;
    m = SessionState{};
    m.max_cells = a.max_cells; m.k = a.k;
    m.layer_lo = a.layer_lo; m.layer_hi = a.layer_hi;
    m.qsa_ord0 = a.qsa_ord0; m.qsa_alloc = a.qsa_alloc; m.gdn_ord0 = a.gdn_ord0; m.gdn_alloc = a.gdn_alloc;
    m.gdn_state = dev.take<float>(z.gdn);
    m.ple_hist = a.ple_hist ? dev.take<float>(z.ple) : nullptr;
    slot.qsa.assign((size_t) (a.qsa_ord0 + a.qsa_alloc), QsaState{});
    for (int64_t j = 0; j < a.qsa_alloc; ++j) {
        const size_t o = (size_t) (a.qsa_ord0 + j);
        if (!mirror_state(a.qsa_states[o], g, true, z, dev, host, slot.qsa[o], error)) return false;
    }
    m.qsa_states = slot.qsa.data();
    slot.has_draft = draft != nullptr;
    return !draft || mirror_state(*draft, g, false, z, dev, host, slot.draft, error);
}

// Every range of one side's conversation state at `upto` cells, in a fixed order both sides share.
bool ranges(const SessionState& s, const QsaState* draft, const ModelGeometry& g, int64_t upto,
            std::vector<ConversationRange>& out, std::string& error) {
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, s, z, error)) return false;
    out.clear();
    out.push_back({s.gdn_state, z.gdn, false});
    out.push_back({s.ple_hist, s.ple_hist ? z.ple : 0, false});
    std::array<ConversationRange, 5> kv;
    for (int64_t j = 0; j < s.qsa_alloc; ++j) {
        const QsaState& st = s.qsa_states[(size_t) (s.qsa_ord0 + j)];
        out.push_back({st.idx_tail, z.tail, false});
        out.push_back({st.idx_dead, z.dead, false});
        out.push_back({st.idx_block_pos, z.block_pos, false});
        if (!conversation_kv_ranges(st, g, upto, true, kv, error)) return false;
        out.insert(out.end(), kv.begin(), kv.end());
    }
    if (!draft) return true;
    if (!conversation_kv_ranges(*draft, g, upto, false, kv, error)) return false;
    out.insert(out.end(), kv.begin(), kv.end());
    return true;
}

// THE MOVE THREADS MUST NOT INHERIT THE HOST LOOP'S PIN.  The serving thread pins itself to the first physical
// core for the whole session (SessionLoopScratch, session.cpp), and a std::thread starts with its creator's mask:
// sixteen "parallel" threads on one core moved 1.58 GiB each way in 192 ms where the same exchange takes 50 ms
// across the machine.  So each one widens itself to the process's affinity as it was at start-up - a static
// initializer runs before anything pins, and keeps a `taskset` the user gave.
#if defined(_WIN32)
void unpin() {
    DWORD_PTR process = 0, system = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) && process)
        SetThreadAffinityMask(GetCurrentThread(), process);
}
#else
const cpu_set_t g_start_cpus = [] {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) CPU_ZERO(&set);
    return set;
}();
void unpin() {
    if (CPU_COUNT(&g_start_cpus) > 0) pthread_setaffinity_np(pthread_self(), sizeof g_start_cpus, &g_start_cpus);
}
#endif

// The pinned ranges on CPU threads, in 8 MiB pieces taken from a shared counter.
void host_moves(ConversationSlotMove how, const std::vector<std::pair<ConversationRange, ConversationRange>>& work,
                int threads) {
    constexpr size_t piece = 8u << 20;
    struct Piece { uint8_t* a; uint8_t* b; size_t n; };
    std::vector<Piece> pieces;
    for (const auto& [a, b] : work)
        for (size_t at = 0; at < a.bytes; at += piece)
            pieces.push_back({(uint8_t*) a.data + at, (uint8_t*) b.data + at, std::min(piece, a.bytes - at)});
    if (pieces.empty()) return;
    std::atomic<size_t> next{0};
    auto run = [&] {
        std::vector<uint8_t> tmp(how == ConversationSlotMove::swap ? 64u << 10 : 0);
        for (size_t i; (i = next.fetch_add(1)) < pieces.size();) {
            const Piece& p = pieces[i];
            if (how == ConversationSlotMove::copy) { std::memcpy(p.b, p.a, p.n); continue; }
            for (size_t at = 0; at < p.n; at += tmp.size()) {
                const size_t n = std::min(tmp.size(), p.n - at);
                std::memcpy(tmp.data(), p.a + at, n);
                std::memcpy(p.a + at, p.b + at, n);
                std::memcpy(p.b + at, tmp.data(), n);
            }
        }
    };
    const int n = std::max(1, std::min<int>(threads, (int) pieces.size()));
    std::vector<std::thread> pool;
    for (int t = 1; t < n; ++t) pool.emplace_back([&] { unpin(); run(); });
    run();
    for (auto& t : pool) t.join();
}

} // namespace

void conversation_slot_unpin_thread() { unpin(); }

ConversationIo ConversationSlot::io(void* stream) const {
    ConversationIo io;
    io.stream = stream;
    io.residency = false;
    if (uva && host_mapped != nullptr) {
        io.host_lo = host_mapped;
        io.host_hi = (const uint8_t*) host_mapped + host_bytes;
    }
    return io;
}

bool conversation_slot_bytes(const SessionState& active, const QsaState* draft, const ModelGeometry& g,
                             ConversationSlotBytes& bytes, std::string& error) {
    Carve dev, host;
    ConversationSlot sizing;
    if (!build(active, draft, g, dev, host, sizing, error)) return false;
    bytes.device = dev.used;
    bytes.host = host.used;
    return true;
}

bool conversation_slot_init(ConversationSlot& slot, const SessionState& active, const QsaState* draft,
                            const ModelGeometry& g, std::string& error) {
    conversation_slot_free(slot);
    ConversationSlotBytes need;
    if (!conversation_slot_bytes(active, draft, g, need, error)) return false;
    if (cudaMalloc(&slot.device, std::max<size_t>(need.device, 256)) != cudaSuccess) {
        cudaGetLastError();
        slot.device = nullptr;
        return fail(error, "cannot allocate " + std::to_string(need.device >> 20) + " MiB of VRAM for the standby slot");
    }
    if (need.host) {
        if (cudaHostAlloc(&slot.host, need.host, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer(&slot.host_mapped, slot.host, 0) != cudaSuccess) {
            cudaGetLastError();
            conversation_slot_free(slot);
            return fail(error, "cannot pin " + std::to_string(need.host >> 20) + " MiB of RAM for the standby slot's K/V");
        }
        slot.uva = slot.host_mapped == slot.host;
    }
    slot.device_bytes = need.device;
    slot.host_bytes = need.host;
    Carve dev{(uint8_t*) slot.device}, host{(uint8_t*) slot.host_mapped};
    if (!build(active, draft, g, dev, host, slot, error)) {
        conversation_slot_free(slot);
        return false;
    }
    // Zeroed, so a never-filled slot holds a well-defined (empty) state if it is ever swapped in.
    if (cudaMemset(slot.device, 0, need.device) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        conversation_slot_free(slot);
        return fail(error, "cannot clear the standby slot");
    }
    return true;
}

void conversation_slot_free(ConversationSlot& slot) {
    if (slot.device) cudaFree(slot.device);
    if (slot.host) cudaFreeHost(slot.host);
    slot = ConversationSlot{};
}

bool conversation_slot_move(ConversationSlotMove how, SessionState& active, const QsaState* draft,
                            ConversationSlot& slot, const ModelGeometry& g, int64_t upto, void* stream,
                            std::string& error, ConversationSlotStats* stats, int threads) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!slot.ready()) return fail(error, "no standby slot");
    if ((draft != nullptr) != slot.has_draft) return fail(error, "the standby slot was built for another drafter");
    std::vector<ConversationRange> a, m;
    if (!ranges(active, draft, g, upto, a, error)) return false;
    {
        const ConversationIo io = slot.io(stream);
        ConversationIoScope scope(io);
        if (!ranges(slot.ss, slot.has_draft ? &slot.draft : nullptr, g, upto, m, error)) return false;
    }
    if (a.size() != m.size()) return fail(error, "the standby slot's layout differs from the session's");
    cudaStream_t cs = (cudaStream_t) stream;
    std::vector<std::pair<ConversationRange, ConversationRange>> cpu;
    ConversationSlotStats st;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].bytes != m[i].bytes || a[i].host != m[i].host)
            return fail(error, "the standby slot's layout differs from the session's");
        if (!a[i].bytes) continue;
        if (!a[i].data || !m[i].data) return fail(error, "missing state buffer");
        (a[i].host ? st.host_bytes : st.device_bytes) += a[i].bytes;
        if (a[i].host && slot.uva) { cpu.emplace_back(a[i], m[i]); continue; }
        if (how == ConversationSlotMove::swap) conversation_device_swap(a[i].data, m[i].data, a[i].bytes, stream);
        else if (cudaMemcpyAsync(m[i].data, a[i].data, a[i].bytes, cudaMemcpyDefault, cs) != cudaSuccess)
            return fail(error, std::string("copy: ") + cudaGetErrorString(cudaGetLastError()));
    }
    // the GPU moves the VRAM ranges while the CPU moves the pinned ones
    const int hw = (int) std::max(2u, std::thread::hardware_concurrency());
    const auto c0 = std::chrono::steady_clock::now();
    host_moves(how, cpu, threads > 0 ? threads : std::min(16, hw / 2));
    const auto c1 = std::chrono::steady_clock::now();
    const cudaError_t e = cudaStreamSynchronize(cs);
    st.cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    st.gpu_wait_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c1).count();
    if (e != cudaSuccess) return fail(error, std::string("move: ") + cudaGetErrorString(e));
    if (how == ConversationSlotMove::swap) std::swap(active.ple_prev, slot.ss.ple_prev);
    else std::memcpy(slot.ss.ple_prev, active.ple_prev, sizeof(active.ple_prev));
    st.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (stats) *stats = st;
    return true;
}

bool conversation_slot_refresh(const SessionState& active, const QsaState* draft, const ModelGeometry& g,
                               int64_t upto, void* stream, std::string& error) {
    for (int64_t j = 0; j < active.qsa_alloc; ++j)
        if (!conversation_kv_refresh(active.qsa_states[(size_t) (active.qsa_ord0 + j)], g, upto, stream, error))
            return false;
    if (draft && !conversation_kv_refresh(*draft, g, upto, stream, error)) return false;
    const cudaError_t e = cudaStreamSynchronize((cudaStream_t) stream);
    if (e == cudaSuccess) return true;
    return fail(error, std::string("refresh: ") + cudaGetErrorString(e));
}

// the drafter's K/V included
bool conversation_slot_bytes(const SessionState& active, const QsaState& draft, const ModelGeometry& g,
                             ConversationSlotBytes& bytes, std::string& error) {
    return conversation_slot_bytes(active, &draft, g, bytes, error);
}
bool conversation_slot_init(ConversationSlot& slot, const SessionState& active, const QsaState& draft,
                            const ModelGeometry& g, std::string& error) {
    return conversation_slot_init(slot, active, &draft, g, error);
}
bool conversation_slot_move(ConversationSlotMove how, SessionState& active, const QsaState& draft,
                            ConversationSlot& slot, const ModelGeometry& g, int64_t upto, void* stream,
                            std::string& error, ConversationSlotStats* stats, int threads) {
    return conversation_slot_move(how, active, &draft, slot, g, upto, stream, error, stats, threads);
}
bool conversation_slot_refresh(const SessionState& active, const QsaState& draft, const ModelGeometry& g,
                               int64_t upto, void* stream, std::string& error) {
    return conversation_slot_refresh(active, &draft, g, upto, stream, error);
}

} // namespace strata::core
