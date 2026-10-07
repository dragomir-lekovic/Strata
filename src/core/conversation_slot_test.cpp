// The standby VRAM slot (conversation_slot.hpp) against the snapshot core it mirrors.
//
// Every claim is checked by CAPTURING both sides with conversation_snapshot_save and comparing the images, so the
// slot is held to the same bytes a park/restore round trip produces.  The wrong-but-plausible outcomes it pins:
//   * a swap that moves only one direction (both sides equal the incoming conversation);
//   * a swap that moves only the shorter conversation's extent (the longer one's tail cells are lost) - A and B
//     have different lengths and the swap is asked for the longer one;
//   * a mirror filled through the worker's IO scope that skips the pinned memcpy (its K/V stays zero);
//   * a refresh that forgets the streamed window (page table still names the outgoing pages) or the ring;
//   * the CPU and the mapped-kernel paths for pinned ranges disagreeing (both are run).
#include "strata/core/conversation_slot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
// The same synthetic QSA state conversation_snapshot_test.cpp builds: one layer's pools, its host copy when
// streamed, a ring table for the drafter.
struct Fixture {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device, host;
    std::array<void*,5> sources{};
    std::array<size_t,5> sizes{};

    template<class T> void alloc(T*& p, size_t n, bool pinned = false) {
        if (!n) return;
        void* raw = nullptr;
        if (pinned) {
            cuda_check(cudaHostAlloc(&raw, n, cudaHostAllocMapped));
            host.push_back(raw);
            void* mapped = nullptr;
            cuda_check(cudaHostGetDevicePointer(&mapped, raw, 0));
            p = static_cast<T*>(mapped);
        } else {
            cuda_check(cudaMalloc(&raw, n)); device.push_back(raw); p = static_cast<T*>(raw);
        }
    }
    Fixture(int fmt, int mode) {
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt==kKvInt8; st.kv_q4 = fmt==kKvQ4;
        st.n_pages = 24; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = mode==2 ? 2 : st.max_cells/4+2;
        const size_t per = fmt==kKvQ4 ? kv_q4_bytes_per_head((int)g.head_dim) : g.head_dim*(st.kv_int8 ? 1:2);
        const size_t rows = st.max_cells*g.n_head_kv, slot_rows = st.n_slots*4*g.n_head_kv;
        sizes = {rows*per, rows*per, fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0,
                 fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0, (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
        if (fmt==kKvQ4) {
            alloc(st.k_q4,slot_rows*per); alloc(st.v_q4,slot_rows*per);
            if (mode) {alloc(st.host.k_q4,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);}
            sources[0]=mode?st.host.k_q4:st.k_q4; sources[1]=mode?st.host.v_q4:st.v_q4;
        } else if (fmt==kKvInt8) {
            alloc(st.k_q,slot_rows*per); alloc(st.v_q,slot_rows*per);
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2); alloc(st.v_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true); alloc(st.host.v_scale,sizes[3],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q:st.v_q;
            sources[2]=mode?st.host.k_scale:st.k_scale; sources[3]=mode?st.host.v_scale:st.v_scale;
        } else {
            alloc(st.k_pool,slot_rows*per); alloc(st.v_pool,slot_rows*per);
            if (mode) {alloc(st.host.k_pool,sizes[0],true); alloc(st.host.v_pool,sizes[1],true);}
            sources[0]=mode?st.host.k_pool:st.k_pool; sources[1]=mode?st.host.v_pool:st.v_pool;
        }
        alloc(st.idx_pooled,sizes[4]); sources[4]=st.idx_pooled;
        alloc(st.page_table,st.n_pages*4);
        if (mode==1) {
            auto& m=st.map;
            m.page_table=st.page_table; m.n_blocks=st.n_pages; m.n_slots=st.n_slots;
            alloc(m.slot_block,st.n_slots*4); alloc(m.slot_stamp,st.n_slots*4); alloc(m.slot_ref,st.n_slots*4);
            alloc(m.miss_block,st.n_slots*4); alloc(m.miss_slot,st.n_slots*4); alloc(m.ctl,kKvCtlInts*4);
            cuda_check(cudaMemset(st.page_table,0,st.n_pages*4));   // "resident": a refresh must clear it
        } else if (mode==2) {
            kv_ring_table(st.page_table,st.n_pages,st.n_slots,nullptr);
        }
    }
    void fill(uint8_t salt) {
        for (size_t i=0;i<sources.size();++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j=0;j<data.size();++j) data[j]=(uint8_t)(salt+i*31+j*7+j/257);
            if (!data.empty()) cuda_check(cudaMemcpy(sources[i],data.data(),data.size(),cudaMemcpyDefault));
        }
    }
    ~Fixture() { for (void* p:device) cudaFree(p); for (void* p:host) cudaFreeHost(p); }
};
bool equal(const ConversationKv& a,const ConversationKv& b) {
    return a.k==b.k && a.v==b.v && a.k_scale==b.k_scale && a.v_scale==b.v_scale && a.pooled==b.pooled;
}
bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.kv.size()!=b.kv.size()) return false;
    for (size_t i=0;i<a.kv.size();++i) if (!equal(a.kv[i],b.kv[i])) return false;
    return a.live.gdn==b.live.gdn && a.live.ple==b.live.ple && a.live.tails==b.live.tails &&
           a.live.dead==b.live.dead && a.live.block_pos==b.live.block_pos;
}

void slot_session(int fmt, int mode, bool force_mapped, int threads) {
    Fixture main(fmt,mode), draft(fmt,mode ? 2 : 0);
    auto& g=main.g;
    g.n_layers=4;
    g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    SessionState ss;
    ss.max_cells=96; ss.qsa_states=&main.state;
    ss.layer_hi=g.n_layers; ss.gdn_alloc=g.n_gdn_layers(); ss.qsa_alloc=g.n_qsa_layers();
    std::string err;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g,sizes,err),"geometry sizes");
    main.alloc(ss.gdn_state,sizes.gdn); main.alloc(ss.ple_hist,sizes.ple);
    main.alloc(main.state.idx_tail,sizes.tail); main.alloc(main.state.idx_dead,sizes.dead);
    main.alloc(main.state.idx_block_pos,sizes.block_pos);

    ConversationSlotBytes need;
    check(conversation_slot_bytes(ss,draft.state,g,need,err),"price the standby slot");
    check(need.device >= sizes.gdn + sizes.ple + sizes.tail + sizes.dead + main.sizes[4],
          "the slot's VRAM covers the running state and the indexer rows");
    check((need.host != 0) == (mode != 0),"pinned K/V exactly when the layers stream");
    ConversationSlot slot;
    check(conversation_slot_init(slot,ss,draft.state,g,err),"allocate the standby slot");
    if (force_mapped) slot.uva = false;   // the mapped-kernel path for pinned ranges

    // A: 65 tokens, B: 70 - different extents, so a swap sized by the shorter one loses B's tail
    std::vector<int32_t> ids_a(65), ids_b(70);
    for (size_t i=0;i<ids_a.size();++i) ids_a[i]=(int32_t)i+1;
    for (size_t i=0;i<ids_b.size();++i) ids_b[i]=(int32_t)i+1001;
    const std::vector<ConversationImageKey> images;
    const std::vector<ConversationCheckpoint> none;
    const ConversationView view_a{ids_a,images,none,true}, view_b{ids_b,images,none,true};
    auto fill=[&](uint8_t salt, size_t tokens) {
        main.fill(salt); draft.fill(salt);
        for (const auto& [p,n] : std::vector<std::pair<void*,size_t>>{
                 {ss.gdn_state,sizes.gdn},{ss.ple_hist,sizes.ple},{main.state.idx_tail,sizes.tail},
                 {main.state.idx_dead,sizes.dead},{main.state.idx_block_pos,sizes.block_pos}})
            cuda_check(cudaMemset(p,salt,n));
        cuda_check(cudaMemcpy(main.state.idx_pooled+(tokens/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        cuda_check(cudaDeviceSynchronize());
    };
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    const ConversationIo io = slot.io(stream);
    auto capture_active=[&](const ConversationView& v) {
        SavedConversation s;
        check(conversation_snapshot_save(s,v,ss,g,draft.state,err),"capture the active session");
        return s;
    };
    auto capture_slot=[&](const ConversationView& v) {
        SavedConversation s;
        ConversationIoScope scope(io);
        check(conversation_snapshot_save(s,v,slot.ss,g,slot.draft,err),"capture the standby slot through the worker scope");
        return s;
    };

    fill(13,ids_a.size());
    const SavedConversation a = capture_active(view_a);
    fill(177,ids_b.size());
    const SavedConversation b = capture_active(view_b);
    check(!same(a,b),"the fixture's two conversations differ");

    {   // the worker's load: a parked image into the mirror, without touching the active session
        ConversationIoScope scope(io);
        check(conversation_snapshot_restore(a,slot.ss,g,slot.draft,err)==ConversationRestore::restored,
              "load A into the standby slot (no window, no ring)");
    }
    check(same(capture_active(view_b),b),"loading the slot leaves the active session alone");
    check(same(capture_slot(view_a),a),"the slot holds A byte for byte");
    check(slot.ss.ple_prev[0]==64 && slot.ss.ple_prev[1]==65,"the slot's PLE window is A's");

    // swap: B (70) is the longer one, so the swap is sized by it
    ConversationSlotStats stats;
    check(conversation_slot_move(ConversationSlotMove::swap,ss,draft.state,slot,g,70,stream,err,&stats,threads),
          "swap A in");
    check(conversation_slot_refresh(ss,draft.state,g,65,stream,err),"refresh the active residency");
    check(same(capture_active(view_a),a),"after the swap the active session is A");
    check(same(capture_slot(view_b),b),"after the swap the slot is B, tail cells included");
    check(ss.ple_prev[0]==64 && ss.ple_prev[1]==65 && slot.ss.ple_prev[0]!=64,"the PLE window moved with the state");
    check(stats.device_bytes>0 && (mode==0 || stats.host_bytes>0),"the move reports what it moved");
    if (mode==1) {
        std::vector<int32_t> table((size_t)main.state.n_pages);
        cuda_check(cudaMemcpy(table.data(),main.state.page_table,table.size()*4,cudaMemcpyDeviceToHost));
        check(std::all_of(table.begin(),table.end(),[](int32_t x){return x==-1;}),"the outgoing window pages are forgotten");
        uint64_t fingerprint=0;
        check(conversation_kv_verify(a.kv.back(),draft.state,g,65,false,fingerprint,err),"the drafter's ring holds A's pages");
    }

    check(conversation_slot_move(ConversationSlotMove::swap,ss,draft.state,slot,g,70,stream,err,nullptr,threads),
          "swap B back");
    check(conversation_slot_refresh(ss,draft.state,g,70,stream,err),"refresh again");
    check(same(capture_active(view_b),b) && same(capture_slot(view_a),a),"A/B/A: both survive two swaps");

    // copy: the active conversation preserved in the slot before a rewind overwrites it
    check(conversation_slot_move(ConversationSlotMove::copy,ss,draft.state,slot,g,70,stream,err,nullptr,threads),
          "copy B into the slot");
    check(same(capture_slot(view_b),b) && same(capture_active(view_b),b),"copy duplicates and leaves the source");
    fill(91,ids_a.size());
    check(same(capture_slot(view_b),b),"a later rewrite of the active session does not reach the copy");

    {   // the worker's write-back reads the mirror while the active session holds something else
        SavedConversation back = capture_slot(view_b);
        check(same(back,b),"write-back of the slot equals the conversation it holds");
    }
    cuda_check(cudaStreamDestroy(stream));
    conversation_slot_free(slot);
    check(!slot.ready(),"free releases the slot");
}

void device_swap_shapes() {
    for (size_t n : {size_t(0), size_t(1), size_t(15), size_t(16), size_t(4097), size_t(1) << 20}) {
        std::vector<uint8_t> x(n+3), y(n+3);
        for (size_t i=0;i<x.size();++i) { x[i]=(uint8_t)(i*3+1); y[i]=(uint8_t)(i*5+2); }
        uint8_t *a=nullptr,*b=nullptr;
        cuda_check(cudaMalloc(&a,n+3)); cuda_check(cudaMalloc(&b,n+3));
        cuda_check(cudaMemcpy(a,x.data(),x.size(),cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(b,y.data(),y.size(),cudaMemcpyHostToDevice));
        for (size_t off : {size_t(0), size_t(1)}) {   // aligned (wide kernel) and misaligned (byte kernel)
            cuda_check(cudaMemcpy(x.data(),a,x.size(),cudaMemcpyDeviceToHost));
            cuda_check(cudaMemcpy(y.data(),b,y.size(),cudaMemcpyDeviceToHost));
            conversation_device_swap(a+off,b+off,n,nullptr);
            cuda_check(cudaDeviceSynchronize());
            std::vector<uint8_t> ga(x.size()), gb(y.size());
            cuda_check(cudaMemcpy(ga.data(),a,ga.size(),cudaMemcpyDeviceToHost));
            cuda_check(cudaMemcpy(gb.data(),b,gb.size(),cudaMemcpyDeviceToHost));
            bool ok=true;
            for (size_t i=0;i<x.size();++i) {
                const bool in = i>=off && i<off+n;
                ok = ok && ga[i]==(in ? y[i] : x[i]) && gb[i]==(in ? x[i] : y[i]);
            }
            check(ok,"device swap exchanges exactly the range");
        }
        cudaFree(a); cudaFree(b);
    }
}
}

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    device_swap_shapes();
    for (int fmt : {kKvF16,kKvInt8,kKvQ4}) for (int mode : {0,1}) {
        slot_session(fmt,mode,false,4);
        slot_session(fmt,mode,true,1);
    }
    std::printf("conversation_slot_test: %d checks passed\n",checks);
}
