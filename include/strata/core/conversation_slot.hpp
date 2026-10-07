// include/strata/core/conversation_slot.hpp - the second VRAM slot: two conversations held on one GPU.
//
// THE PROBLEM.  Two users take turns on one engine.  The session holds ONE conversation's state at fixed
// addresses (every captured graph bakes them in), so the other one is either re-read from token 0 - minutes at
// 128K - or, with --conversation-cache-mib, parked in RAM and restored: ~1.7 GiB of K/V, GDN recurrence and
// indexer rows copied out and in on EVERY switch, in the foreground, while the next user waits.
//
// THE SLOT.  A mirror of exactly the bytes a snapshot captures (conversation_snapshot.hpp), allocated once with
// the same shapes as the active session's, so the active session and the mirror are two permanent slots:
//
//   * VRAM (device): the 36 GDN rows (118 MB), the PLE history, the 12 indexers' tails, spare keys and pooled
//     rows (201 MB at 128K) - and the K/V itself for layers that are fully resident (no --kv-resident);
//   * pinned RAM (host): the authoritative K/V of the streamed layers and the drafter's host copy (1.55 + 0.13
//     GiB at 128K int8).  A streamed layer's VRAM window is NOT mirrored: it is a cache of that host copy, and
//     after a switch it is invalidated and refills on demand (kv_stream.hpp) - ~2 MB per layer per token for the
//     first few windows, against the 415 MB a mirrored window would cost in VRAM.
//
// THE MOVES.  Nothing re-captures a graph, so a switch moves BYTES between the two slots at fixed addresses:
//   * swap: the active conversation and the standby one exchange places (the next turn of the other user);
//   * copy: the active conversation is duplicated into the empty standby slot before a rewind overwrites it
//     (a new chat that mounts through the active one's system-prompt checkpoint).
// Device ranges move with one kernel each at VRAM bandwidth (~1 ms for 320 MB); host ranges are exchanged by CPU
// threads when the pinned memory is addressable at its device address (UVA - checked at allocation), otherwise
// by the same kernel over the mapped pointers.  Only the first `upto` cells of each K/V array move (the identity
// layout makes them a prefix), so a short conversation swaps in proportion to its length.
//
// WHAT FILLS AND DRAINS IT.  The serving loop's slot worker loads a parked RAM image into the mirror and writes
// the mirror back to RAM with the ordinary snapshot functions inside a ConversationIoScope (its own stream, plain
// memcpy for the pinned part), WHILE another request decodes - which is what makes the next switch a swap.
#pragma once

#include "strata/core/conversation_snapshot.hpp"

#include <string>
#include <vector>

namespace strata::core {

struct ConversationSlot {
    SessionState ss;               ///< conversation state only: no scratch, no graphs, no residency
    std::vector<QsaState> qsa;     ///< backs `ss.qsa_states` (global ordinal indexing, as the session's)
    QsaState draft;                ///< the drafter's authoritative K/V (empty without a drafter)
    bool has_draft = false;
    void* device = nullptr;        ///< one VRAM allocation
    void* host = nullptr;          ///< one pinned, device-mapped allocation (null when nothing is streamed)
    void* host_mapped = nullptr;   ///< its device address
    size_t device_bytes = 0, host_bytes = 0;
    bool uva = false;              ///< the pinned memory's device address is its host address: CPU moves
    bool ready() const { return device != nullptr; }
    /// The scope the slot worker loads/saves this mirror in (residency off, pinned range memcpy'd, `stream`).
    ConversationIo io(void* stream) const;
};

struct ConversationSlotBytes {
    size_t device = 0, host = 0;
};
/// What a mirror of `active` + `draft` costs, before anything is allocated (the VRAM plan reads it).
bool conversation_slot_bytes(const SessionState& active, const QsaState& draft, const ModelGeometry& g,
                             ConversationSlotBytes& bytes, std::string& error);
/// Allocate and carve the mirror.  Fails (and allocates nothing) when VRAM or pinned memory is short.
bool conversation_slot_init(ConversationSlot& slot, const SessionState& active, const QsaState& draft,
                            const ModelGeometry& g, std::string& error);
void conversation_slot_free(ConversationSlot& slot);

enum class ConversationSlotMove { swap, copy };
struct ConversationSlotStats {
    size_t device_bytes = 0, host_bytes = 0;   ///< per side
    double ms = 0, cpu_ms = 0, gpu_wait_ms = 0;   ///< the whole move; the pinned part; the VRAM part's tail
};
/// Swap (both ways) or copy (active -> slot) the state of conversations of up to `upto` cells.  For a swap, pass
/// the LONGER of the two.  The caller has synchronized the device: no kernel may touch either slot meanwhile.
/// Runs on `stream` and returns with it synchronized; the active session's residency is NOT refreshed (call
/// conversation_slot_refresh after a swap).  `threads` CPU threads move the pinned ranges (0 = a default).
bool conversation_slot_move(ConversationSlotMove how, SessionState& active, const QsaState& draft,
                            ConversationSlot& slot, const ModelGeometry& g, int64_t upto, void* stream,
                            std::string& error, ConversationSlotStats* stats = nullptr, int threads = 0);
/// After a swap: the active session's VRAM windows forget the outgoing pages, the drafter's ring is refilled for
/// a sequence of `upto` cells.  Synchronizes `stream`.
bool conversation_slot_refresh(const SessionState& active, const QsaState& draft, const ModelGeometry& g,
                               int64_t upto, void* stream, std::string& error);

/// Widen the calling thread to the process's CPU affinity as it was at start-up.  A thread created by the serving
/// thread inherits its pin to the host loop's core (SessionLoopScratch); the slot worker and the move threads must
/// not run there - the worker shares it with the host loop's spin WHILE a request decodes.
void conversation_slot_unpin_thread();

/// The same with `draft == nullptr`: no drafter (no --mtp), so the mirror and the moves hold the session's own
/// state only - as a snapshot taken with a null draft does (conversation_snapshot.hpp).
bool conversation_slot_bytes(const SessionState& active, const QsaState* draft, const ModelGeometry& g,
                             ConversationSlotBytes& bytes, std::string& error);
bool conversation_slot_init(ConversationSlot& slot, const SessionState& active, const QsaState* draft,
                            const ModelGeometry& g, std::string& error);
bool conversation_slot_move(ConversationSlotMove how, SessionState& active, const QsaState* draft,
                            ConversationSlot& slot, const ModelGeometry& g, int64_t upto, void* stream,
                            std::string& error, ConversationSlotStats* stats = nullptr, int threads = 0);
bool conversation_slot_refresh(const SessionState& active, const QsaState* draft, const ModelGeometry& g,
                               int64_t upto, void* stream, std::string& error);

/// Exchange `bytes` between two device-accessible buffers (VRAM, or pinned memory through its mapping).
void conversation_device_swap(void* a, void* b, size_t bytes, void* stream);

} // namespace strata::core
