#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <cuda_runtime.h>

namespace strata::core { struct ConversationIo; }

namespace strata::core::conversation_detail {
/// The calling thread's ConversationIoScope, or null (conversation_snapshot.cpp).
const ConversationIo* io();
/// One copy between a state buffer and snapshot storage, as the thread's scope says (see ConversationIo).
cudaError_t io_copy(void* dst, const void* src, size_t bytes);
/// The device synchronisation before/after a capture or restore, as the thread's scope says.
cudaError_t io_sync();

inline bool add(size_t& total, size_t n) {
    if (n > std::numeric_limits<size_t>::max() - total) return false;
    total += n;
    return true;
}
inline bool product(size_t& out, std::initializer_list<uint64_t> factors) {
    size_t value = 1;
    for (uint64_t factor : factors) {
        if (factor > std::numeric_limits<size_t>::max() ||
            (factor && value > std::numeric_limits<size_t>::max() / factor)) return false;
        value *= (size_t) factor;
    }
    out = value;
    return true;
}
} // namespace strata::core::conversation_detail
