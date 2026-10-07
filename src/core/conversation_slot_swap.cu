// src/core/conversation_slot_swap.cu - the in-place exchange behind a slot swap (conversation_slot.hpp).
//
// In place, because a third buffer of the swapped size would cost the VRAM the slot exists to save.  16 B per
// thread when both sides and the length allow it (every range the slot moves does: the carve aligns to 256 B and
// the K/V rows are whole 16 B multiples), one byte per thread otherwise.
#include "strata/core/conversation_slot.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::core {
namespace {

__global__ void swap16_kernel(uint4* a, uint4* b, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 x = a[i];
        a[i] = b[i];
        b[i] = x;
    }
}

__global__ void swap1_kernel(uint8_t* a, uint8_t* b, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint8_t x = a[i];
        a[i] = b[i];
        b[i] = x;
    }
}

} // namespace

void conversation_device_swap(void* a, void* b, size_t bytes, void* stream) {
    if (!bytes) return;
    cudaStream_t cs = (cudaStream_t) stream;
    const bool wide = (((uintptr_t) a | (uintptr_t) b | bytes) & 15) == 0;
    const size_t n = wide ? bytes / 16 : bytes;
    const unsigned blocks = (unsigned) std::min<size_t>(1024, (n + 255) / 256);
    if (wide) swap16_kernel<<<blocks, 256, 0, cs>>>((uint4*) a, (uint4*) b, n);
    else swap1_kernel<<<blocks, 256, 0, cs>>>((uint8_t*) a, (uint8_t*) b, n);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "conversation slot: swap launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

} // namespace strata::core
