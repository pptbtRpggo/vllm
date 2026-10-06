// SPDX-License-Identifier: Apache-2.0
// Experimental stream-ordered delay for Ascend 910B (50 MHz system counter).
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

__aicore__ inline void Flush(GlobalTensor<int64_t>& values) {
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE>(values);
}

extern "C" __global__ __aicore__ void mark_clock(GM_ADDR address) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int64_t> values;
    values.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(address), 3);
    values.SetValue(0, GetSystemCycle());
    Flush(values);
}

extern "C" __global__ __aicore__ void stretch_clock(
    GM_ADDR address, float factor, uint64_t fixed_cycles) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<int64_t> values;
    values.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(address), 3);
    Flush(values);
    int64_t started = values.GetValue(0);
    int64_t end = GetSystemCycle();
    uint64_t delay = static_cast<uint64_t>((end - started) * factor) + fixed_cycles;
    while (static_cast<uint64_t>(GetSystemCycle() - end) < delay) {}
    values.SetValue(1, end);
    values.SetValue(2, GetSystemCycle());
    Flush(values);
}

extern "C" __global__ __aicore__ void wait_clock(uint64_t cycles) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    int64_t started = GetSystemCycle();
    while (static_cast<uint64_t>(GetSystemCycle() - started) < cycles) {}
}

extern "C" void launch_mark(void* stream, void* address) {
    mark_clock<<<1, nullptr, stream>>>(static_cast<uint8_t*>(address));
}
extern "C" void launch_stretch(void* stream, void* address, float factor,
                                uint64_t fixed_cycles) {
    stretch_clock<<<1, nullptr, stream>>>(static_cast<uint8_t*>(address),
                                         factor, fixed_cycles);
}
extern "C" void launch_wait(void* stream, uint64_t cycles) {
    wait_clock<<<1, nullptr, stream>>>(cycles);
}
