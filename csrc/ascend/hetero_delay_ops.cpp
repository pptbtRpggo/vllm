// SPDX-License-Identifier: Apache-2.0
// Submit device timing/waits through TorchNPU's queue alongside native ops.
#include <ATen/ATen.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" void launch_mark(void*, void*);
extern "C" void launch_stretch(void*, void*, uint64_t, uint64_t);
extern "C" void launch_wait(void*, uint64_t);

namespace {
void check_buffer(const at::Tensor& buffer) {
  TORCH_CHECK(buffer.device().type() == c10::DeviceType::PrivateUse1 &&
                  buffer.scalar_type() == at::kLong && buffer.is_contiguous() &&
                  buffer.numel() == 3,
              "delay buffer must be a contiguous three-element NPU int64 tensor");
}

void mark(const at::Tensor& buffer) {
  check_buffer(buffer);
  const c10_npu::NPUGuard guard(buffer.device());
  // stream(false) does not drain the CPU submission queue. The handler is
  // queued after preceding native operations and only submits a device kernel.
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto handler = [buffer, stream]() -> int {
    launch_mark(stream, buffer.data_ptr());
    return 0;
  };
  at_npu::native::OpCommand command;
  command.Name("VllmDelayMark").SetCustomHandler(handler).Run();
}

void stretch(const at::Tensor& buffer, int64_t factor, int64_t cycles) {
  check_buffer(buffer);
  TORCH_CHECK(factor >= 0 && cycles >= 0, "delay arguments must be nonnegative");
  const c10_npu::NPUGuard guard(buffer.device());
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto handler = [buffer, stream, factor, cycles]() -> int {
    launch_stretch(stream, buffer.data_ptr(), factor, cycles);
    return 0;
  };
  at_npu::native::OpCommand command;
  command.Name("VllmDelayStretch").SetCustomHandler(handler).Run();
}

void wait(const at::Tensor& anchor, int64_t cycles) {
  check_buffer(anchor);
  TORCH_CHECK(cycles >= 0, "wait cycles must be nonnegative");
  const c10_npu::NPUGuard guard(anchor.device());
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto handler = [anchor, stream, cycles]() -> int {
    launch_wait(stream, cycles);
    return 0;
  };
  at_npu::native::OpCommand command;
  command.Name("VllmDelayWait").SetCustomHandler(handler).Run();
}
}  // namespace

TORCH_LIBRARY(vllm_ascend_delay, m) {
  m.def("mark(Tensor(a!) buffer) -> ()");
  m.def("stretch(Tensor(a!) buffer, int factor, int cycles) -> ()");
  m.def("wait(Tensor anchor, int cycles) -> ()");
}

TORCH_LIBRARY_IMPL(vllm_ascend_delay, PrivateUse1, m) {
  m.impl("mark", &mark);
  m.impl("stretch", &stretch);
  m.impl("wait", &wait);
}
