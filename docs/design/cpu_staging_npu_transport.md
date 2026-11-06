# CPU 中转的 NPU 点对点通信

## 实现范围

`vllm/distributed/device_communicators/cpu_staging.py` 提供显式的
`CpuStagingTransport.isend/irecv`，以及阻塞封装 `send/recv`。

路径为：发送端 NPU → CPU 连续缓冲区 → Gloo → 接收端 CPU → 本机 NPU。
只建立 CPU/Gloo 通信组，不需要两个 NPU 型号组成 HCCL 通信域。
这不改变卡的计算能力，也不修改 CANN、驱动或 PyTorch 的全局接口。

这是通信层实现，不会自动改变默认 Worker、Model Runner 或 GroupCoordinator。
新增的显式可选接入见[异构Ascend PP后端](heterogeneous_ascend_pp.md)；其CPU验证
不等于310P＋910B完整模型推理已经通过实机验证。

## 调用方式

两个进程先通过 `torchrun` 启动，并各自选择本机设备。示意：

```python
import torch
import torch.distributed as dist
import torch_npu
from vllm.distributed.device_communicators.cpu_staging import CpuStagingTransport

# 两个节点都使用 Gloo 初始化；不能将混合型号的 HCCL group 传入。
dist.init_process_group("gloo")
torch.npu.set_device(0)
transport = CpuStagingTransport(dist.group.WORLD)

if dist.get_rank() == 0:
    activation = torch.ones((16, 4096), dtype=torch.float16, device="npu:0")
    work = transport.isend(activation, dst=1, tag=10)
else:
    activation = torch.empty((16, 4096), dtype=torch.float16, device="npu:0")
    work = transport.irecv(activation, src=0, tag=10)
work.wait()
# 此时接收 tensor 已可在本机计算 stream 使用。
transport.close()
dist.destroy_process_group()
```

`src/dst` 始终为全局 rank，包括传入子组时。两端必须事先对齐 shape、dtype、
tag、消息顺序；这些元数据应由上层协议提供。设备编号属于本机，不跨节点传递。

## 完成语义

- `isend` 返回前完成 D2H 快照，返回后原 tensor 可复用；网络传输异步执行。
- `irecv` 先提交 CPU 接收，后台线程等待网络完成，再执行 H2D。
  接收前记录调用 stream 的 event，防止覆盖该 stream 尚在使用的目标存储。
  接收完成状态包含 H2D 和对应 stream 同步。
- `work.wait()` 阻塞 CPU，成功返回 `True`；`is_completed()` 表示操作已终止，
  包括失败。必须调用 `wait()` 或 `close()` 获取异常。
- `wait(timedelta(...))` 超时抛 `TimeoutError`，不会取消操作；可再次等待。
  还应设置 Gloo 通信组的网络超时，以处理失联节点。
- 丢弃 work 句柄不会释放传输中的存储。销毁 process group 前调用 `close()`，
  等待所有传输并检查失败。
- 支持 CPU/NPU 的 dense strided tensor，包括非连续视图、标量、空 tensor。
  网络传送逻辑连续值的字节，不传 NPU 内部物理布局。接收端实际 dtype 必须由
  该设备支持，不能借助此模块让 310P 执行本身不支持的 BF16 算子。
- 面向 eager inference；不提供 autograd、图捕获、取消、Future 或完整 c10d
  Work API。每个在途操作使用一个后台线程，上层应限制在途消息数量。

## 验证命令

独立 CPU/Gloo 测试不需要载入模型：

```bash
python tests/distributed/test_cpu_staging.py
python benchmarks/cpu_staging_smoke.py --device cpu --spawn --bf16 --output /tmp/cpu-check
python benchmarks/cpu_staging_smoke.py --device npu --spawn --bf16 --output /tmp/npu-check
```

`--spawn` 只检查单机两进程。真正两机时，在两台机器运行相同版本的三个文件，
每台启动一个进程，去掉 `--spawn`，并选择两机可达的 Gloo 网卡。例如：

```bash
GLOO_SOCKET_IFNAME=eth0 torchrun \
  --nnodes=2 --nproc-per-node=1 --node-rank=0 \
  --master-addr=HEAD_IP --master-port=29510 \
  benchmarks/cpu_staging_smoke.py --device npu --output /tmp/hetero-check
```

另一台使用 `--node-rank=1`。310P 上不传 `--bf16`，除非已经确认其软硬件支持。
脚本固定全局 backend 为 Gloo，记录设备名、源文件 SHA、各 rank 的正确性结果，
覆盖 prefill/decode 大小的 activation、反向 token、空值、转置视图、NPU format29
和非默认 stream；结果是功能验证，不是模型性能或 SLO 结论。

## 完整异构 PP 的后续接入

需要在 Worker/分布式初始化层使用兼容的 CPU 控制组，并让各节点选择本机正确的
Worker 和支持 PP 的 Runner。PP tensor 字典的元数据、activation 收发、反向
token/结果及必要广播均需接入，不能只替换一处 `irecv`。首版应限制 TP=1、
FP16、eager、关闭 async scheduling，最后用真实 310P＋910B 两机分别验证
prefill、连续 decode 和请求级输出正确性。

`CpuStagingTransport` 不会自动绕过现有 vLLM 启动时的全局 HCCL 初始化，也不
解除某版 310P Runner 对 PP>1 的限制。
