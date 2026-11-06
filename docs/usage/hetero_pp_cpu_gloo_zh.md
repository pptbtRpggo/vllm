# 两节点四卡 PP：手动启动与 CPU/Gloo 通信验证

本文对应从官方 **vLLM v0.23.0** 整理出的
`codex/hetero-pp-v023` 分支，配合 vLLM-Ascend **0.23.0.post1**。
它说明如何启动现有功能，以及用什么证据判断 tensor 确实经过 CPU/Gloo。
本轮重新验证的最终结果和原始记录以验收报告为准；本文不把旧测试当作新测试。

## 先弄清楚启动的是什么

两台 Ray 节点各提供两张 910B，每张卡负责一个 PP stage，共四个 stage。
每个 stage 内 TP=1。模型按**层**切开，每个 stage 只执行自己的连续层区间。
以 32 层 CodeLlama-7B-Instruct 为例：

| PP rank | 所在节点 | 方案 A 层数 / 区间 | 方案 B 层数 / 区间 |
|---|---|---|---|
| 0 | hw | 12 / `[0,12)` | 5 / `[0,5)` |
| 1 | hw | 8 / `[12,20)` | 7 / `[5,12)` |
| 2 | hw2 | 7 / `[20,27)` | 8 / `[12,20)` |
| 3 | hw2 | 5 / `[27,32)` | 12 / `[20,32)` |

这里的区间只统计 transformer 层；embedding、最终 norm 和输出 head
仍按原生模型的首末 stage 规则放置。因此“层数相同”不等于实际工作量相同。

**Ray 负责把四个 worker 放到指定节点、分配设备和传递调度指令；Gloo
负责 stage 之间 tensor 的 CPU 载荷。** tensor 的路径是：

```text
本 stage 的 NPU tensor
    → 复制成独立 CPU buffer（D2H）
    → Gloo 发送 CPU 字节
    → 对端 CPU buffer 接收完成
    → 复制到对端 NPU（H2D）并等待完成
    → 下一个 stage 才计算
```

本实现一次只允许一个 scheduler batch 在执行，先保证功能和完成依赖正确。
它没有承诺四张卡始终同时忙碌，也没有承诺优于原生 PP 的性能。
910B 测试不能替代 310P、混合 SoC 或不同物理主机网络的验收。

## 1. 两端都先准备好环境

下面用已有匹配的虚拟环境，不安装或替换驱动、CANN、TorchNPU。
`REPO` 必须指向这次 v0.23 分支的**可运行源码树**，两端源码必须一致。
裸 `git clone` 后只设置 `PYTHONPATH` 不一定足够：源码树仍需要与环境匹配的
版本文件及所需构建产物。请使用已配置好的隔离测试源码树，或按 Ascend
对应版本的安装流程准备；不要把旧 v0.13 工作树混入 `PYTHONPATH`。

两端各自打开终端，设置自己的路径和 IP：

```bash
export REPO=/path/to/vllm-v023
export VENV=/path/to/matched/venv
export PY="$VENV/bin/python"
export MODEL=/same/path/to/CodeLlama-7b-Instruct/model

# 当前两个容器的例子：hw 为 172.20.183.252，hw2 为 172.20.183.225。
# 重新创建实例后先重新核实，不要沿用过期地址。
export NODE_IP=172.20.183.252   # hw2 上改为 172.20.183.225
export HEAD_IP=172.20.183.252

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
SITE="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SITE/vllm_ascend:$LD_LIBRARY_PATH"
export GLOO_SOCKET_IFNAME=eth0
export VLLM_HOST_IP="$NODE_IP"
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

`eth0` 必须是两端互通的接口；换机器后需重新确认。各端必须使用自己的
本地 CANN/plugin 环境，不能把未来 910B 节点的库直接复制给 310P。

本次容器映射中，hw 对应物理卡 0/1，hw2 对应物理卡 2/3，
但**两个容器内部的可见逻辑设备号都是 0/1**。hw2 不能因此设置成 `2,3`。
Ray 为每个 actor 分配一张卡后，actor 在自己的隔离视图中使用逻辑设备 0。

先检查实际导入位置、版本和设备数：

```bash
"$PY" - <<'PY'
import importlib.metadata
import torch
import torch_npu
import vllm

print("vLLM source:", vllm.__file__)
print("vLLM:", vllm.__version__)
print("vLLM-Ascend:", importlib.metadata.version("vllm-ascend"))
print("visible NPU count:", torch.npu.device_count())
assert torch.npu.device_count() == 2
PY
npu-smi info
```

`vllm.__file__` 应在这次源码树内；不能只看 package metadata 的版本号。
两张卡必须真实存在且空闲。下面声明 Ray `NPU:2` 的依据就是这个检查，
不能在实际只有一张卡或没有卡时虚报资源。

两端需要相同的模型配置、tokenizer 和权重。可使用每端各一份、同路径且
逐文件 SHA 相同的副本，也可使用双方真正可访问的共享模型目录。
本次 hw/hw2 的 userdata 是共享存储；两端看到同一份文件并不构成两个独立备份。

## 2. 终端一：在 hw 启动 Ray head

先执行上面的环境块，然后运行：

```bash
"$PY" -m ray.scripts.scripts start \
  --head --node-ip-address="$NODE_IP" --port=16379 \
  --num-cpus=8 --num-gpus=0 --resources='{"NPU":2}' \
  --object-store-memory=268435456 \
  --node-manager-port=16380 --object-manager-port=16381 \
  --min-worker-port=16400 --max-worker-port=16479 \
  --include-dashboard=false --disable-usage-stats --block
```

这条命令启动 Ray 编排服务，还没有加载大模型。
端口仅为本例；启动前检查没有与现有任务冲突。除 head 端口外，
Ray worker 端口和 Gloo 动态 TCP 连接也必须互通。

## 3. 终端二：在 hw2 加入 Ray 集群

同样先执行环境块，确认这里的 `NODE_IP` 是 hw2 的地址：

```bash
"$PY" -m ray.scripts.scripts start \
  --address="$HEAD_IP:16379" --node-ip-address="$NODE_IP" \
  --num-cpus=8 --num-gpus=0 --resources='{"NPU":2}' \
  --object-store-memory=268435456 \
  --node-manager-port=16380 --object-manager-port=16381 \
  --min-worker-port=16400 --max-worker-port=16479 \
  --disable-usage-stats --block
```

这一步把 hw2 的两张卡注册给同一个 Ray 集群，仍未加载模型。
不要在不清楚已有任务归属时运行全局 `ray stop` 或 `pkill`。

## 4. 终端三：在 hw 启动四 stage API 服务

执行 hw 的环境块，并检查集群：

```bash
export RAY_ADDRESS="$HEAD_IP:16379"
"$PY" -m ray.scripts.scripts status --address="$RAY_ADDRESS"
```

应看到两个存活节点、合计四个 NPU 资源。然后创建方案 A：

```bash
export PLAN=/path/to/layout_a.json
cat > "$PLAN" <<EOF
{
  "validation_mode": "910b_only",
  "timeout_s": 420,
  "stages": [
    {"node_ip": "$HEAD_IP", "soc": "910B", "layers": 12},
    {"node_ip": "$HEAD_IP", "soc": "910B", "layers": 8},
    {"node_ip": "172.20.183.225", "soc": "910B", "layers": 7},
    {"node_ip": "172.20.183.225", "soc": "910B", "layers": 5}
  ]
}
EOF
export VLLM_HETERO_PP_PLAN="$(cat "$PLAN")"

"$PY" -m vllm.distributed.hetero_pp.serve \
  --model "$MODEL" --served-model-name codellama7b \
  --pipeline-parallel-size 4 --tensor-parallel-size 1 \
  --dtype float16 --enforce-eager --no-async-scheduling \
  --max-model-len 512 --max-num-batched-tokens 256 --max-num-seqs 4 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --num-gpu-blocks-override 64 --gpu-memory-utilization 0.5 \
  --host 127.0.0.1 --port 18080 2>&1 | tee hetero_pp_api.log
```

方案中 hw2 的地址要随实际环境修改。API 必须运行在方案的第一个节点。
这条命令才会创建四个 Ray actor、初始化 Gloo、加载对应层并启动 HTTP。
这里固定 64 个 KV blocks 是功能检查配置，不是此前研究实验的配置或性能调优建议。

请用这个专用 Python 入口。直接给普通 `vllm serve` 传执行器类名字符串，
不足以通过 v0.23 的 PP 准入检查；专用入口会传入真正的执行器类。
不要额外传 `--distributed-executor-backend`，也不需要 `--nnodes`。

启动日志的 `Heterogeneous PP startup reports` 应包含四个 rank：
节点顺序是 hw/hw/hw2/hw2，真实层区间与表中一致，
`world_backend` 为 `gloo`，源码 SHA 一致。加载的层区间不一致会直接拒绝继续。

要验证方案 B，先正常停止 API，确认原 actor 退出且卡空闲，再把层数改为
`5,7,8,12` 并重新启动 API。**不要并行启动两套四卡服务。**

## 5. 发送请求并检查输出

服务启动后，在 hw 的另一个 shell 中发送：

```bash
curl --fail http://127.0.0.1:18080/health
curl --fail http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "codellama7b",
    "prompt": "The capital of France is",
    "temperature": 0,
    "max_tokens": 16,
    "ignore_eos": true,
    "return_token_ids": true
  }' > response.json
```

HTTP 200 和能读懂的文字只能证明服务有响应。正确性检查还要核对：

- `usage.completion_tokens` 为 16，返回的完整 `token_ids` 有 16 个。
- 同一模型、相同输入 IDs、相同采样参数下，完整输出 IDs 与单卡参照一致。
- 单请求和 batch=4 都检查，并覆盖会触发 chunked prefill 的较长输入。
- 两种层划分都检查；退出时不能只有结果文件成功、后台 actor 已异常死亡。

这类少量 greedy 请求是功能检查，不代表模型质量评估或性能测试。

## 6. 如何证明实际走的是 CPU/Gloo

需要三个层次的证据，不能只看 `GLOO_SOCKET_IFNAME=eth0`。

### 第一层：核对真实执行路径

相关代码按顺序是：

1. `vllm/distributed/hetero_pp/worker.py`：在调用下一 stage 的计算前，
   先接收前一 stage 的 `IntermediateTensors`；离开本 stage 前等待发送完成。
2. `vllm/distributed/hetero_pp/channel.py`：传 tensor dictionary 的
   key、dtype、shape 和序号；tensor 数值交给 `CpuStagingTransport`。
3. `vllm/distributed/device_communicators/cpu_staging.py`：
   构造时要求显式 group 的 backend 为 `gloo`，不接受 HCCL。
   `isend()` 创建 `device="cpu"` 的独立 buffer，并完成源 tensor 的 D2H；
   `_wire()` 把它视作连续 CPU `uint8` 字节，再调用 `dist.isend()`。
   `irecv()` 同样把 `dist.irecv()` 的目标设为 CPU buffer，随后完成 H2D。
   接收 `wait()` 返回前会同步这次目标复制，消费者不会读到未完成数据。
4. `vllm/distributed/hetero_pp/groups.py`：创建 CPU/Gloo group 时禁用
   device communicator。stage tensor 没有交给 HCCL 或 Ray tensor 序列化发送。

两端虚拟地址无须相同，也不会作为跨节点通信协议传给对方。
协议传逻辑 shape、dtype 和数值；目标 tensor 由接收方在自己的 NPU 上分配。

### 第二层：记录 Gloo 边界收到的实际 tensor

运行本轮附带的独立通信 probe 时，要在 `dist.isend/irecv` 的实际调用边界
记录并断言：`tensor.device.type == "cpu"`、
`dist.get_backend(group) == "gloo"`，以及 rank、peer、tag、dtype、字节数。
检查 tensor 本身是 CPU tensor，比只打印 WORLD backend 更有说服力。

这个 probe 应使用真实 NPU 输入和输出，走同一个 `TensorDictChannel` 和
`CpuStagingTransport`。只在 CPU 上运行的 fixture 不能证明 D2H/H2D 正确。
观察包装器必须完整委托原方法；本轮探针不应用于推导性能开销。

提供的入口是 `examples/online_serving/hetero_pp/verify_cpu_gloo.py`。
先停止 API 并确认四卡空闲，然后在两端分别执行。这里的 `--node-rank`
指 Ray 之外这一次 torchrun 诊断的节点编号；该诊断不需要 Ray：

```bash
# hw：先执行第 1 节的本机环境块。
"$PY" -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=2 --node-rank=0 \
  --master-addr="$HEAD_IP" --master-port=19090 \
  "$REPO/examples/online_serving/hetero_pp/verify_cpu_gloo.py" \
  --output /path/to/new-cpu-gloo-receipts

# hw2：在另一个终端执行自己的环境块，再运行相同命令但 node-rank=1。
"$PY" -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=2 --node-rank=1 \
  --master-addr="$HEAD_IP" --master-port=19090 \
  "$REPO/examples/online_serving/hetero_pp/verify_cpu_gloo.py" \
  --output /path/to/new-cpu-gloo-receipts
```

这两条命令分别在两个终端运行，不能在 hw 的同一个 shell 顺序等待它们。
端口 19090 必须空闲且两端可达。每端输出自己的两个 rank 回执；
若目录不共享，将四份回执收集到一起核对。`cpu_gloo_rank0.json` 到
`cpu_gloo_rank3.json` 均需 `passed=true`，每份四项 `checks` 全部正确，
所有 `wire_calls` 都需 `device=cpu`、`backend=gloo`。
其中 rank1 的 `isend` 目标 peer=2、rank2 的 `irecv` 来源 peer=1，
对应真正跨两个容器节点的 stage 边界。

本探针只在自己的诊断进程中包装 `dist.isend/irecv`，记录实参后完整调用
原方法；它不修改服务代码，也没有给运行中的模型打补丁。服务与探针的
`TensorDictChannel`、`CpuStagingTransport` 源码 SHA 必须相同，才能将
这份通信检查与模型路径核对结合起来使用。

### 第三层：完成依赖与内容核对

至少检查以下内容：

- `hidden_states`、`residual`、整数 token、空 tensor、`None` 均正确。
- 非连续 view 按逻辑数值传输；发送后覆盖源 tensor，接收结果仍正确，
  证明发送持有了独立快照。
- 接收 `wait()` 后再在 NPU 消费，完整内容逐值匹配；最后 stage 向首 stage
  返回 token 的反向路径也检查。
- 两端 torchrun 进程均正常退出，四个 rank 的完整检查回执齐全。
- 真模型的层归属、源码 SHA、完整输出 token IDs 和退出检查也通过。

代码路径、边界观察和真实内容检查结合，才支持“这套 tensor 通信经过
CPU/Gloo”的结论。仅设置环境变量、Ray 显示四张卡，或 HTTP 能出 token，
都不能单独证明这一点。

## 7. 停止和本次验证的边界

先对 API 前台进程按 Ctrl-C，等待请求收尾、通信完成和 actor 退出。
再在各自终端停止**本次自己启动**的 Ray 服务；不要清理其他人的 Ray 任务。
最后核对 API/Ray 端口、本次进程和两端 `npu-smi info`，确认资源释放。

当前 hw/hw2 是同一物理宿主机上的两个容器，使用四张不同的物理 910B3。
因此即使所有功能检查通过，也只验证了这个拓扑的跨容器 CPU/Gloo 路径。
它不证明跨物理机网卡性能，不证明指定带宽，不证明 310P 兼容性，
也没有加入软件等待或物理网络限速。
