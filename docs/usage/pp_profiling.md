# PP profiling：从 serving 样本到 partition

DP 的计算、通信成本来自目标设备上的实测 trace。默认使用 `shape-affine`
计算拟合和 `--comm-source serving` 通信计时，不根据 mock 算力、带宽或延迟
配置生成成本。模拟异构只改变 worker 实际执行，真实异构设备不需要设置模拟变量。

## 完整流程

1. 从目标服务日志抽取 requests，保留 prompt、生成参数、到达间隔和并发设置。
2. 在目标设备上启动独立 profiling 服务；先 warmup，再采集 scheduler 实际组成
   的 microbatch。保持目标服务的 model、dtype、TP、cache 和调度设置一致。
3. 保持设备映射和 PP size 不变，更换连续层 partition，重复采集。每个 rank 至少
   测两种层数，建议三种以上，以便检查对未参与拟合的 partition 的预测误差。
4. 用主 trace 的实际 microbatch 分布作为权重，拟合计算成本并统计通信均值。
   同时准备目标 serving 配置下的内存容量上界。
5. 分别执行 latency 和 throughput DP。两个目标如果对应不同的并发/到达负载，
   应分别采样和拟合，不能把一个负载的最优结果当作另一个负载的最优结果。
6. 用没有参与 profiling 的请求，关闭 tracing，启动 DP 推荐、均分和相邻方案。
   每次启动都 warmup，改变方案顺序并重复启动测量。报告实测候选排名；DP 的
   预测最优不代表 vLLM serving 的全局最优。

当前实现针对同构结构的 decoder layers、固定设备顺序和固定 PP size，不实现
EdgeShard 的任意设备选择。异构硬件可以有不同的实测系数；模型结构不同的层
不能无条件共用一个每层系数。

## 请求采样和 trace

输入 JSONL：

```json
{"at_s": 0.0, "request": {"prompt": "Explain pipeline parallelism.", "max_tokens": 64, "temperature": 0}}
{"at_s": 0.2, "request": {"prompt": "Explain KV cache.", "max_tokens": 32, "temperature": 0}}
```

`prompt` 可以是字符串或 token ID 数组。工具保留 EOS 和生成参数；不把真实请求
强制改成固定生成长度。它使用 completions 接口，chat/multimodal 流量必须先按目标
服务真实模板处理。`--sample-size` 从文件中选取连续请求窗口并平移起始时间，保留
窗口内到达间隔；随机丢弃单条请求会改变到达速率，因此不采用这种抽样方式。

```bash
# 仅在独立、受控的 profiling 服务开启管理 RPC。
export VLLM_SERVER_DEV_MODE=1
export VLLM_PP_TRACE_SESSION=unique-session-for-this-server
export VLLM_PP_STAGE_TRACE=/tmp/pp-profile/base
export VLLM_PP_LAYER_PARTITION=8,8,8,8
vllm serve <model> --host 127.0.0.1 --pipeline-parallel-size 4 \
  --tensor-parallel-size 1

python -m vllm.distributed.pp_serving_profile requests.jsonl \
  --model <model> --concurrency 16 --warmup-rounds 2 \
  --sample-size 100 --seed 42 --output /tmp/pp-profile/base/client.json
```

Ascend 加 `--worker-cls vllm.v1.worker.pp_ascend_worker.PPAscendWorker`。
所有 workers 的 session 必须相同，并且每次启动必须更换；每种 partition 使用独立
目录。采样客户端记录实际 dispatch 和完成时间，便于检查并发上限是否改变提交节奏。
两轮 warmup 不等于已经证明性能稳定，最终评估仍需检查重复轮次。

## 计算成本

计时包围 worker 的 `model_runner.execute_model`，到设备同步完成结束。
包含这个调用中的输入准备、decoder、embedding 和末端 logits 计算等工作；
不包含单独调用的 `sample_tokens`，也不等于 request 的端到端 latency。

记录每个 request 本轮的 query tokens、已计算 context 和 prompt tokens。
因此可以表示 mixed microbatch，不依赖 batch 级的 prefill/decode 二选一标签。
默认 `--workload all` 包含全部正式 microbatch；其余过滤条件仅用于诊断。

模型是：

```text
stage_ms(n, microbatch) = fixed(microbatch) + n * layer(microbatch)
```

如果参考 microbatch shape 在多种层数中都有测量，直接拟合相同 shape 的平均耗时。
否则使用非负最小二乘：输入为常数项、request 数、prompt tokens、generation tokens、
以及包含 context 的 causal query-key 对数；固定项和每层项分别学习系数。
这些量是描述实际 workload 的特征，系数全部从实测耗时学习，不是设备配置公式。
每份 profile 的总训练权重相同，避免 scheduler 多产生 microbatch 就增加其训练权重。

最后用主 trace 的 microbatch 出现比例计算期望固定成本和期望每层成本。
每层成本是 homogeneous decoder 的边际成本估计，不是声称所有层实测耗时完全相同。
固定项也不是单独测出的 embedding/LM head 耗时，不应作因果解释。

不再要求不同 partition 重现完全一样的 microbatch，但仍必须有不同层数的测量。
DP 只在实测层数范围内搜索。JSON 保存系数、训练误差、设计矩阵 rank 和按 profile
留出验证的误差；只有两种层数时通常无法做这种留出验证。低训练误差不代表最终
serving 排名准确；特征不足、样本偏移、硬件干扰和调度反馈仍会造成误差。

## 通信成本

默认配对同一个 microbatch 的相邻 stage：

```text
D = recv_end_ns - max(send_start_ns, recv_start_ns)
```

start 在 Python 通信调用前，end 在调用返回并完成设备同步后。这个区间扣除了
两端进入函数的时间差，仍包含 metadata、接收端分配、backend 调度和同步。
它不是设备级 wire time，也不保证与 partition 无关。trace 同时保留 send_end，
因为接收端完成和发送端释放资源的时间可以不同。

配对检查 session、step、microbatch hash、传输字节数和时钟来源。当前支持同机
Linux、TP=1；跨主机时间不可直接相减，时钟来源未知或配对失败就报错。
mock 通信延迟位于计时窗口外，所以该模式拒绝非 1 的通信减速倍数。

独立通信测量保留作显式选择，适用于不能比较两端时钟的情况和实验对照：

```bash
python -m vllm.distributed.pp_serving_profile requests.jsonl \
  --model <model> --concurrency 16 --warmup-rounds 2 --measure-links \
  --output /tmp/pp-profile/base/client.json
# 对应 DP 使用 --comm-source replay
```

这条路径按 trace 的实际 tensor 布局，在同样设备和通信路径上重复测量；当前也是
TP=1。默认不再要求额外通信回放。没有任何模式会退回带宽公式或直接使用含对端
等待的原始 send_ms。

## DP 和验证

```bash
vllm pp-profile --skip-run --trace-dir /tmp/pp-profile/base \
  --fit-trace-dir /tmp/pp-profile/split-a \
  --fit-trace-dir /tmp/pp-profile/split-b \
  --memory-profile /path/to/memory.json --warmup-steps 0 \
  --objective throughput
```

`latency` 最小化各 stage 期望计算成本与各链路期望通信成本之和。
`throughput` 最小化最大 stage 的 `E[recv + compute + send]`，包括发送端占用。
若多个 partition 的瓶颈成本相同，再选择串行总成本最小的方案。实现使用两遍
DP：先求最小瓶颈，再在该瓶颈上界内最小化总成本。不能只给第一遍 DP 的局部
并列状态加排序，因为后续 stage 可能掩盖此前不同的瓶颈。
它是平均资源占用近似，不是 `E[max(stage耗时)]`，也不是 vLLM 调度模拟器。
生成过程的反馈、pipeline 填充/排空和 shared-link 竞争仍需最终 serving 实测。

固定设备顺序下，固定开销之和不会改变 latency 的切分排名，但会影响总耗时预测；
throughput 中固定开销会影响哪个 stage 是瓶颈，因此必须保留。

只想采集 synthetic smoke-test trace 时仍可用 `vllm pp-profile ... --collect-only`。
该模式不要求 memory profile，也不生成可部署方案。单次采集不能自动辨别固定开销
和每层成本，不能省略多种层数的 profiling。

## 内存实测与容量检查

worker RPC `get_pp_memory_observation` 在请求结束后读取每层实际 tensor storage、
非 decoder 层 storage、allocator 已分配/保留量及峰值、设备空闲和总内存。
共享 storage 去重。它提供实际测量依据，但 allocator 峰值包含已分配 KV pool，
不能再把峰值和完整 KV 估算相加。非 PyTorch 分配和未见过的候选 workload 也不能
仅凭这个接口推定，因此完整容量预算仍需要显式给出。

## Memory feasibility

The planner now requires a `--memory-profile /path/to/profile.json`, or a
`pp_memory_profile.json` sidecar in the trace directory. Both Python APIs and
CLIs reject missing bounds by default. The explicit `--allow-unchecked-memory`
escape hatch is for timing-only analysis; its output is marked unchecked and
has no generated serve command. An invalid supplied profile is never silently
ignored, even when that flag is present.

For every candidate interval `[start_layer, end_layer)` on every PP/TP device:

```text
required = sum(layer_weights_bytes[start:end])
         + sum(layer_kv_bytes[start:end])
         + runtime_bytes + activation_bytes + workspace_bytes
         + communication_bytes + graph_bytes + safety_margin_bytes
         + first_stage_bytes  (rank 0 only)
         + last_stage_bytes   (last selected rank only)
required <= budget_bytes
```

This check occurs before every DP transition for latency, blocking throughput,
and ideal-overlap throughput, including the one-rank case. All TP ranks must
fit, not only the TP0 timing representative. Dropping a trailing PP rank moves
the head reserve onto the new last rank. A one-rank candidate charges both
endpoint reserves. If no partition fits, planning raises an actionable error
instead of returning the fastest infeasible candidate. Memory costs are not
multiplied by compute or communication slowdown factors.

**These are checks against supplied bounds, not automatic hardware profiling or
an unconditional OOM guarantee.** Existing stage timing cannot establish memory
for layers moved onto a device or for new graph/batch shapes. A trace of a
16-layer shard does not justify dividing its total peak by 16. Collect or
calculate the following for the target serving configuration:

- `budget_bytes`: usable budget on that individual device after accounting for
  other processes and the intended GPU-memory utilization; not total GPU VRAM
  and not the post-KV-allocation free-memory counter.
- `layer_weights_bytes`: a full-model-length vector of local weight/storage bytes
  on this TP device, including actual quantization scales/padding. Do not simply
  divide total checkpoint bytes by TP size; replicated tensors matter.
- `layer_kv_bytes`: a full-model-length vector for the **target serving workload**,
  including allocator block rounding and concurrent requests. For independent
  full-attention requests a conservative starting point is
  `max_num_seqs * ceil(max_model_len / block_size) * local_page_size_bytes` per
  layer, using that rank's actual KVCacheSpec, not total model KV heads. Other
  cache types require their own bounds; do not apply this formula to all models.
- The six reserve fields must conservatively cover every candidate shard/shape:
  persistent runtime allocations, peak activations, workspaces, communication
  buffers, graph pools, and safety headroom. Account for allocator overhead;
  reserves must not omit non-PyTorch allocations.
- `first_stage_bytes` includes embedding/input-side storage; `last_stage_bytes`
  includes final norm/head/output-side storage for **any** device that might
  become last. Keep these out of decoder weight vectors. Tied weights may be
  conservatively double-counted on a one-rank candidate, but not omitted.

All numbers are integer bytes, all device rows and all layer entries are
required, and zero reserves must be explicit. A small synthetic example (four
layers, two PP ranks, TP=1) is:

```json
{
  "version": 1,
  "num_layers": 4,
  "pp_size": 2,
  "tp_size": 1,
  "serving_config": {
    "model": "example-model", "revision": null, "dtype": "float16",
    "quantization": null, "kv_cache_dtype": "auto", "block_size": 16,
    "max_model_len": 128, "max_num_seqs": 2, "max_num_batched_tokens": 32,
    "gpu_memory_utilization": 0.9
  },
  "devices": [
    {
      "pp_rank": 0, "tp_rank": 0, "budget_bytes": 10485760,
      "layer_weights_bytes": [1048576, 1048576, 1048576, 1048576],
      "layer_kv_bytes": [524288, 524288, 524288, 524288],
      "runtime_bytes": 1048576, "activation_bytes": 1048576,
      "workspace_bytes": 1048576, "communication_bytes": 524288,
      "graph_bytes": 1048576, "safety_margin_bytes": 524288,
      "first_stage_bytes": 1048576, "last_stage_bytes": 1048576
    },
    {
      "pp_rank": 1, "tp_rank": 0, "budget_bytes": 10485760,
      "layer_weights_bytes": [1048576, 1048576, 1048576, 1048576],
      "layer_kv_bytes": [524288, 524288, 524288, 524288],
      "runtime_bytes": 1048576, "activation_bytes": 1048576,
      "workspace_bytes": 1048576, "communication_bytes": 524288,
      "graph_bytes": 1048576, "safety_margin_bytes": 524288,
      "first_stage_bytes": 1048576, "last_stage_bytes": 1048576
    }
  ]
}
```

These illustrative values are not model/device defaults. `pp_size` describes
all profiled ranks; every row needs memory vectors for all layers so the planner
can evaluate new assignments. `tp_size` requires a row for each physical TP
rank, even though fitting selects one timing representative per PP rank.

Live profiling validates the profile's model/revision, dtype/quantization, KV
dtype/block size, context/concurrency/batch-token limits, GPU-memory utilization,
and PP/TP dimensions
against the resolved engine config before generation. Fixed KV allocation
overrides (`kv_cache_memory_bytes` or `num_gpu_blocks_override`) are rejected
until the bounds model supports those allocation policies. Offline planning uses the
scope declared in the supplied profile; old traces cannot prove that identity.
New timing traces record TP size and reject a conflicting memory profile.

Plan JSON embeds the original memory profile and per-device `required_bytes`,
`budget_bytes`, `headroom_bytes`, layer range and cost breakdown. A sidecar is
saved for subsequent `--skip-run`. `memory_feasibility_checked=true` means checks
against `supplied_memory_bounds` passed, not that the candidate was loaded and
benchmarked. Generated serve commands retain the declared serving limits and
TP size; keep the same devices, budgets and other runtime settings (attention
backend, graph mode, adapters, etc.) used to establish the reserves. Recalibrate
the bounds after any such change and validate peak usage on hardware.


## 已移除的旧入口

已移除 `layer-linear` 成本拟合、`scale_rank_costs` 离线成本缩放、
`--allow-wall-time-comm` 以及 `send_transfer_ms` trace 字段。旧 trace 缺少新的
测量和 microbatch 信息时需重新采集。模拟异构工具仍用于实验，不能作为 DP 成本。
