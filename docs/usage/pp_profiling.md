# PP profiling：从 serving 样本到 partition

DP 的计算、通信成本来自目标设备上的实测 trace。默认使用 `shape-affine`
计算拟合和 `--comm-source serving` 通信计时，不根据 mock 算力、带宽或延迟
配置生成成本。模拟异构只改变 worker 实际执行，真实异构设备不需要设置模拟变量。

## 完整流程

1. 从目标服务日志抽取 requests，保留 prompt、生成参数、到达间隔和并发设置。
2. 在目标设备上启动独立 profiling 服务；先 warmup，再采集 scheduler 实际组成
   的 microbatch。保持目标服务的 model、dtype、TP、cache 和调度设置一致。
3. 拟合模式保持设备映射和 PP size 不变，更换 partition，每个 rank 至少测两种层数。
   逐层模式每个设备测当前负责的 decoder layers 即可，不要求每个设备测遍所有层。
4. 拟合模式按主 trace 的 microbatch 分布加权；逐层模式使用
   `--layer-aggregation microbatch`，对实际出现的每个 microbatch 等权平均，包括 mixed
   batch，同时按同样的样本范围统计 embedding、LM head、final norm 和 runner 开销。
   通信也按对应 workload 汇总实测值。准备目标 serving 配置下的内存容量上界。
5. 分别执行 latency 和 throughput DP。两个目标如果对应不同的并发/到达负载，
   应分别采样和统计成本，不能把一个负载的最优结果当作另一个负载的最优结果。
6. 用没有参与 profiling 的请求，关闭 tracing，启动 DP 推荐、均分和相邻方案。
   每次启动都 warmup，改变方案顺序并重复启动测量。报告实测候选排名；DP 的
   预测最优不代表 vLLM serving 的全局最优。

默认保持设备顺序和 PP size。逐层模式加 `--device-selection` 后，DP 可选择设备子集、
顺序和层数，源设备固定为主 trace 的设备 0。两种模式都近似认为同一设备上的各 decoder
layer 具有相同的代表成本；raw trace 仍保留逐层数据供检查误差。

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

## 选择计算 profiling 模式

| 模式 | 采集内容 | DP 如何得到一段层的计算成本 |
| --- | --- | --- |
| `shape-affine`（默认） | 多种切分下整个 stage 的耗时 | 拟合的固定开销 + 层数 × 拟合的每层耗时 |
| `layer-measured` | 每次实际 microbatch 中，每个 decoder layer 的耗时 | 层数 × 代表 layer 实测均值 + 首尾模块实测值 + runner 开销 |

两种模式共用通信测量、内存检查和 DP。计算、通信成本都来自实测，不从模拟算力或
带宽配置推算。`layer-measured` 不做计算成本拟合，也不会在数据缺失时自动退回拟合模式。

**用真实服务 requests 采集逐层 trace：**在上面的 `vllm serve` 命令启动之前设置
`VLLM_PP_COMPUTE_MODEL=layer-measured`，并给服务加上 `--enforce-eager`，使用 TP=1。
客户端采样、warmup 和发送 requests 的流程不变。worker 在 decoder layer 的调用前后
记录 CUDA/NPU events，整段 stage 完成同步后统一读取；不会在每层之间插入同步。
计时边界是该 decoder module 的调用边界，当前 stream 上两个 events 之间的时间，
可能包含设备等待 CPU 提交 kernel 的间隙，不保证等于所有 kernel 耗时之和。

```bash
export VLLM_PP_COMPUTE_MODEL=layer-measured
# 按前面的流程启动服务（加 --enforce-eager），采集 base / extra 等独立目录。
# 每个目录都使用相同模式、设备映射、模型、TP 和服务参数。
python -m vllm.distributed.pp_partition /tmp/pp-profile/base \
  --compute-model layer-measured \
  --layer-aggregation microbatch \
  --memory-profile /tmp/pp-profile/memory.json --objective throughput
```

逐层模式对每个 microbatch，先平均其中测到的 decoder layers 耗时。
`--layer-aggregation microbatch` 再对这些 microbatch 均值直接取平均：每次实际执行
占一份权重，不按 token 数二次加权，不要求同时存在 prefill 和 decode，也不排除 mixed。
通信和首尾模块使用相同的 aggregation。warmup 和空 microbatch 均排除。
plan 的 `workload_aggregation=observed_microbatch_mean` 标明此方式。

为兼容之前的实验，省略该选项仍使用 `--layer-aggregation phase-balanced`：分别求
prefill、decode 的 microbatch 平均值，最后计算 `(prefill_mean + decode_mean) / 2`。
两个阶段各占一半，不按 decode step 数量加权；mixed microbatch 不混入该平均。
若缺少任一阶段的正式样本，`--workload all` 报错；可显式选择 `prefill`、`decode`
或 `mixed` 单独分析。plan 的 `phase_summary` 保存各阶段样本数和 layer 均值。

只采一次 partition，也可以给设备分配它没执行过的 decoder layer：DP 使用该设备的
`t_layer_ms` 代表所有 decoder layers。这是明确的同结构层近似，不声称各层实际完全等时。
`--fit-trace-dir` 可追加测量，逐层模式不拟合，也不再要求相同 shape 或 layer ID 覆盖。

raw trace 中 `layer_compute_ms` 保留全局 layer ID。`embedding_ms` 测量实际
`embed_tokens` 调用；`lm_head_ms` 测量 `logits_processor`（包含实际 head projection
和 logits 处理，因为 ParallelLMHead 不一定调用 forward）；`final_norm_ms` 测量末端 norm。
`non_layer_compute_ms` 仍为 stage wall 减去 decoder 总耗时，便于兼容审计；
`runner_overhead_ms` 再扣除上述三个 endpoint，防止重复计费。
DP 在包含第一层的 stage 加 embedding，在包含最后一层的 stage 加 head 和 final norm。
这些 endpoint 没有测到时是 `null`，不是零，设备不能承担缺少测量的首尾角色。

当前逐层采集要求模型暴露 `model[.model].layers`（按全局编号索引的 `ModuleList`）及
`start_layer/end_layer`，每个本地 decoder 在一次 execute_model 中恰好调用一次。
不支持 CUDA Graph replay、compilation、TP>1 或 stage 末尾的 mock compute slowdown；
这些情况会报错。真实异构设备不需要 mock slowdown。逐层模式的模拟异构可采用
设备实际资源限制（例如 NPU core quota），让 device events 测到真实执行时间。
另一种实验方式是在各 layer 和 endpoint 的调用结束前实际加入 sleep，再记录结束 event；
此时必须关闭原来的 stage compute sleep，避免重复减速，并在 profiling 与最终 serving
中保留相同的模拟 hooks。不要把 stage 末尾 sleep 离线分摊到各层，冒充逐层实测。
逐层 hooks/events 本身有开销，且 eager 的结果不代表开启 compilation/graphs 后的性能；
最终 serving 对照必须关闭 tracing，并使用目标执行设置重新测量。

对于现有合成 workload 工具，可直接使用 `vllm pp-profile ... --compute-model layer-measured
--enforce-eager --collect-only` 采集；该命令会在创建 engine 前设置 worker 模式。
重新使用拟合模式时，采集服务设置 `VLLM_PP_COMPUTE_MODEL=shape-affine`，规划使用
`--compute-model shape-affine`（默认）。两类 trace 不允许混用；旧 trace 没有逐层数据，
不能用于逐层模式。

## 拟合模式的计算成本

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

start 在 Python 通信调用前，end 在调用返回、完成设备同步和实际 mock 等待之后。这个区间扣除了
两端进入函数的时间差，仍包含 metadata、接收端分配、backend 调度和同步。
它不是设备级 wire time，也不保证与 partition 无关。trace 同时保留 send_end，
因为接收端完成和发送端释放资源的时间可以不同。

配对检查 session、step、microbatch hash、传输字节数和时钟来源。当前支持同机
Linux、TP=1；跨主机时间不可直接相减，时钟来源未知或配对失败就报错。
新 trace 使用 `comm_delay_in_window=true` 标记 mock 等待已经计入窗口，DP 直接
使用实际计时，不再乘减速倍数。旧 trace 如果将 mock 通信等待放在窗口外，仍然拒绝使用，
避免静默漏算模拟延迟。

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
该模式不要求 memory profile，也不生成可部署方案。拟合模式仍需多种层数；逐层模式
直接记录 decoder 和 endpoint，可以从一次 partition 生成代表层成本。

## 设备选择、顺序与链路测量

在服务请求结束后，对所有设备对回放真实 trace 中的 activation 布局：

```python
llm.collective_rpc("profile_pp_links", args=(3, 10, True))
```

也可在受控服务的 `/collective_rpc` 使用同一方法名和参数；`vllm pp-profile` 的 live run
加 `--device-selection` 会自动执行。输出 `pp_topology_ppA_toB.jsonl`，保留原始重复测量、
方向、payload 和 reference trace ID。当前 TP=1，使用设备 0 的 payload 作为公共 workload，
适用于所有 decoder 边界 activation 布局相同的模型。跨主机也不相减两端时钟。
rank-indexed mock 通信参数不能解释任意设备对，因此 all-pair replay 拒绝它们；计算资源
配额模拟不受影响。缺少 link 的路径不可用，不会按带宽配置补算通信成本。

```bash
vllm pp-profile --skip-run --trace-dir /tmp/pp-profile/base \
  --compute-model layer-measured --device-selection \
  --warmup-steps 0 --memory-profile /path/to/memory.json \
  --objective throughput
```

DP 状态保留已选设备集合、最后一段的层区间、设备和上一个设备。只有确定后继设备后，
才能把最后一段的 outgoing 通信加入发送端占用；仍使用 blocking `recv + compute + send`。
latency 是 forward 的串行计算与 activation 传输之和；不含生成 token 返回、sampling、
请求排队等 E2E 成本。throughput 仍是平均占用近似，不能据此声称 serving 全局最优。

单次 PP profile 只有源设备测到了 embedding，末设备测到了 head/norm。其它设备可作
middle stage，但不能凭 decoder 比例推算 head。若需要比较不同末设备，可追加交换末设备
的短 profile，在额外目录写 `pp_device_map.json`，例如 `[0, 3, 2, 1]` 表示该次 rank
0/1/2/3 分别是主 trace 的设备 0/3/2/1。用 `--fit-trace-dir` 导入，按原设备身份汇总。
无需为这些设备轮换测遍每一层；只是补充所需的首尾模块测量。模型和 workload 必须一致。
主 trace 的设备映射必须为 identity；拟合模式不支持重排 profile。

plan 输出 `device_order`（主 trace 的设备 ID）和对应顺序的 `partitions`。部署时必须
同时按这个顺序绑定设备，并设置 `VLLM_PP_LAYER_PARTITION`；只改层数不改设备绑定是错的。
工具不会把 profiling ID 擅自当成 CUDA/NPU 的物理编号。选中设备的内存 profile 会重排为
新的 PP rank，并写到 `pp_selected_memory_profile.json`，保留原候选池的 profile 不覆盖。

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
