# v0.23 的 EdgeShard 式 profiling：采集、规划和手动复验

这里的功能是：**先测四个 PP stage 的实际执行成本，再用动态规划推荐每张卡的层数，
最后关闭采集、按推荐层数重新启动，检查模型是否正确运行。**
推荐方案是一个需要实机复验的候选，不是“运行规划器就已经证明更快”。
“EdgeShard 式”指采用按设备成本、通信和内存约束分配连续模型层的方法，
不表示逐项复刻原论文的完整系统。

本文对应干净官方 `v0.23.0` 基线上的 `ysy/hetero-pp-v023` 分支，
使用匹配的 vLLM-Ascend `0.23.0.post1`。旧 profiling 的 v0.13 历史记录保留原版本；
不能用它代替本分支新 CPU/Gloo 路径的采集结果。

## 1. 当前范围与启动前提

当前入口固定为四个 PP stage、TP=1、FP16、eager、同步调度、原四设备顺序。
每节点两个 Ray actor，每个 actor 占一张卡。代码准入允许
`LlamaForCausalLM` / `Qwen2ForCausalLM` 的普通 dense 文本模型；
本轮实际只验证 CodeLlama-7B-Instruct / Llama，没有 Qwen2 实机验收。
不在这条入口中做设备子集选择、设备重排、改变 PP 数量、PD、量化、LoRA 或图执行。
入口显式拒绝列举的自定义 tokenizer、HF、注意力和加载参数。
本轮只验证本文的基础配置集合；完整 EngineArgs JSON 留档不等于生成器会通用重放
所有 EngineArgs，也不能据此保证任意 additional_config、LoRA 等自定义组合可用。

本轮目标硬件是 hw/hw2 两个容器各两张 910B3，共四张不同的物理卡，
**两个容器位于同一物理宿主机**。这既不是 310P 验收，也不是跨物理机网络性能测试。
原 `shared-clock` 通信成本配对要求两端有已验证相同的单调时钟域；
两台独立物理机的时间戳不能直接相减，NTP 对时也不足以满足这个条件。
2026-11-12 新增 `--comm-timing ack`，允许不同单调时钟域，用发送端计时和
接收完成确认测交接成本，详见第 3.1 节。该新增模式的 CPU/Gloo 测试与既有
910B 实机证据分开；尚未把它标为跨物理机 NPU 性能验收通过。

先按[CPU/Gloo 手动启动指南](hetero_pp_cpu_gloo_zh.md)准备匹配环境、核对源码导入位置，
并分别启动 Ray head 与 worker。采集前不要同时运行 API 或另一套占卡任务。
下面假设在 hw 上已有：

```bash
export REPO=/path/to/vllm-v023
export VENV=/path/to/matched/venv
export PY="$VENV/bin/python"
export MODEL=/same/path/to/CodeLlama-7b-Instruct/model
export PLAN=/path/to/layout_uniform.json
export RAY_ADDRESS=172.20.183.252:16379
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export VLLM_USE_V2_MODEL_RUNNER=0
```

`PLAN` 是原来的 topology JSON，本轮采集使用 32 层模型的 `8/8/8/8`：
节点 IP、SoC 和 stage 顺序都要正确。模型路径和源码应在双方一致可用。
`TRACE` 目录需要在双方相同绝对路径可见且实际共享；本轮的共享 userdata 满足这个条件。
仅创建两个同名本地目录不够：Ray 不自动共享文件，本入口也不自动搬运远端 raw。
共享模型、共享 trace 目录不等于拥有两份独立备份。

## 2. 先采集：只收真实测量，不立即选层数

为每次采集选择一个**全新或空**目录。入口发现非空目录会拒绝，
不会删除旧 raw、拼接上次测量或覆盖成功回执。

```bash
export TRACE=/shared/path/edgeshard_profile_run001
test ! -e "$TRACE"

"$PY" -m vllm.entrypoints.cli.main pp-profile "$MODEL" \
  --hetero-plan "$PLAN" --trace-dir "$TRACE" --collect-only \
  --compute-model layer-measured --comm-timing ack \
  --input-len 64 --output-len 16 --num-prompts 4 \
  --num-iters-warmup 1 --num-iters 2 --seed 7 \
  --dtype float16 --enforce-eager --no-async-scheduling \
  --max-model-len 512 --max-num-batched-tokens 256 --max-num-seqs 4 \
  --block-size 128 --num-gpu-blocks-override 64 \
  --gpu-memory-utilization 0.5 \
  --enable-chunked-prefill --no-enable-prefix-caching
```

这就是 `vllm pp-profile` 的显式 Python 入口，确保使用 `$PY` 所在的匹配环境。
它选用真正的 `HeteroPPExecutor` 类；不要改成普通 Ray executor 或旧 HCCL worker。
上面使用新 ACK 口径，适用于同机或跨物理机的三条链路。要复用原共享时钟口径，
显式换成 `--comm-timing shared-clock`；不传参数时，实时采集仍默认 `shared-clock`。

这个例子中，每次 `generate()` 输入四条合成 token prompt：一轮 warmup，
两轮正式采集。warmup 与正式 prompt 不同，正式迭代之间的 prompt 也不同。
输出固定 16 token、`ignore_eos=True`、greedy；这不是完整 GSM8K 题目或模型质量评测。
`--num-prompts 4` 是采集负载的批量参数，不是原研究实验每点 128 请求的那套协议。

采集参数由 head 显式传给 Ray actors；不能只指望已启动的 Ray daemon
自动继承 head 后来设置的环境变量。warmup 切换、flush 和内存读取前，
执行器会等已提交的 RPC 收尾，避免 `generate()` 返回后还有 bookkeeping 工作在途。

`--collect-only` 结束后，应有以下证据：

| 文件 | 作用 |
| --- | --- |
| `pp_stage_pp0_tp0.jsonl` 至 `pp_stage_pp3_tp0.jsonl` | 四个 rank 的逐 step 原始测量，含 warmup 标记 |
| `flush_receipts.json` | 四个 worker 的完成、计数、待处理量、错误与 raw SHA 回执 |
| `memory_observations.json` | 真实 storage、KV geometry、allocator 及设备内存观察 |
| `pp_memory_profile.json` | 显式内存约束，例子采用 version 2 fixed-block policy |
| `memory_evidence.json` | 原始内存观察与推导假设，说明哪些是测量、哪些是外推 |
| `engine_args.json`、`engine_config.json`、`model_config.json` | 原输入参数、实际运行配置、实际 checkpoint 配置 |
| `profile_manifest.json` | 统一 session、源码、模型/config/topology、实际 runtime 和上述文件的 SHA 绑定 |

只看到 raw 文件还不够。controller 要收到四个完整 flush 回执、验证实际身份和
SHA、读取内存证据，并完成服务收尾后，才写 `profile_manifest.json` 的完整回执。
发生错误时保留原目录诊断，不能把失败目录清空后当同一轮重试成功。

### 2.1 单文件脚本与通信速率统计

如果尚未启动容器和 Ray，优先按
[双端启动指南的自动 profiling 模式](hetero_pp_start_zh.md#自动-profiling-模式两边仍使用原入口)
在两台宿主机运行 `start_hetero_pp_docker.sh --profile --trace-root ...`。
该模式复用原环境探测和 Ray 连接，head 自动调用下方脚本；不需要先启动 API，
也无需手工填写 Python/plan 路径。它的显存默认值为原启动入口的 0.8。

也可使用 `examples/online_serving/hetero_pp/profile_edgeshard.sh`，在文件顶部
填写 `MODEL`、`PLAN`、`RAY_ADDRESS`、真正共享的 `TRACE_ROOT`，并指定匹配服务
环境的 Python 路径 `PY`，只在 head 容器中执行：

```bash
bash examples/online_serving/hetero_pp/profile_edgeshard.sh
```

它采用上面的小规模采集配置，采集后打印计算/交接耗时表及三条相邻链路的有效
交接速率，再运行离线 latency 目标 DP。每个 stage/链路分别列出全部、prefill、decode 和存在时的 mixed
step；阶段由真实调度 token 计数判定。排除预热，零 payload 只计排除数量，
不计入速率样本。汇总和逐交接记录分别保存到采集目录的
`communication_speed_summary.csv`、`communication_speed_samples.csv`。

耗时表 `stage_and_link_timing_summary.csv` 包含四个 stage 的 `compute` 行与
三条链路的 `handoff` 行，列出样本数、平均值、P50/P90/P95/P99，单位 ms，
另存总和、最小值和最大值。逐步计算保存在 `stage_compute_samples.csv`。
计算时长取 `compute_wall_ms`，这是 `execute_device_ms + sample_device_ms`
两个设备事件区间的和，不是 host 函数调用时长，不包含两个 RPC 之间的空档；
包含末 stage 的采样设备区间。缺少有效事件时长或区间不守恒会拒绝汇总，
不会把缺失值补成零。交接时长取配对核验后的 `send_ack_ms`，包含零 payload
交接；速率统计则排除零 payload。各阶段/链路的分位数不能直接相加得到请求延迟。

- `mean_gbps`：逐交接有效速率的算术平均。
- `aggregate_gbps`：总 payload 字节数乘 8，除以各交接耗时之和；不是整轮墙钟吞吐。
- P10/P50/P90/P95/P99：逐交接速率排序后按 `(n-1)×q` 线性插值。
  P10 表示慢侧，P90/P99 表示快侧；CSV 另存最小值、最大值、样本数、payload 范围。

所有速率单位为 Gbps，窗口使用第 3.1 节的 ACK 完成交接口径，包括 D2H、Gloo、
H2D 和 DONE 确认，不是物理网卡带宽。all 行会混合不同 payload，阶段拆分也不
代表固定 payload 测速；当前小规模配置的样本量有限，分位数不代表稳定性能边界。
统计会核对完整采集 manifest、四份 raw SHA、step 配对和两端 ACK 证据；这是通信
统计的输入检查，不代替后续 DP 的完整配置/内存校验或推荐方案实机复验。

已有完整 ACK 采集可以只重算统计，不启动 Ray、模型或 NPU：

```bash
PY=/path/to/matched/venv/bin/python \
  bash examples/online_serving/hetero_pp/profile_edgeshard.sh --stats-only "$TRACE"
```

## 3. 测到的“计算”和“通信”分别是什么

`layer-measured` 用实际模块事件测每层 decoder、首端 embedding、末端 norm
和 LM head。v0.23 的 `execute_model` 与 `sample_tokens` 是两个独立 RPC，
因此同一 step 会合并这两段的事件耗时，并明确保留各自耗时；二者之间的 RPC
空档不算设备计算。LM head 和 sampling 不是同一个操作，未执行的模块标为缺失，
不能把“没有观察到”写成零成本。

通信走同一个 CPU/Gloo 通道，包含真实 D2H、Gloo 和 H2D 的完成依赖。
原 `shared-clock` 配对窗口来自实际发送/接收 host 时间戳，
不由“配置了多少 Gbit/s”换算出来。
它含 CPU 复制、元数据、分配、Gloo 排程和同步，**不等于纯网卡传输时间**。

profiling 还在发送前增加完成事件检查，防止把尚未完成的计算尾部再次计入通信成本。
这段 readiness 等待已有计算计时覆盖，不会另加一次成本。
这种 instrumentation 可能改变原有的元数据与设备尾部重叠；
trace-off 服务没有这项 profiling 检查。因此不能直接把 profile 中的耗时当原生服务延迟。

### 3.1 跨物理机：READY/DONE ACK 计时

实时采集加 `--comm-timing ack`，head 会把该设置显式传给所有 Ray actors，
无需修改 Ray daemon 环境。三个相邻 hop 都采用相同口径：

1. 接收端进入本 step，发送 `READY`，然后调用原来的 tensor 接收通道。
2. 发送端等本地计算完成，再等到 `READY`。从此后发送数据之前记录 `t0`。
3. 原有通道发送元数据和实际 tensor，保留 hidden states/residual 等原始内容。
4. 接收端等全部 Gloo 接收及 CPU→NPU 复制完成，再发送 `DONE`。
5. 发送端收到 `DONE`，记录 `t1`，交接成本为本机时钟的 `t1−t0`。

这个窗口包含元数据、D2H、Gloo、分配、H2D 和 DONE 返回成本；
不包含发送端等待 READY 的时间，也不包含已经单独计时的计算就绪检查。
它不是纯网卡传输时间、物理单向延迟或 NPU 原生互联带宽，不应除以二。
`payload_bytes×8/(耗时ms×10^6)` 可称为本次 payload 的 **有效交接速率（Gbps）**，
该 payload 不含控制消息/元数据的线缆开销，不是对物理链路速率的验收。

控制帧使用同一显式 CPU/Gloo group 的保留 tag `17000/17001`，
与原 header tag `0`、tensor tag `100..163` 分开。
帧通过 SHA256 绑定协议版本、READY/DONE、session、step、batch、源/目的 rank、
payload 字节数。DONE 只在实际接收完成后发送；错误身份、丢失确认或超时会失败，
不自动重试，不发布完整 profile manifest。每条控制消息等待有上限
`min(topology timeout_s, 60)` 秒，另保留 Gloo group 和父执行器的原有超时。

四份 `pp_stage_pp*_tp0.jsonl` 会增加这些字段：

| 字段 | 含义 |
| --- | --- |
| `comm_timing` | 本轮为 `ack`；旧口径为 `shared-clock` |
| `send_ack.identity` / `recv_ack.identity` | 两端相同的 session、step、batch、rank、字节数 |
| `send_ack.ready_start_ns/ready_end_ns` | 发送端等待 READY 的本地时间，单列、不加进交接成本 |
| `send_ack.start_ns/end_ns` | 发送端数据开始至收到 DONE 的本地时间 |
| `recv_ack.data_ready_ns` | 接收通道返回、所有 H2D 已完成的本地时间 |
| `recv_ack.done_start_ns/done_end_ns` | 接收端发送 DONE 的本地时间 |
| `send_start_ns/end_ns`、`recv_start_ns/end_ns` | 原发送/接收窗口仍保存，不与 ACK 窗口重复相加 |

`profile_manifest.json` 和每个 worker 的 flush receipt 都绑定 `comm_timing`。
离线 `--skip-run` 默认读取 manifest 中的口径；显式传入另一种口径会拒绝，
不能把旧 trace 改标为新测量。分析时完整配对记为 `paired_ack`，
规划器来源记为 `measured_serving_ack`，每条 hop 只加入一次成本；
`pp_partition_plan.json` 的 rank 成本中可检查该来源和 `t_comm_out_ms`。
两种口径不能混在一轮或多个拟合 trace 中。新模式目前只用于固定设备顺序切层，
不用于设备子集选择或设备重排。

READY/DONE 会改变 profiling 的握手和部分重叠。推荐分区仍须在 trace-off 模式
重新部署，核对输出 IDs，并另测 TTFT/TPOT/吞吐。没有启用 tracing 的服务不增加
任何 READY/DONE；原成功数据和历史 shared-clock 结果不改写。

raw 中 `device_id` / `profile_device_id` 是全局 profiling 身份 `0/1/2/3`，
用于把设备成本与 stage 绑定；`local_device_id` 才是节点内逻辑卡号，
可能为 `0/1/0/1`。两者都不是可随意替换的物理 PCI 卡号。

## 4. 离线规划：同一份 raw，可以看两个目标

`--skip-run` 不加载模型、不占 NPU，只校验既有证据并做 DP。
它拒绝没有本分支 manifest 的旧 trace，也拒绝 session、源码、topology、
config、transport 或 raw SHA 不匹配的数据。

下面为两个目标各复制一份小型采集证据，避免它们的推荐文件和派生回执互相覆盖。
这不产生新测量，不重复运行 warmup 或请求。

```bash
export LATENCY=/shared/path/edgeshard_analysis_latency_run001
export BOTTLENECK=/shared/path/edgeshard_analysis_bottleneck_run001
test ! -e "$LATENCY"
test ! -e "$BOTTLENECK"
cp -a "$TRACE" "$LATENCY"
cp -a "$TRACE" "$BOTTLENECK"

"$PY" -m vllm.entrypoints.cli.main pp-profile \
  --hetero-plan "$PLAN" --trace-dir "$LATENCY" \
  --skip-run --objective latency --workload all \
  --output-json "$LATENCY/latency_plan.json"

"$PY" -m vllm.entrypoints.cli.main pp-profile \
  --hetero-plan "$PLAN" --trace-dir "$BOTTLENECK" \
  --skip-run --objective throughput --workload all \
  --output-json "$BOTTLENECK/bottleneck_plan.json"
```

两个目标的含义：

- **`latency`，默认目标**：最小化各 stage 代表性计算成本之和，加相邻 hop 的实测通信成本。
  它不包含完整 HTTP、排队、动态调度和请求到达过程，不是用户端 E2E 的预测器。
- **`throughput`**：比较最慢 stage 的 `recv + compute + send` 占用成本。
  这是有足够在途工作时的 stage 瓶颈候选模型；当前执行器只有一个在执行的
  scheduler batch，不能把它称作本执行器实测吞吐最优，也不能据此宣称性能提升。

默认 `microbatch` 汇总让每个观测到的 step 等权。
设备上的“代表层耗时”来自本机已加载层的均值；把它用于其他同结构层是模型近似，
不是所有层都在所有卡上实测过。换负载、batch/context 分布或执行模式，要重新评估代表性。

每个分析目录会生成 `recommended_hetero_plan.json` 与
`recommendation_receipt.json`。推荐只改变原 topology 的 `layers`，
不改 node IP、SoC、设备顺序和其他 topology 字段。
此入口固定 PP4、warmup 由 `is_warmup` 排除，不开放改变 PP 数量或跳过内存检查的发布路径。

## 5. 内存检查为什么仍有假设

例子的 `--num-gpu-blocks-override 64` 是明确的固定 KV policy，不是测得的自动 KV 容量。
如果使用自动 KV、又没有给出匹配配置的显式 `--memory-profile`，入口会拒绝继续，
不会把未经检查的候选写成可直接发布的方案。

自动构造的 fixed-block 内存约束按设备分别计算：

1. 权重部分取这张卡实际见到的 decoder storage 最大值，外推其他同结构层。
   可能保守计入共享 buffer；这并不是每卡实测了完整模型。
2. 每层 KV 由 native page size 乘固定 block 数量得到，必须与实际 KV pool
   的独立 storage owner 总字节数严格守恒。不吻合会拒绝，不能猜一个容量。
3. 非 KV 预留使用观测的峰值 reserved，扣除本机已加载模型和真实 KV pool，
   再计入当前非 PyTorch 占用。另留显式 1 GiB safety margin；这不是实测分配。
   不会把包含 KV 的峰值与完整 KV 估算重复相加。

DP 每个候选都要在这些界限内。但这是**当前负载、同结构层、原设备顺序和首末角色固定**
条件下的候选约束，不是任意分区、任意未来负载都不会 OOM 的保证。
因此下一步实际加载推荐方案不能省略。

## 6. 关闭 profiling，按推荐 plan 启动服务

先确认采集进程结束、四个 actor 已退出且卡空闲；保留 Ray 集群供新服务使用。
选定一个分析目录，例如默认 latency 候选：

在最后启动 HTTP 服务前，先做推荐分区的模型正确性复验。
生产 collector 本身不保存 `LLM.generate()` 的完整 token 输出；
本轮因此使用独立 public-generate 记录器，完整委托原调用后保存实际
input IDs、output IDs 和 text，不替换 model/worker/forward。
再在 tracing 关闭的推荐分区上重放同样的 input IDs，逐条对照 profile on/off
全部输出，并复验独立 heldout 单请求和 batch=4 参照。
该记录器与结果属于本轮独立验证证据，不是 collector 自动提供的功能。

模型对照和退出检查结束、卡再次空闲后，才启动 API 做最后的服务检查：

```bash
unset VLLM_PP_STAGE_TRACE VLLM_PP_TRACE_SESSION VLLM_PP_COMPUTE_MODEL
export VLLM_HETERO_PP_PLAN="$(cat "$LATENCY/recommended_hetero_plan.json")"
export VLLM_USE_V2_MODEL_RUNNER=0

"$PY" -m vllm.distributed.hetero_pp.serve \
  --model "$MODEL" --served-model-name codellama7b \
  --pipeline-parallel-size 4 --tensor-parallel-size 1 \
  --dtype float16 --enforce-eager --no-async-scheduling \
  --max-model-len 512 --max-num-batched-tokens 256 --max-num-seqs 4 \
  --block-size 128 --num-gpu-blocks-override 64 \
  --gpu-memory-utilization 0.5 \
  --enable-chunked-prefill --no-enable-prefix-caching --seed 7 \
  --no-disable-cascade-attn \
  --host 127.0.0.1 --port 18080
```

这条手写命令对应本文的采集配置。若你的采集参数不同，以该目录打印的专用
trace-off 命令和配置回执为准，不要随手改 context、KV blocks 或 HBM 比例。
原始参数与平台最终 effective 值可能不同；本轮 cascade 参数的 effective 值为 False，
已显式保存并用 `--no-disable-cascade-attn` 重放。
不要换成普通 `vllm serve`，否则可能退回别的执行器和通信路径。

按 [CPU/Gloo 手动指南](hetero_pp_cpu_gloo_zh.md)检查 startup reports、真实层区间、
设备与源码身份，再发送 `return_token_ids=true` 的 HTTP 请求并核对完整 IDs。
HTTP 是最后的服务复验，不能替代前面的 profiler 完整性和 profile on/off 对照。
还要检查退出成功、通信组清理和卡空闲；“有结果文件”不等于后台退出正确。

如果还想比较速度，需要另行设计 trace-off、相同负载和指标边界的对照测量。
规划目标分数更低、推荐了不同层数、一次请求更快，都不能单独证明稳定收益。

## 7. 本轮验收证据位置

源码与本地验收目录：`output/edgeshard_v023/`。
本轮汇总见[验收记录](../design/edgeshard_v023_validation.md)。
该目录保留 red/green 回归、环境核对、独立运行 session 和完整实际验证回执。
最终 v5 采集四 rank 各 98 行（32 warmup、66 正式），生成 latency `1/1/1/29`
和 stage-bottleneck `9/8/7/8` 两个候选。两种推荐分区都实际加载，
各 17 请求 / 272 输出 token 的完整 IDs/text 与参照相等。
最后 HTTP 200、16 输出 IDs/text 一致，API 正常退出；两端四卡和自有进程、端口已检查清理。

相关 CPU 回归两端各 217 项通过，已改文件的 pre-commit 和 mypy 3.10/3.12 通过；
这不等于完整仓库 pytest 或全 CI。Ray CLI 的受控 SIGINT 退出码 1 和原生 manager
收尾日志原样保留，不声称全程没有 warning。

因此，本轮这条四卡 910B 功能链路已有完整证据。v4 的中间候选仍单独保留，
不能当最终方案；不同轮次的候选变化也不证明稳定最优。
仍未验证 Qwen2、310P、跨物理主机网络或性能收益。
