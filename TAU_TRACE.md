# Ascend 上运行 τ-Batch / ShareGPT trace 采集

本流程面向当前分支、单机 Ascend、TP=1、PP=2 和纯文本生成模型。
`serve_tau.sh` 启动服务，`bench_tau.sh` 下载/读取 ShareGPT、压测、检查 trace。
服务直接调用当前环境的 `vllm serve`，流量直接调用 `vllm bench serve`。
配置和 trace 辅助程序使用 `python`，可通过 `PYTHON=/path/to/python` 指定。
`vllm` 与 `python` 应来自同一已配置环境。
脚本不会安装或升级 CANN、torch-npu、vllm-ascend。

`serve_tau.sh` 默认选择本仓库的
`vllm.v1.worker.tau_ascend_worker.TauAscendWorker`。
它继承已安装的 `NPUWorker`，在插件完成设备和 runner 初始化后，
仅包装该 runner 实例的 `execute_model()`，记录 `compute`；无需修改或重装插件。
PP 通信、模型加载和返回值处理仍由插件完成。

计时口径沿用 host 起止时间和结束时 NPU 同步，不是纯设备 kernel 时间。
没有 `tau_fwd_id` 的调用不会添加同步。模型执行或同步失败会抛出异常，
不会写出正常的 `compute` 记录。只有 `stage` 而没有 `compute` 时，
采集检查仍会失败，不会把外层时间误作计算标签。
此接入基于官方 vllm-ascend v0.13.0 的调用路径；自定义/其他版本插件仍需实机 smoke 验证。

## 1. 环境准备

在远端进入**包含本次脚本的当前分支**，激活已经能运行该模型的 Ascend Python 环境，
并按服务器现有方式加载 CANN 环境变量。服务和压测应在同一服务器、同一 Python 环境运行。

```bash
cd /path/to/vllm
npu-smi info
python -m pip show vllm vllm-ascend torch torch-npu
bash serve_tau.sh /absolute/path/to/model --dry-run
```

应使用与本分支 vLLM 0.13.0 相容的 vllm-ascend 和依赖组合。
服务参数统一在 `configs/serve.yaml`，包括模型路径、PP/TP、batch 上限和 trace 开关。
填写后直接 `bash serve_tau.sh`；脚本读取配置并构造 `vllm serve` 命令。
`SCHEDULER: tau/default` 分别使用 `SCHEDULERS.tau/default` 下的调度参数。
下文采集/拟合流程使用 tau；默认调度器可运行同一 bench/SLO 测试，但不生成 Tau trace。
脚本设置仓库 `PYTHONPATH` 来使用当前源码；
环境仍须有兼容的 vLLM 编译产物、Ascend 插件和模型依赖。
仅有一个未构建的源码 checkout 不等于环境安装完成。

## 2. 终端 A：启动服务

默认启动只需要模型路径，运行目录和本地 trace 文件自动生成：

```bash
cd /path/to/vllm
bash serve_tau.sh /absolute/path/to/model
```

服务保持前台运行，Ctrl+C 停止。启动时会打印实际 `RUN_DIR`、`TRACE` 和终端 B 命令。
服务通过 shell 的 `exec` 启动，当前 PID 直接成为 vLLM，退出信号由 vLLM 处理。
Python 辅助命令只保存配置和环境记录，不监护服务；不再生成 `server_exit.json`。
`RUN_DIR` 默认是仓库下 `output/<时间>_<id>/`；`TRACE` 默认在系统临时目录
（Linux 通常为 `/tmp`）下使用独立文件名。
仍可通过环境变量或 `--run-dir` 指定运行目录，通过 `TRACE` 指定 trace 文件。
指定的路径必须尚不存在；如需恢复自动路径，先 `unset RUN_DIR RUN TRACE`。
不再通过删除正在写入的 trace 来开启新一轮；重启时使用新目录。
`MODEL=/absolute/path/to/model bash serve_tau.sh` 形式仍然可用。

默认参数如下，可通过同名环境变量覆盖：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1` | 两张 NPU |
| `TP` / `PP` | `1` / `2` | 张量/流水线并行度 |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 本机访问地址；跨机访问时自行设置 HOST |
| `MAX_MODEL_LEN` | `4096` | 请求总长度上限 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 单次 forward token 预算 |
| `MAX_NUM_SEQS` | `4` | 单次 forward / 单个 microbatch 的请求数上限 |
| `MAX_MICROBATCHES` | `0` | 不额外限制 wave 的 microbatch 数 |
| `MIN_WAITING` | `0` | 不因等待阈值阻塞小样本或最后几个请求 |
| `GPU_MEM` | `0.90` | 传给 vLLM 的 `--gpu-memory-utilization`，Ascend 沿用该名字 |

相对旧脚本：`MIN_WAITING` 从 `MAX_NUM_SEQS` 改为 `0`；token 预算从 4096 改为 8192；
HOST 改为本机地址；trace 默认使用每次启动的独立本地临时文件。
8192 是本轮采集的预算选择，并非 Ascend 的强制要求，也不是 KV cache 容量。
当改变 microbatch 上限时，仍可能因 token 预算或 KV 余量而形成更小的 microbatch。
EOS 回调策略没有变化。

## 3. 终端 B：先 smoke

激活相同 Python 环境。服务启动时自动更新 `output/latest`，bench 默认读取它，无需复制目录。
多服务时可显式传入运行目录；优先级：命令行目录 > `RUN_DIR` 环境变量 > `output/latest`。
`latest` 记录启动配置，不保证服务已经就绪；bench 会等待服务就绪。

```bash
cd /path/to/vllm
bash bench_tau.sh --mode smoke --download
```

`--download` 仅在默认 `datasets/sharegpt.json` 不存在时下载
[ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
的 `ShareGPT_V3_unfiltered_cleaned_split.json`，已有文件会复用。
无法访问 Hugging Face 时，可自行把该 JSON 传到服务器，再运行：

```bash
bash bench_tau.sh --dataset /data/ShareGPT_V3_unfiltered_cleaned_split.json
```

默认 smoke：8 个请求、每个生成 16 tokens、最大并发 8、无限请求发送速率。
使用 HTTP 流式请求 `/v1/completions`，默认设置 `--ignore-eos`，便于获得足够多的 decode 样本。
脚本等待 `/v1/models` 就绪，并核对模型名。不要在同一个服务上同时发送其他请求或启动另一份 bench。

压测参数统一在 `configs/bench.yaml`，包括数据集、smoke/collect 负载、预热和 SLO。
填写后直接 `bash bench_tau.sh`。两脚本均支持 `--config /path/to/配置.yaml`；
优先级为命令行 > 非空环境变量 > 配置文件，最终命令和配置来源写入 meta。
配置内相对路径按 YAML 所在目录解析；启动脚本不再保存第二套实验默认值。

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `MODE` | `smoke` | `collect` 先预热再采集 |
| `NUM_PROMPTS` | smoke 8 / collect 1000 | 正式请求数 |
| `OUTPUT_LEN` | smoke 16 / collect 256 | 单请求目标输出长度 |
| `CONCURRENCY` | smoke 8 / collect 32 | 最多同时未完成的 HTTP 请求数，不控制 microbatch 大小 |
| `REQUEST_RATE` | `inf` | 目标请求数/秒；实际发送还受并发限制 |
| `BURSTINESS` | `1` | 有限速率下，1 按指数分布采样间隔；小于 1 更突发；inf 等间隔 |
| `IGNORE_EOS` | `1` | 0 允许自然结束；trace 检查仍要求 prefill/decode 覆盖 |
| `SEED` | `0` | 数据与到达间隔的随机种子 |
| `WARMUP_REQUESTS` | `32` | collect 的额外预热请求数 |
| `READY_TIMEOUT` | `300` | 等待服务就绪的秒数 |

可以直接修改 YAML 配置，也可以临时设置，例如：

```bash
REQUEST_RATE=5 BURSTINESS=inf CONCURRENCY=16 bash bench_tau.sh --mode collect
```

这是目标每秒 5 个请求、等间隔到达、最多 16 个未完成请求；达到并发上限时客户端等待。
`REQUEST_RATE=inf` 时不设到达间隔，`BURSTINESS` 不起作用。
Python 辅助程序只读取配置、记录采集区间和校验结果；预热与正式请求的执行顺序在 shell 中。
运行期间按 Ctrl+C 或向 bench shell 发送 TERM 会停止客户端，不关闭服务。

检查包含：客户端成功数、prefill/decode 在每个 PP rank 上的 compute 覆盖、
每个 emit 是否有全部 rank 的 compute、重复记录、计时区间和特征字段是否有效。
通过时返回 0；失败返回非零，并保留日志和报告。
`--dry-run` 只打印将执行的 bench 命令（collect 包含预热和正式两条），不下载数据或发送请求。

### 缺少 compute 时

查看 `bench/smoke_*/summary.json` 和服务日志。
如果有 `stage`，但 `compute` 为 0，说明仅有 worker 外层执行区间。
先确认服务已停止并用更新后的 `serve_tau.sh` 重新启动，不能只更新 bench。
新服务的 `server_meta.json` 中应包含：

```text
--worker-cls vllm.v1.worker.tau_ascend_worker.TauAscendWorker
```

`server.log` 中应出现 `Tau Ascend compute tracing installed`。
如果没有，检查是否仍在使用旧服务或旧代码。
如果选择了新 worker 仍缺少记录，需核对远端插件是否在初始化后替换了 runner，
或使用不同的执行入口；保留 `server_meta.json` 和 `server.log` 来定位。
不要跳过检查器继续采集。

`stage` 包括其包围的通信等待和调度开销；`done - emit` 则是一次 forward 从提交到回收的时长。
两者都不能直接除以 PP 数当作单 stage 的计算时间。
当前 compute 记录的是 host 区间加结束同步，
可能包括排队和调度开销，同步也可能改变流水线重叠。需要在 NPU 上验证测量边界；
脚本结构检查通过不等于纯 kernel 时间已经测准。

## 4. smoke 通过后：正式采集

```bash
bash bench_tau.sh \
  --mode collect \
  --dataset "$PWD/datasets/sharegpt.json" \
  --num-prompts 1000 \
  --output-len 256 \
  --concurrency 32 \
  --seed 0
```

如果使用自备 JSON，替换 `--dataset`。模型、端口、PP 从服务的 `server_meta.json` 自动读取。
当前采集流程要求 `TP=1`，因为 trace 没有区分 TP rank；也要求 `MIN_WAITING=0`。
正式模式先执行 32 个独立预热请求并检查 trace，通过后才执行指定的正式请求。
两段用 JSONL 的字节区间区分；拟合时只读取 `summary.json` 中
`trace` 字段下 `[trace_start_offset, trace_end_offset)` 的记录。
这些 offset 对应原始二进制文件，不能按文本字符偏移解释，也不能先修改/裁剪原文件。
检查器会流式读取 trace，不把整个 trace 文件加载进内存。

`--ignore-eos` 固定目标输出长度，仍受模型长度和采样器过滤影响；这组数据适合建立受控计算耗时样本，
不代表自然 EOS 下的线上完成时间分布。预热和正式采样默认同一个 seed，可能使用相同 prompt；
预热不用于训练，且“预热完成”不代表所有输入形状都已充分热身。

输出位于：

```text
output/
  latest -> <最近一次启动目录>/
  <时间>_<id>/
    server_meta.json        # 服务参数、命令、源码 commit、关键版本、trace 路径
    server.log
    bench/collect_<时间>_<id>/
      bench_meta.json       # 数据/SLO 快照与哈希、种子、命令、各阶段信息
      requests.jsonl        # 正式请求的分配与逐条结果（默认不保存生成文本）
      summary.json          # 正式统计、分组 goodput、trace 区间/检查、warmup 摘要
      error.log             # 仅失败时生成
```

每次启动创建新目录，原子更新 `output/latest`；旧目录保留。`latest` 表示最近启动，
不代表服务已就绪。bench 在启动时解析软链接，后续不会因 latest 切换而换输出目录。
`bench/<时间标识>/` 区分同一服务下不同轮压测，避免覆盖。
原始 trace 是 `server_meta.json` 的 `trace` 字段指向的单个 JSONL，默认写本机临时目录
（Linux 通常为 `/tmp/tau_<运行标识>.jsonl`），不实时写到项目目录。
设置 `--no-trace` 时该字段为 null，summary 的 `trace.enabled=false`。
不再生成 `packages.txt`、`npu.txt`、独立配置/分配副本和预热文件。

请求计划在发流量前保存；正常完成后原子替换为结果。中断时保留的 `planned` 行
只说明已计划，`success=null`，不表示请求尚未发送或已成功。临时日志、原生结果和
预热逐条明细写本机临时目录，正常退出或可捕获的中断后清理；失败日志归入 `error.log`。
serve/bench/campaign 不自动拟合或生成 parameters 文件；手动拟合见下文。

也可手动检查整个文件：

```bash
RUN_DIR="$(cd output/latest && pwd -P)"
TRACE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace"])' "$RUN_DIR/server_meta.json")
python tools/tau_batch_run.py check-trace "$TRACE" --pp 2
```

## 5. 为预测器扩大覆盖范围

仅一次 `MAX_NUM_SEQS=4` 运行不足以覆盖所有 n 和上下文长度。
可依次以 `MAX_NUM_SEQS=1、2、4、8` 重启服务，每次使用新 `RUN_DIR`，
在各服务上分别运行 `--output-len 64、256、512`，保持模型、TP/PP、token 预算等条件固定。
需要稳定复现实验时固定 seed；需要不同样本时显式改变 seed，并按不同运行划分训练和验证集。
`MAX_NUM_SEQS` 只是上限，训练时使用 trace 中实际的 `n`、`seq_lens`、`tokens` 和 `pp_rank`。

本分支 ShareGPT sampler 默认过滤 prompt 大于 1024 tokens、prompt+output 大于 2048 tokens 的样本。
因此即使服务设为 `MAX_MODEL_LEN=4096`，本流程也不会自然产生接近 4096 的上下文覆盖。
更长上下文需要另行扩展采样器或采用可控长度数据，不能把本轮拟合结果直接视为全长度通用预测器。

## 6. 全量分批采集与可选的手动拟合

共享 NFS 上的 JSONL 文件锁可能阻塞调度和 worker。当前默认把实时 trace 放在系统临时目录；
若该目录被平台改到共享盘，启动前通过 `TRACE=/tmp/tau_<唯一名称>.jsonl` 指定本地磁盘。
不要移动、清空或替换运行中的 trace。下面的 campaign 会把每批已完成的区间备份到输出目录。

在服务已经启动、smoke 通过且没有其他客户端请求的前提下运行：

```bash
cd /path/to/vllm
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python tools/tau_batch_campaign.py "$RUN_DIR" \
  --dataset "$PWD/datasets/sharegpt.json" \
  --index-dir "$RUN_DIR/sharegpt_index_seed0" \
  --output-dir "$RUN_DIR/campaign_all_01" \
  --shard-size 1000 --output-len 256 --concurrency 32 \
  > "$RUN_DIR/campaign_all_01.log" 2>&1
```

这是长时间任务，可在已有 tmux 会话中运行。
脚本不启动、不重启服务。输出目录必须是新目录，不会覆盖上次结果。
索引目录首次使用时依据模型 tokenizer 生成；复用时核对数据集 SHA256、模型路径和 seed。

“全量”指 sampler 可接受的全部原始行：使用前两轮对话，prompt 4–1024 tokens，
输出固定 256 tokens、ignore EOS；不是重放整段多轮会话，也不是使用原回答长度。
先按 seed 排列有效样本，再分为互不重叠的分片。每批只采一次正式数据，
使用 `smoke` 模式跳过额外 warmup，但显式覆盖请求数和输出长度。
已有首批正式采集可通过 `--adopt-report .../summary.json --adopt-count 1000`
纳入；必须使用同一完整数据集、seed 和输出长度。
`--await-exit` 可等待外部监督程序写入的、含 `returncode` 字段的退出 JSON。

输出包含：

- `status.json`：完成数、当前分片、状态及失败原因。
- `manifest.json`、`source_indices.json`：运行配置、工具哈希、原始行编号。
- `shard_*/trace.jsonl`、`summary.json`：已归档的正式区间及校验/归档信息，已重定位字节偏移。
- 每片 `bench_*/` 保留标准压测文件；不再自动拟合或生成 parameters 文件。

失败会停止，不自动重放部分成功的分片。可创建 `campaign_all_01/STOP_AFTER_BATCH`
让它在当前批完成并归档后停止。磁盘余量低于默认 10 GiB、服务不空闲或 trace 被替换时也停止。
当前脚本不自动恢复失败任务；检查 `status.json` 和已归档原始行编号后再决定续采范围。

也可以单独拟合已有的校验通过区间（只需要 NumPy，不导入 vLLM 或 NPU）：

```bash
python tools/tau_batch_fit.py \
  --trace /tmp/tau_<唯一名称>.jsonl \
  --report "$RUN_DIR/bench/collect_<时间>_<id>/summary.json" \
  --stage-layers 16 --output "$RUN_DIR/parameters_01.json"
```

输出单位为毫秒。每个 PP rank、prefill/decode 分别拟合：

- τ-Batch 有效形式：`a*n*s_max + b*n + c`。
- SCLS 四参数形式：`p1*n*s_max + p2*n + p3*s_max + p4`；decode 对应 `d1..d4`。
- 对照形式：`u1*s_sum + u2*n + u3*s_max + u4`，用于检验未 padding 的实际 token 总量是否更合适。

固定模型宽度时，τ-Batch 公式的 `alpha_proj*d_model²` 与 `beta` 都乘同一变量 `n`，
无法分别辨识。脚本只报告其和，以及除以每 stage 层数后的有效参数。
若设计矩阵秩不足，则输出 `rank_deficient`，不输出任意一组系数冒充唯一解。
最后 20% 的完整 wave 留作验证，并与训练集平均耗时这一常数基线比较。
`all_data_fit` 是使用全部样本的最终拟合，验证指标使用独立的 `training_fit`。
结构检查通过不等于计时口径已经校准：标签仍是含主机执行和结束同步的 runner 耗时。

## NPU 上验证 EOS 触发

保持通过 `serve_tau.sh` 启动的服务在线且空闲，在服务器另一终端运行：

```bash
python tools/tau_batch_eos_smoke.py --run-dir "/absolute/path/to/output/<本次服务目录>"
```

脚本使用 Python 标准库，只发送三个小请求，不启动或重启服务。
从该目录的 `server_meta.json` 读取模型路径和 trace 路径，要求开启 `compute` trace。
端口不是 8000 时加 `--base-url http://127.0.0.1:<端口>`。

| 用例 | 控制方式 | 必须观察到的结果 |
| --- | --- | --- |
| prefill EOS | 只允许 EOS token | 输出 1 个 EOS，`finish_reason=stop`，prefill 阶段触发一次 |
| decode EOS | 首个 token 禁止 EOS，之后强烈偏向 EOS | 输出非 EOS、EOS 两个 token，`finish_reason=stop`，decode 阶段触发一次 |
| ignore EOS | 只允许 EOS，开启 `ignore_eos`，限制 3 个输出 token | 输出 3 个 EOS，`finish_reason=length`，最后才触发一次 |

decode 用例使用 `min_tokens=1`、两个允许的 token 和 `logit_bias`，
并检查实际返回的 token ID；不把概率偏置直接当作成功证据。
默认从模型 `config.json` 读取 EOS 和 BOS，BOS 作为非 EOS 的对照 token。
模型配置有多个 EOS 或没有 BOS 时，需要明确传入
`--eos-token-id <实际EOS> --other-token-id <有效且非停止的token>`。

脚本核对请求 ID、实际输出 token、完成原因、执行阶段、每次执行的各 PP rank
计算记录，以及最后一次结果处理后的唯一 `eos` 事件。
请求参数、响应、原始 trace 片段和 `report.json` 存在脚本打印的
`<run-dir>/bench/eos_<时间>_<id>/`，失败返回非零退出码，不改写原 trace。

`eos` trace 位于 `planner.on_eos(event)` 调用之前；当前默认策略不执行动作。
这些测试验证实机走到触发点，结合单元测试验证回调转发，不验证回调后的重组策略。
当前触发口径是请求在输出处理时完成，因此达到长度上限也会触发。
