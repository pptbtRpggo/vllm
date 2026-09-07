# Ascend 上运行 τ-Batch / ShareGPT trace 采集

本流程面向当前分支、单机 Ascend、TP=1、PP=2 和纯文本生成模型。
`serve_tau.sh` 启动服务，`bench_tau.sh` 下载/读取 ShareGPT、压测、检查 trace。
两者均使用当前环境的 `python`，也可以通过 `PYTHON=/path/to/python` 指定。
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
脚本通过 `python -m vllm.entrypoints.cli.main` 并设置仓库 `PYTHONPATH` 来使用当前源码；
环境仍须有兼容的 vLLM 编译产物、Ascend 插件和模型依赖。
仅有一个未构建的源码 checkout 不等于环境安装完成。

## 2. 终端 A：启动服务

以下模型路径需要替换；`RUN_DIR` 必须是尚不存在的目录，脚本负责创建。

```bash
cd /path/to/vllm
export RUN_DIR="$PWD/trace_runs/mb4_$(date +%Y%m%d_%H%M%S)"
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MAX_REQS_PER_MB=4 \
bash serve_tau.sh /absolute/path/to/model --run-dir "$RUN_DIR"
```

服务保持前台运行，Ctrl+C 停止。启动时会打印实际 `RUN_DIR` 和终端 B 命令。
不再通过删除正在写入的 trace 来开启新一轮；重启时使用新目录。
`MODEL=/absolute/path/to/model bash serve_tau.sh` 形式仍然可用。

默认参数如下，可通过同名环境变量覆盖：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1` | 两张 NPU |
| `TP` / `PP` | `1` / `2` | 张量/流水线并行度 |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 本机访问地址；跨机访问时自行设置 HOST |
| `MAX_MODEL_LEN` | `4096` | 请求总长度上限 |
| `MAX_NUM_SEQS` | `32` | 单次 forward 请求容量 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 单次 forward token 预算 |
| `MAX_REQS_PER_MB` | `4` | 单个 microbatch 请求数上限 |
| `MAX_MICROBATCHES` | `0` | 不额外限制 wave 的 microbatch 数 |
| `MIN_WAITING` | `0` | 不因等待阈值阻塞小样本或最后几个请求 |
| `GPU_MEM` | `0.90` | 传给 vLLM 的 `--gpu-memory-utilization`，Ascend 沿用该名字 |

相对旧脚本：`MIN_WAITING` 从 `MAX_NUM_SEQS` 改为 `0`；token 预算从 4096 改为 8192；
HOST 改为本机地址；trace 默认使用每次启动的独立目录。
8192 是本轮采集的预算选择，并非 Ascend 的强制要求，也不是 KV cache 容量。
当改变 microbatch 上限时，仍可能因 token 预算或 KV 余量而形成更小的 microbatch。
EOS 回调策略没有变化。

## 3. 终端 B：先 smoke

激活相同 Python 环境，把以下目录替换为终端 A 打印的完整路径。
两个终端不会自动共享 `RUN_DIR` 变量。

```bash
cd /path/to/vllm
export RUN_DIR=/absolute/path/to/vllm/trace_runs/mb4_20260907_120000
bash bench_tau.sh "$RUN_DIR" --download
```

`--download` 仅在默认 `datasets/sharegpt.json` 不存在时下载
[ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
的 `ShareGPT_V3_unfiltered_cleaned_split.json`，已有文件会复用。
无法访问 Hugging Face 时，可自行把该 JSON 传到服务器，再运行：

```bash
bash bench_tau.sh "$RUN_DIR" --dataset /data/ShareGPT_V3_unfiltered_cleaned_split.json
```

默认 smoke：32 个请求、每个生成 64 tokens、最大并发 32、无限请求发送速率。
使用 `/v1/completions`，并设置 `--ignore-eos`，便于获得足够多的 decode 样本。
脚本等待 `/v1/models` 就绪，并核对模型名。不要在同一个服务上同时发送其他请求或启动另一份 bench。

检查包含：客户端成功数、prefill/decode 在每个 PP rank 上的 compute 覆盖、
每个 emit 是否有全部 rank 的 compute、重复记录、计时区间和特征字段是否有效。
通过时返回 0；失败返回非零，并保留日志和报告。
`--dry-run` 只打印最终 bench 命令，不下载数据或发送请求。

### 缺少 compute 时

查看 `bench/smoke_*/result_trace_check.json` 和服务日志。
如果有 `stage`，但 `compute` 为 0，说明仅有 worker 外层执行区间。
先确认服务已停止并用更新后的 `serve_tau.sh` 重新启动，不能只更新 bench。
新服务的 `run.json` 中应包含：

```text
--worker-cls vllm.v1.worker.tau_ascend_worker.TauAscendWorker
```

`server.log` 中应出现 `Tau Ascend compute tracing installed`。
如果没有，检查是否仍在使用旧服务或旧代码。
如果选择了新 worker 仍缺少记录，需核对远端插件是否在初始化后替换了 runner，
或使用不同的执行入口；保留 `run.json`、`packages.txt` 和 `server.log` 来定位。
不要跳过检查器继续采集。

`stage` 包括其包围的通信等待和调度开销；`done - emit` 则是一次 forward 从提交到回收的时长。
两者都不能直接除以 PP 数当作单 stage 的计算时间。
当前 compute 记录的是 host 区间加结束同步，
可能包括排队和调度开销，同步也可能改变流水线重叠。需要在 NPU 上验证测量边界；
脚本结构检查通过不等于纯 kernel 时间已经测准。

## 4. smoke 通过后：正式采集

```bash
bash bench_tau.sh "$RUN_DIR" \
  --mode collect \
  --dataset "$PWD/datasets/sharegpt.json" \
  --num-prompts 1000 \
  --output-len 256 \
  --concurrency 32 \
  --seed 0
```

如果使用自备 JSON，替换 `--dataset`。模型、端口、PP 从服务的 `run.json` 自动读取。
当前采集流程要求 `TP=1`，因为 trace 没有区分 TP rank；也要求 `MIN_WAITING=0`。
正式模式先执行 32 个独立预热请求并检查 trace，通过后才执行指定的正式请求。
两段用 JSONL 的字节区间区分；拟合时只读取 `result_trace_check.json` 中
`[trace_start_offset, trace_end_offset)` 的记录。
这些 offset 对应原始二进制文件，不能按文本字符偏移解释，也不能先修改/裁剪原文件。
检查器会流式读取 trace，不把整个 trace 文件加载进内存。

`--ignore-eos` 固定目标输出长度，仍受模型长度和采样器过滤影响；这组数据适合建立受控计算耗时样本，
不代表自然 EOS 下的线上完成时间分布。预热和正式采样默认同一个 seed，可能使用相同 prompt；
预热不用于训练，且“预热完成”不代表所有输入形状都已充分热身。

输出位于：

```text
RUN_DIR/
  run.json                 # 服务参数、命令、源码 commit、环境版本
  packages.txt             # pip freeze
  npu.txt                  # npu-smi info
  server.log
  server_exit.json         # 服务停止后写出
  trace.jsonl
  bench/collect_<时间>_<id>/
    dataset.json           # 数据集路径、大小、SHA256
    warmup.json / warmup.log / warmup_command.json / warmup_trace_check.json
    result.json / result.log / result_command.json / result_trace_check.json
```

也可手动检查整个文件：

```bash
python tools/tau_batch_run.py check-trace "$RUN_DIR/trace.jsonl" --pp 2
```

## 5. 为预测器扩大覆盖范围

仅一次 `MAX_REQS_PER_MB=4` 运行不足以覆盖所有 n 和上下文长度。
可依次以 `MAX_REQS_PER_MB=1、2、4、8` 重启服务，每次使用新 `RUN_DIR`，
在各服务上分别运行 `--output-len 64、256、512`，保持模型、TP/PP、token 预算等条件固定。
需要稳定复现实验时固定 seed；需要不同样本时显式改变 seed，并按不同运行划分训练和验证集。
`MAX_REQS_PER_MB` 只是上限，训练时使用 trace 中实际的 `n`、`seq_lens`、`tokens` 和 `pp_rank`。

本分支 ShareGPT sampler 默认过滤 prompt 大于 1024 tokens、prompt+output 大于 2048 tokens 的样本。
因此即使服务设为 `MAX_MODEL_LEN=4096`，本流程也不会自然产生接近 4096 的上下文覆盖。
更长上下文需要另行扩展采样器或采用可控长度数据，不能把本轮拟合结果直接视为全长度通用预测器。
