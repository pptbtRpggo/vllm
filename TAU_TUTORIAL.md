# τ-Batch 远端使用指南

适用：远端 Ascend 环境已配置好，项目位于 `/home/m7zhang/code/vllm`。
下面使用两张 NPU，`PP=2`、`TP=1`。当前支持 microbatch 调度、trace 采集与离线拟合；EOS 后重组策略尚未实现。

## 1. 更新代码

在远端终端执行；已有服务运行时，先按第 7 节关闭，再更新。

```bash
cd /home/m7zhang/code/vllm
git switch tau-batch/wave-planner-v013
git pull --ff-only
```

## 2. 终端 A：启动服务

先编辑 `configs/serve.yaml`：把 `MODEL` 填为实际模型路径，其他服务参数也在此文件配置。然后启动：

```bash
cd /home/m7zhang/code/vllm
bash serve_tau.sh
```

脚本直接调用已配置环境中的 `vllm serve`，自动创建本次运行目录和独立 trace 文件，并打印终端 B 的 benchmark 命令。
`configs/serve.yaml` 是服务参数入口，Tau 专用参数在 `SCHEDULERS.tau`；`serve_tau.sh` 负责读取配置、构造命令和启动。
配置读取工具使用项目已有的 PyYAML 依赖，不导入 vLLM/NPU；元信息工具记录最终配置。
终端 A 保持运行；长时间任务建议在已有 tmux 会话中启动。

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1` | 使用的 NPU 编号 |
| `PP` / `TP` | `2` / `1` | 流水线 / 张量并行度；当前采集要求 TP=1 |
| `MAX_NUM_SEQS` | `4` | 每个 microbatch 最多 4 个请求 |
| `MIN_WAITING` | `0` | 小批量和尾部请求不受等待数量阈值阻塞 |
| `RUN_DIR` | 自动生成 `output/<时间>_<id>/` | 配置、日志和测试结果目录 |
| `TRACE` | 系统临时目录下的独立 JSONL 文件（Linux 通常为 `/tmp`） | 原始 trace，避免共享文件系统拖慢执行 |

默认监听 `127.0.0.1:8000`，请求总长度上限为 4096 tokens，单次 forward token 预算为 8192。

需要调参时，可以直接修改 `configs/serve.yaml`，也可以临时覆盖，例如：

```bash
MAX_NUM_SEQS=8 bash serve_tau.sh /absolute/path/to/model
```

如果当前终端曾手动 `export RUN_DIR`、`RUN` 或 `TRACE`，先执行一次 `unset RUN_DIR RUN TRACE`，恢复自动路径。

配置优先级为 **命令行 > 非空环境变量 > YAML**。普通使用只编辑 YAML；meta 的
`launch_config` 记录配置路径、SHA256 和被覆盖的字段，原有字段/命令记录最终生效值。
配置文件中的相对路径相对于该 YAML 所在目录；命令行/环境变量中的相对路径仍相对于当前目录。
服务输出目录留空时继续自动使用项目 `output/`。自定义配置可复制完整 YAML，再用
`bash serve_tau.sh --config /path/to/serve.yaml` 或 `bash bench_tau.sh --config /path/to/bench.yaml`。
`--dry-run` 可以预览，不启动服务、不发送请求。
正式执行时，两脚本也会打印生效参数、配置来源及覆盖项、输出路径和完整命令。
bench 额外显示 SLO 档位/比例，并分别标明预热和正式请求数。

### 切换 vLLM 默认调度器

同一份 `configs/serve.yaml` 用 `SCHEDULER` 选择配置区：

```yaml
SCHEDULER: default  # 改回 tau 即恢复 TauScheduler
SCHEDULERS:
  tau:
    # 保留文件中现有的 Tau 配置
    MAX_NUM_SEQS: 4
    MAX_NUM_BATCHED_TOKENS: 8192
    MAX_MICROBATCHES: 0
    MIN_WAITING: 0
    TRACE_ENABLED: true
  default:
    MAX_NUM_SEQS: null
    MAX_NUM_BATCHED_TOKENS: null
    ENABLE_CHUNKED_PREFILL: null
    ENABLE_PREFIX_CACHING: null
    ASYNC_SCHEDULING: null
    SCHEDULING_POLICY: null
```

模型、设备、PP/TP、显存比例和输出目录仍使用文件顶部的公共设置。
`default` 下的 `null` 表示不传对应参数，由当前 vLLM/平台决定；最终解析值看服务启动日志。
要指定对照条件，例如同样的 batch 预算，把该区的两个上限改为 `4`、`8192`。
布尔项可设 true/false/null，策略可设 fcfs/priority/null。参数组合仍需满足当前
vLLM 与 Ascend 平台约束；这些选项可传入，不表示每种组合都可运行。

默认模式不传 `--scheduler-cls`，由 vLLM 选择自身调度器，也不传 Tau 的 worker 或 trace 参数。
当前 Tau trace 不支持默认调度器；只改 `SCHEDULER` 即会按模式关闭采集，显式 `--trace` 会报错。
Tau 固定关闭 chunked prefill、prefix caching、async scheduling，这些约束会显示在启动参数中。
每次切换需关闭旧服务再启动，新运行仍更新 `output/latest`。

```bash
bash serve_tau.sh                       # 使用 YAML 选择
bash serve_tau.sh --scheduler default   # 可选的临时覆盖
bash bench_tau.sh                       # 沿用同一份 bench 配置
```

bench 继续按每条请求的 SLO 评估 goodput，默认模式跳过 Tau trace 校验。
SLO 阈值会随请求发送；默认调度器不会因此自动按 TTFT/TPOT SLO 调度。
普通参数覆盖仍遵循命令行 > 非空环境变量 > 当前配置区；对照实验留意继承的
`MAX_NUM_SEQS` 等环境变量是否覆盖了 YAML。

## 3. 终端 B：检查服务并运行 smoke

先在 `configs/bench.yaml` 填写 `DATASET`；已有数据时保留 `DOWNLOAD: false`，需要下载时改为 true。

服务启动时会将运行目录记录到 `output/latest`，bench 自动读取，无需复制路径。它表示最近一次启动的配置，不代表服务已经就绪；bench 会等待服务就绪。

同时启动多个服务时，可用 `bash bench_tau.sh /指定运行目录 --mode smoke` 选择服务；显式目录优先于 `RUN_DIR` 环境变量，最后才使用 `latest`。

```bash
cd /home/m7zhang/code/vllm
bash bench_tau.sh
```

脚本自动读取模型和端口、等待服务就绪，然后发送 8 个请求，每个生成 16 tokens。
默认读取 `datasets/sharegpt.json`；文件缺失时仅在 `DOWNLOAD: true` 或传 `--download` 时下载。
使用其他数据文件时加 `--dataset /实际路径/sharegpt.json`。

**通过标志：**检查报告显示 `"passed": true`，末尾打印“压测完成”。结果保存在打印的 `bench/smoke_…/` 目录。

同一服务上，各项测试和采集依次执行，不要同时运行多个客户端。

`configs/bench.yaml` 集中配置数据集、流量、预热和 SLO；`MODES.smoke`、`MODES.collect` 分别配置请求数、输出长度和并发数。
设置 `SLO.enabled: true` 即按同文件中的档位/比例发送请求，省去独立 SLO JSON。
它读取 ShareGPT prompt，通过 HTTP 请求 `/v1/completions`；microbatch 由服务端组建。

| 参数 | 默认值 | 控制什么 |
| --- | --- | --- |
| `MODE` | `smoke` | 小测试；改成 `collect` 则先预热、再正式采集 |
| `NUM_PROMPTS` | smoke 8 / collect 1000 | 正式请求总数 |
| `OUTPUT_LEN` | smoke 16 / collect 256 | 每个请求的目标输出 token 数 |
| `CONCURRENCY` | smoke 8 / collect 32 | 最多同时未完成的请求数 |
| `REQUEST_RATE` | `inf` | 目标请求数/秒；`inf` 表示尽快发送 |
| `BURSTINESS` | `1` | 有限速率下，`1` 为随机间隔、`inf` 为等间隔 |
| `IGNORE_EOS` | `1` | 忽略 EOS，便于采固定输出长度；`0` 允许自然结束 |

例如，目标平均每秒 5 个请求，最多 16 个并发：

```bash
REQUEST_RATE=5 CONCURRENCY=16 bash bench_tau.sh
```

改成目标等间隔到达：

```bash
REQUEST_RATE=5 BURSTINESS=inf CONCURRENCY=16 bash bench_tau.sh
```

并发上限达到后，请求会在客户端等待，因此实际发送速率可能低于目标速率。

## 4. 采集 ShareGPT trace

### 先采集 1000 个请求

在终端 B 执行：

```bash
bash bench_tau.sh --mode collect
```

先执行 32 个预热请求，再采集 1000 个正式请求；生成时忽略 EOS，目标输出长度为 256 tokens。
每次自动创建新的结果目录。下文的 `RUN_DIR` 指服务运行目录，`RESULT_DIR` 指本次 bench 打印的结果目录。

| 文件 | 内容 |
| --- | --- |
| `$RUN_DIR/server_meta.json` | 模型、服务参数、trace 路径与代码版本 |
| `$RUN_DIR/server.log` | 服务日志 |
| `server_meta.json` 中的 `trace` 路径 | 实时原始 trace |
| `$RESULT_DIR/bench_meta.json` | 数据来源/SLO 配置快照、种子、命令与服务引用 |
| `$RESULT_DIR/requests.jsonl` | 正式请求分配、长度、实测 TTFT/TPOT、success/attained |
| `$RESULT_DIR/summary.json` | 正式统计、分组 goodput、trace 检查及字节区间；warmup 为独立字段 |
| `$RESULT_DIR/error.log` | 仅失败时生成 |

只有 `summary.json` 中 `trace.passed=true` 才继续拟合。

### 可选：分批采集全部有效样本

这是独立的一次采集，会重新遍历数据集，不会自动接续上面的 1000 个请求。

```bash
RUN_DIR="$(cd output/latest && pwd -P)"
export CAMPAIGN_DIR="$RUN_DIR/campaign_01"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python tools/tau_batch_campaign.py "$RUN_DIR" \
  --dataset "$PWD/datasets/sharegpt.json" \
  --index-dir "$RUN_DIR/sharegpt_index_seed0" \
  --output-dir "$CAMPAIGN_DIR" \
  --shard-size 1000 --output-len 256 --concurrency 32 \
  > "$RUN_DIR/campaign_01.log" 2>&1
```

每批归档 trace，不自动拟合。“全部”指采样器接受的样本：使用前两轮对话，prompt 长度为 4–1024 tokens。

在另一个终端将 `RUN_DIR` 设为同一运行目录后，查看进度，或让采集在当前批结束后停止：

```bash
cat "$RUN_DIR/campaign_01/status.json"

# 需要停止采集时执行；服务会继续运行。
touch "$RUN_DIR/campaign_01/STOP_AFTER_BATCH"
```

每批 trace 在 `campaign_01/shard_*/`；不生成 parameters 文件。

## 5. 可选：手动拟合 stage 耗时预测模型

对第 4 节的单批正式采集执行；将 `RESULT_DIR` 替换为该次 bench 打印的结果目录：

```bash
RUN_DIR="$(cd output/latest && pwd -P)"
RESULT_DIR="/本次bench打印的结果目录"
TRACE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace"])' "$RUN_DIR/server_meta.json")
python tools/tau_batch_fit.py \
  --trace "$TRACE" \
  --report "$RESULT_DIR/summary.json" \
  --output "$RESULT_DIR/parameters.json"
```

终端会打印各 stage 的样本数、参数和验证误差，完整结果保存在 `parameters.json`。
输出文件必须尚不存在。

| 结果字段 | 看什么 |
| --- | --- |
| `groups` | 分 PP rank、prefill/decode 的结果 |
| `all_data_fit.coefficients` | 使用全部样本拟合的参数 |
| `validation` | 使用训练集参数在留出数据上的误差，例如 `rmse_ms` |
| `status=rank_deficient` | 当前样本不足以唯一确定参数，不输出该组系数 |

同时比较 `tau_affine`、`scls_bilinear`、`unpadded_comparison` 三种模型。
计时单位为毫秒，标签是 **runner 调用开始至返回并完成 NPU 同步的耗时**，不是纯 kernel 时间。
这是离线拟合，参数不会自动加载到在线调度器中。

## 6. 测试 EOS 触发

保持服务空闲，在终端 B 执行：

```bash
python tools/tau_batch_eos_smoke.py --run-dir "$RUN_DIR"
```

预期输出：

```text
prefill_eos: PASS
decode_eos: PASS
ignore_eos: PASS
```

结果在打印的 `bench/eos_…/` 目录，包含请求、响应、trace 片段和 `report.json`。
当前钩子在请求完成时触发，因此 EOS 和达到长度上限都会触发；本测试不执行重组策略。
若模型没有单一 EOS/BOS 配置，按提示指定 `--eos-token-id` 和 `--other-token-id`。

## 7. 关闭服务

先结束采集。服务仍在终端 A 前台运行时，按一次 **Ctrl+C**，等待退出。

找不到原终端，或启动脚本已退出但服务仍在时，在服务器执行：

```bash
pgrep -af '[v]llm.* serve'
read -r -p '请输入上面对应模型和端口的服务 PID: ' SERVER_PID
kill -TERM "$SERVER_PID"
```

等待十几秒，再检查进程和 NPU：

```bash
pgrep -af '[v]llm.* serve|[V]LLM::'
npu-smi info
```

确认对应 API、EngineCore 和 worker 已退出。不要沿用上一次服务的 PID。

服务完全停止后，在保留第 3 节变量的终端 B 备份本地 trace：

```bash
TRACE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace"])' "$RUN_DIR/server_meta.json")
cp -n -- "$TRACE" "$RUN_DIR/trace.jsonl"
```

保留原始文件，不要在服务运行时删除、清空或替换 trace。

## 8. 常见问题

| 问题 | 处理 |
| --- | --- |
| 提示运行目录或 trace 已存在 | 换新的运行名称，并同步修改两个终端的路径 |
| `HTTP 200` 未出现 | 查看 `server.log`，确认模型加载成功和端口一致 |
| 找不到 `tau_batch_run.py serve` | 使用第 7 节命令查实际 API 服务进程 |
| trace 缺少 `compute` | 确认服务使用更新后的 `serve_tau.sh`；日志应出现 `Tau Ascend compute tracing installed` |
| 采集很慢 | 确认 trace 位于本地 `/tmp`；先用 smoke 排查，避免直接扩大请求数 |
| 全量采集中途失败 | 查看 `status.json` 和对应日志；脚本不会自动续跑 |
| EOS 测试请求连接失败 | 默认端口为 8000；其他端口加 `--base-url http://127.0.0.1:端口` |

更多参数与计时细节见同目录下的 `TAU_TRACE.md`。
