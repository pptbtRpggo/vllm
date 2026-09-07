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

只需把下面的路径换成实际模型目录：

```bash
cd /home/m7zhang/code/vllm
bash serve_tau.sh /absolute/path/to/model
```

脚本直接调用已配置环境中的 `vllm serve`，自动创建本次运行目录和独立 trace 文件，并打印终端 B 的 benchmark 命令。
打开 `serve_tau.sh`：顶部是全部可调参数及中文说明，下面是完整的 `vllm serve` 参数列表。
Python 辅助程序只保存配置记录，不负责启动服务。
终端 A 保持运行；长时间任务建议在已有 tmux 会话中启动。

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1` | 使用的 NPU 编号 |
| `PP` / `TP` | `2` / `1` | 流水线 / 张量并行度；当前采集要求 TP=1 |
| `MAX_NUM_SEQS` | `4` | 每个 microbatch 最多 4 个请求 |
| `MIN_WAITING` | `0` | 小批量和尾部请求不受等待数量阈值阻塞 |
| `RUN_DIR` | 自动生成 `trace_runs/<时间>_<id>/` | 配置、日志和测试结果目录 |
| `TRACE` | 系统临时目录下的独立 JSONL 文件（Linux 通常为 `/tmp`） | 原始 trace，避免共享文件系统拖慢执行 |

默认监听 `127.0.0.1:8000`，请求总长度上限为 4096 tokens，单次 forward token 预算为 8192。

需要调参时，可以直接修改脚本顶部的默认值，也可以临时覆盖，例如：

```bash
MAX_NUM_SEQS=8 bash serve_tau.sh /absolute/path/to/model
```

如果当前终端曾手动 `export RUN_DIR`、`RUN` 或 `TRACE`，先执行一次 `unset RUN_DIR RUN TRACE`，恢复自动路径。

## 3. 终端 B：检查服务并运行 smoke

服务启动时会将运行目录记录到 `trace_runs/latest`，bench 自动读取，无需复制路径。它表示最近一次启动的配置，不代表服务已经就绪；bench 会等待服务就绪。

同时启动多个服务时，可用 `bash bench_tau.sh /指定运行目录 --mode smoke` 选择服务；显式目录优先于 `RUN_DIR` 环境变量，最后才使用 `latest`。

```bash
cd /home/m7zhang/code/vllm
bash bench_tau.sh --mode smoke --download
```

脚本自动读取模型和端口、等待服务就绪，然后发送 8 个请求，每个生成 16 tokens。
默认读取 `datasets/sharegpt.json`；文件已存在时直接复用，否则下载 ShareGPT。
使用其他数据文件时加 `--dataset /实际路径/sharegpt.json`。

**通过标志：**检查报告显示 `"passed": true`，末尾打印“采集及覆盖检查完成”。结果保存在打印的 `bench/smoke_…/` 目录。

同一服务上，各项测试和采集依次执行，不要同时运行多个客户端。

`bench_tau.sh` 顶部分别列出 smoke 和 collect 配置，可直接修改对应数值；下面直接调用 `vllm bench serve`。
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
| `$RUN_DIR/run.json` | 模型、服务参数、trace 路径与代码版本 |
| `$RUN_DIR/server.log` | 服务日志 |
| `run.json` 中的 `trace` 路径 | 实时原始 trace |
| `$RESULT_DIR/result.json` | 正式请求的 benchmark 指标 |
| `$RESULT_DIR/result_trace_check.json` | 正式采集是否通过，以及对应的 trace 字节区间 |
| `$RESULT_DIR/warmup_trace_check.json` | 独立的预热区间，不用于拟合 |

只有 `result_trace_check.json` 中 `passed=true` 才继续拟合。

### 可选：分批采集全部有效样本

这是独立的一次采集，会重新遍历数据集，不会自动接续上面的 1000 个请求。
下面的 `16` 仅适用于每个 stage 分配 16 层的模型，换模型时必须修改。

```bash
RUN_DIR="$(cd trace_runs/latest && pwd -P)"
export CAMPAIGN_DIR="$RUN_DIR/campaign_01"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python tools/tau_batch_campaign.py "$RUN_DIR" \
  --dataset "$PWD/datasets/sharegpt.json" \
  --index-dir "$RUN_DIR/sharegpt_index_seed0" \
  --output-dir "$CAMPAIGN_DIR" \
  --stage-layers 16 \
  --shard-size 1000 --output-len 256 --concurrency 32 \
  > "$RUN_DIR/campaign_01.log" 2>&1
```

每批归档 trace 并自动拟合。“全部”指采样器接受的样本：使用前两轮对话，prompt 长度为 4–1024 tokens。

在另一个终端将 `RUN_DIR` 设为同一运行目录后，查看进度，或让采集在当前批结束后停止：

```bash
cat "$RUN_DIR/campaign_01/status.json"

# 需要停止采集时执行；服务会继续运行。
touch "$RUN_DIR/campaign_01/STOP_AFTER_BATCH"
```

每批结果在 `campaign_01/shard_*/`；全部完成后生成 `campaign_01/parameters_all.json`。

## 5. 拟合 stage 耗时预测模型

对第 4 节的单批正式采集执行；将 `RESULT_DIR` 替换为该次 bench 打印的结果目录：

```bash
RUN_DIR="$(cd trace_runs/latest && pwd -P)"
RESULT_DIR="/本次bench打印的结果目录"
TRACE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace"])' "$RUN_DIR/run.json")
python tools/tau_batch_fit.py \
  --trace "$TRACE" \
  --report "$RESULT_DIR/result_trace_check.json" \
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
TRACE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["trace"])' "$RUN_DIR/run.json")
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
