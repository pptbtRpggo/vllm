# 用 GSM8K 测试不同 HTTP 并发下的性能

脚本：[`benchmark_gsm8k.sh`](../../examples/online_serving/hetero_pp/benchmark_gsm8k.sh)。
它仅调用已有的 EvalScope，向已启动的 OpenAI-compatible API 发请求。
不启动/停止模型、Docker、Ray，不修改 NPU 或服务参数，不安装依赖。

## 1. 在一个客户端执行

选择能访问 vLLM head API、且已经安装 EvalScope perf 的环境。
不需要在两台 NPU 机器上同时执行；那会叠加请求，改变总并发。
客户端本身不需要 NPU。

准备本地 GSM8K **JSONL 文件**，每行一个对象，至少包含 `question`：

```json
{"question": "A shop has 12 apples and sells 5. How many remain?", "answer": "7"}
```

上面只是格式示例；正式比较请使用同一份真实 GSM8K 文件。
脚本不会自动下载数据或把示例题当成 GSM8K。
`answer` 不会发送给模型，也不会在这项性能测试中评分。
原始 Parquet/Arrow 需要先导出为这种 JSONL。
若文件不够长，EvalScope 可能循环复用题目；默认每轮含预热共 576 请求，
建议至少准备这么多条，并记录是否发生重复。

```bash
bash examples/online_serving/hetero_pp/benchmark_gsm8k.sh \
  --dataset-path /path/to/gsm8k_test.jsonl
```

默认值：

| 项目 | 默认 |
| --- | --- |
| 服务 | `http://127.0.0.1:8000/v1/chat/completions` |
| API 模型名 | `hetero-model` |
| 并发 | 8、16、32、64，依次执行 |
| 正式请求 | 每个并发 512，共 2048 |
| 预热请求 | 每个并发 64，另列，不计入正式指标 |
| 输出 | 最多 128 token，temperature=0，seed=42，正常 EOS 停止 |
| 流式 | 开启，便于测客户端 TTFT/TPOT |
| 超时 | 连接 10 秒、读取 600 秒、单请求总计 1800 秒 |

预热和正式统计采用 EvalScope 自己的实现。新版本的预热到正式阶段可有重叠，
不是额外添加一次“整批排空屏障”。
需使用支持多值 `--parallel` / `--number` 和 `--warmup-num` 的 EvalScope；
脚本先检查 `perf --help` 中的必要参数。

## 2. 自己指定地址、并发或长度

```bash
bash examples/online_serving/hetero_pp/benchmark_gsm8k.sh \
  --dataset-path /data/gsm8k_test.jsonl \
  --url http://141.147.135.53:8000 \
  --model hetero-model \
  --parallel 8 16 32 64 \
  --number 512 \
  --warmup-num 64 \
  --max-tokens 128
```

`--model` 是 `/v1/models` 显示的服务模型名，不是客户端的权重路径。
`--url` 可写根地址、`/v1` 或完整 `/v1/chat/completions`。
接口需要有可用 chat template，沿用已有服务的模板即可。
此简易入口用于无鉴权的本地实验服务，不提供 API key 参数。

**服务的 context 容量须覆盖题目、chat template 和输出上限。**
现有启动脚本的默认 context=512 可能不足以容纳部分 GSM8K 题目加 128 token 输出；
本脚本不会擅自重启或改变服务，需在启动服务时选择合适的 `--max-model-len`，
或调整本次输出上限，并查看报告中的失败请求数。
客户端并发 64 也不等于服务实际组批 64；`max_num_seqs` 等服务配置仍决定执行批次。

如 EvalScope 不在 PATH，可以指定已有环境的可执行文件：

```bash
bash examples/online_serving/hetero_pp/benchmark_gsm8k.sh \
  --dataset-path /data/gsm8k_test.jsonl \
  --evalscope /path/to/env/bin/evalscope
```

只有客户端本地存在 tokenizer 时才加 `--tokenizer-path /path/to/tokenizer`。
脚本不会将服务容器内的模型路径当成客户端路径。

不发请求的预览：

```bash
bash examples/online_serving/hetero_pp/benchmark_gsm8k.sh \
  --dataset-path /data/gsm8k_test.jsonl --dry-run
```

预览仍要求数据文件存在，但不要求已安装 EvalScope，不创建结果目录。

## 3. 看结果

默认保存到当前执行用户的：

```text
~/vllm_hetero_pp_logs/evalscope_<UTC时间>_<唯一后缀>/
├── command.sh              # 本次完整命令，便于记录和复查
├── evalscope_version.txt   # 版本命令输出；旧版不支持时也保留原输出
├── run.log                 # 完整终端输出，包含每轮指标表
└── results/                # EvalScope 原生结果、逐请求记录及报告
```

EvalScope 原生目录/文件名可能随版本变化；以终端打印的路径为准。
默认自动生成唯一目录；`--output-dir /path/to/new-run` 可以指定一个**尚不存在**的目录，
避免覆盖旧记录。CLI 错误会保留日志并向上返回非零退出码，不自动重跑。
CLI 退出码为 0 不一定意味着全部请求成功，仍须检查 Success/Failed。

重点看各并发的 TTFT、TPOT、Latency/E2E、输出 tokens/s、requests/s，
以及 mean/P50/P90/P95/P99。脚本保留 EvalScope 原生统计，不重算或修改指标。

## 4. 这项测试的口径

- 默认 closed-loop：完成一个完整流式请求，再补发一个；
  `--parallel` 是 HTTP 在途请求上限，不是 NPU 执行并发。
- TTFT/TPOT/E2E 是客户端观察口径，包含 HTTP/网络、服务排队、执行等开销，
  不是内部 stage/kernel 的运行时间。
- TPOT 是每请求首 token 后的平均耗时；ITL 基于客户端流式块间隔，
  流式块不一定严格等于一个 token。
- `max-tokens=128` 是上限，不强制所有回复相同长度。正常 EOS 会提前停止。
  对比策略时需同时看实际输入/输出 token 分布，不能把长短不同回复直接当同等工作量。
- 性能压测不评 GSM8K 正确率；准确率需另用 `evalscope eval`。
- 所有策略应使用同一份题目、模板、生成设置及一致的缓存策略。
  多轮重复题目可能命中 prefix cache；脚本不刷新服务器缓存。

官方接口依据：
[性能参数](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/parameters.html)、
[指标与报告](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html)、
[本地 JSONL question 读取器](https://github.com/modelscope/evalscope/blob/main/evalscope/perf/plugin/datasets/openqa.py)。

验证范围：脚本用隔离的 CLI 替身检查参数转发、日志、错误码、路径空格、
不覆盖、dry-run 及 sh→bash；没有用这些测试冒充真实 EvalScope/HTTP/NPU 压测。
