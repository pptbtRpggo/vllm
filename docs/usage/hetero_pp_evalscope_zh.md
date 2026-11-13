# 用 GSM8K 测试不同 HTTP 并发下的性能

脚本：[`benchmark_gsm8k.sh`](../../examples/online_serving/hetero_pp/benchmark_gsm8k.sh)。
它仅调用已有的 EvalScope，向已启动的 OpenAI-compatible API 发请求。
不启动/停止模型、Docker、Ray，不修改 NPU 或服务参数，不安装依赖。

## 0. 推荐：新建专用 EvalScope 客户端容器

旧环境 EvalScope 缺少预热参数时，不需要修改原推理容器或宿主机 Python。
在 **vLLM head 所在宿主机**的项目根目录运行：

```bash
bash examples/online_serving/hetero_pp/start_evalscope_docker.sh
```

默认复用普通 `quay.io/ascend/vllm-ascend:v0.23.0-openeuler` 镜像，
本地没有时尝试拉取。使用镜像自带的 Python 3.12 在客户端专属的
`/tmp/evalscope-client/.venv` 中自动安装最新版 `evalscope[perf]`，
不使用 Python 3.9 的旧包，不修改镜像原有 Python 包或原服务容器。
首次需要联网下载依赖；安装日志与实际安装版本清单保存到宿主机结果目录。
可用 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` 指定包索引，
或用 `--evalscope-version 版本号` 固定 EvalScope 版本。

安装及必要参数检查通过后进入 Bash，不会自动发请求。直接手动执行：

```bash
# 先试 4 个请求
benchmark_gsm8k.sh --parallel 1 --number 4 --warmup-num 0 --max-tokens 32

# 成功后再测 8 / 16 / 32 / 64，每档 512 正式 + 64 预热
benchmark_gsm8k.sh
```

`evalscope` 与 `benchmark_gsm8k.sh` 已自动加入容器 PATH。
脚本和项目内置的 GSM8K 已实际复制到 `/tmp/evalscope-client/tools/`，
不需要手动找路径、重新下载数据或挂载模型。需要指定其他服务时：

```bash
benchmark_gsm8k.sh --url http://服务地址:8000 --model 实际服务模型名
```

客户端使用 host 网络，同一宿主机上可访问 `http://127.0.0.1:8000`。
不分配 NPU、不挂载驱动，不需要 `--privileged` 或关闭 seccomp，
也不启动 Ray/vLLM。请保持模型服务运行，并且只运行一个压测客户端。

客户端不带 `--rm`，退出后保留。实际容器名与宿主机结果目录会打印出来：

```bash
# 容器退出后再次进入（复用已安装环境）
docker start -ai 实际客户端容器名
# 容器仍在运行时，另开一个终端
docker exec -it 实际客户端容器名 /bin/bash
```

压测结果默认写入容器 HOME 下的 `vllm_hetero_pp_logs`，它指向 `/results`，
对应宿主机 `$HOME/vllm_hetero_pp_logs/evalscope_client_唯一标识/`。
容器按当前宿主机 UID/GID 运行，结果归当前用户所有。
也可用 `--output-dir /你的路径/全新目录` 指定宿主机结果目录。
删除客户端容器会丢失其 venv，但已经挂载回宿主机的结果保留。

验证范围：本地用隔离的 Docker/包下载替身验证编排和安装错误处理，
并实际执行容器准备脚本、复制数据、打开 Bash 和运行压测 dry-run。
尚未在真实 Docker 镜像中完成联网安装或向模型服务发压，不能将这些测试视为实机验收。

## 1. 在一个客户端执行

选择能访问 vLLM head API、且已经安装 EvalScope perf 的环境。
不需要在两台 NPU 机器上同时执行；那会叠加请求，改变总并发。
客户端本身不需要 NPU。

### 当前环境：只有 Python 3.9 中的包，没有 evalscope 命令

直接指定已安装 EvalScope 的 Python，不需要修改 PATH 或创建启动包装文件。
在项目根目录先跑 4 条请求：

```bash
bash examples/online_serving/hetero_pp/benchmark_gsm8k.sh \
  --python python3.9 \
  --parallel 1 --number 4 --warmup-num 0 --max-tokens 32
```

小测试成功后，再跑完整的 8、16、32、64 并发：

```bash
bash examples/online_serving/hetero_pp/benchmark_gsm8k.sh \
  --python python3.9 \
  --url http://127.0.0.1:8000 \
  --model hetero-model
```

若解释器不在 PATH 中，把 `python3.9` 换成它的绝对路径。
实际执行的是 `指定Python -m evalscope.cli.cli perf ...`。
脚本会先检查该环境的 `perf --help`，缺少必要参数/依赖就退出并显示原错误，
不安装或升级任何包。Python 3.9 能否使用所需参数取决于你实际安装的
EvalScope 版本，不保证它能安装当前最新版。`--python` 与 `--evalscope` 互斥。

### 已内置官方测试集

项目现在附带 **1,319 条官方 GSM8K test 数据**：

```text
examples/online_serving/hetero_pp/data/gsm8k/test.jsonl
```

默认按脚本位置读取它，不需要额外下载或填写 `--dataset-path`，
也不要求执行时的当前目录固定。[出处、SHA256 和原始许可证](../../examples/online_serving/hetero_pp/data/gsm8k/README.md)随项目保存。

如果要换成自己的文件，使用 `--dataset-path`。要求为 **JSONL**，
每行一个对象，至少包含 `question`：

```json
{"question": "A shop has 12 apples and sells 5. How many remain?", "answer": "7"}
```

上面只是格式示例；正式比较请使用同一份真实 GSM8K 文件。
脚本不会在运行时下载数据或把示例题当成 GSM8K。
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
