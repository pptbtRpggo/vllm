# 异构 PP：两端监控与停止原因诊断

## 两边宿主机监控

服务启动后，在两台机器的项目目录分别执行同一命令：

```bash
bash examples/online_serving/hetero_pp/monitor_npu.sh --duration 60
```

不依赖宿主机 `npu-smi` 或 Python。默认检测唯一运行中的
`hetero-pp-head` / `hetero-pp-worker` 容器；使用自定义容器名时加
`--container 你的容器名`。需要当前用户本来就有 Docker 访问权限。

默认每两秒采集一次；省略 `--duration` 则持续到 Ctrl+C。
CSV 同时显示在终端，并保存到本机 `$HOME/vllm_hetero_pp_logs/`，
文件名以 `npu_monitor_` 开头，由宿主机当前用户创建，不需要 sudo。
可用 `--output /可写路径/新文件.csv` 指定位置，已有文件拒绝覆盖。
采集中报错仍保留已经得到的 CSV，不重启推理服务。

每行对应本机一张可见卡：时间、节点、逻辑设备号、型号、利用率、
设备显存使用量/总量及错误。逻辑号 0/1 不等于宿主机设备文件编号。
显存是设备总量减空闲量，不是监控 Python 进程自身的 allocator 使用量。
某个指标不受本机驱动/API 支持时留空并标 `partial`，不填成 0。
利用率是设备指标，不能区分同卡上其他进程，也不是每个 PP stage 的耗时。

监控进程在容器内复用启动预检保存的 Python/SDK 环境，单独建立 NPU 上下文，
会有少量开销；精确测性能时应停止监控。默认启动记录路径是
`/work/hetero_logs/start/local_preflight.json`。手动启动的环境可加
`--python /容器内/python`、`--cann-env /容器内/set_env.sh`，
不安装依赖、不改驱动、不重新选择或重置设备。

## 带 role 的请求

在 910B 宿主机执行（默认 API 端口 8000）：

```bash
curl --fail-with-body --max-time 180 \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "hetero-model",
    "messages": [{"role": "user", "content": "Reply with only the word OK."}],
    "temperature": 0,
    "max_tokens": 64,
    "ignore_eos": false,
    "return_token_ids": true
  }'
```

`role` 放在 `messages` 的每条消息中；不要把它作为 `/v1/completions`
的顶层参数。`system` 是否支持取决于模型的 chat template，先用单条
`user` 请求。如果返回没有 chat template 的错误，需确认具体模型及模板，
不能随便套一个模板认为等价。

对于本实验的 **CodeLlama-34B-Instruct**，项目已提供内置模板。
停止自己本次旧测试服务/容器后，在原来两边的启动命令末尾加：

```bash
--chat-template codellama-instruct
```

例如下面两种入口均支持该参数，其他 IP、模型路径和设备号沿用你的配置：

```bash
# 宿主机 Docker 入口（两台宿主机各运行一次）
bash examples/online_serving/hetero_pp/start_hetero_pp_docker.sh \
  --head 53 --peer 54 --model /本机模型目录 --devices 0,1 \
  --chat-template codellama-instruct

# 已有容器内入口（两台机器各运行一次；不要与上面的启动重复执行）
bash examples/online_serving/hetero_pp/start_hetero_pp.sh \
  --head 141.147.135.53 --peer 141.147.135.54 --model /workspace/model \
  --chat-template codellama-instruct
```

别名解析到项目内 `examples/online_serving/hetero_pp/codellama_instruct.jinja`，
模板正文取自 [CodeLlama-34b-Instruct-hf 官方 tokenizer 配置](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/blob/main/tokenizer_config.json)
的 `chat_template` 字段（2026-11-12 核对）。保留 `[INST]`、`<<SYS>>`、
BOS/EOS 和 user/assistant 交替规则；不是给基础版、Python 版或 70B 版通用的模板。
单轮和 system 加多轮消息的渲染字节已做 CPU 回归，但不代表实际异构输出已验证。

需要其他模板时传 `--chat-template /容器内/模板.jinja`，该文件只需 API 所在的
head 端可读。Docker 下可把文件放在项目目录，使用 `/work/vllm/相对路径`；
不要直接传未挂载的宿主机绝对路径。空文件/不存在的文件在 head 端预检拒绝。
省略参数仍用 tokenizer 自带模板，不自动猜测模型类型，也不修改只读模型目录。
实际模板路径可在 head 的 `launch.json` → `serve_command` 中确认。

`finish_reason: "length"` 表示达到生成长度或上下文上限。
`ignore_eos: false` 表示生成 EOS 时允许停止，不保证模型一定生成 EOS。
`stop_reason` 通常是停止字符串/token ID 或 null，并非上述 `length` 字段。

## 不粘贴长响应的诊断

在 910B 宿主机运行，换成实际容器名和已有 Python 路径（通常为 `python3`）：

```bash
docker exec hetero-pp-head python3 \
  /work/vllm/examples/online_serving/hetero_pp/check_eos.py
```

脚本只用 Python 标准库，不加载模型。默认读取 `/workspace/model/` 的
EOS 配置，请求 `http://127.0.0.1:8000` 上的 `hetero-model`。
自定义设置通过 `--model-dir`、`--model`、`--url` 覆盖。
四条请求依次执行，每条一次、默认超时 180 秒，无自动重试：

1. 仅允许 EOS，`ignore_eos=false`，上限 4 token。
2. 仅允许 EOS，`ignore_eos=true`，上限 4 token，作为对照。
3. 正常 prompt，最多 64 token。
4. 同内容的 `role=user` chat，最多 64 token。

前两条只改变本条诊断请求的 token 约束，不改服务参数。
第一条 `stop`、第二条 `length` 支持停止策略路径正常；不证明权重、
异构数值或自然输出正确。若服务不支持相应字段，原始 HTTP 错误会保留，
不会冒充通过。自然生成仍 `length` 时，要继续区分输入模板、模型 EOS 配置、
模型自身生成与异构计算问题，不能只加重复惩罚掩盖症状。

终端只输出每条的 `finish_reason`、`stop_reason`、生成 token 数、
返回 IDs 是否包含 EOS 和最多 80 字符的预览。返回中没有 IDs 时记为 null，
不推断 EOS 不存在。完整请求/响应和配置另存到
`/work/hetero_logs/eos_check_时间_标识/`，对应宿主机本轮日志目录下；
不覆盖旧文件。强制对照不符或任何请求失败时脚本退出码为 1，保留全部证据。
