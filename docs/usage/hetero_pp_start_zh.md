# 两边运行同一脚本，启动 910B/310P 四卡 PP

入口：`examples/online_serving/hetero_pp/start_hetero_pp.sh`。
这是现有异构 PP 后端的启动脚本，不自动安装依赖、重建镜像或发测试请求。
通信检查通过之后，仍需用真实模型完成下面的验证。

## 宿主机自动选择镜像并启动容器

不想手工启动 Docker 时，两边在宿主机的项目根目录使用：

```bash
bash examples/online_serving/hetero_pp/start_hetero_pp_docker.sh \
  --head 53 --peer 54 \
  --model /本机的完整模型目录
```

将 `53` 改为 910B 机器的 IP 末段，`54` 改为 310P 机器的 IP 末段。
默认前缀为 `141.147.135`；以上示例展开为 `141.147.135.53` 和
`141.147.135.54`。也可以直接传完整 IPv4。先运行 910B 端，等待时再运行 310P 端。
两端宿主机模型目录可以不同，容器内统一挂到 `/workspace/model`；内容必须一致。

也可以在脚本开头或环境中填写 `HEAD_LAST`、`PEER_LAST`、`HOST_MODEL_DIR`，
之后直接运行脚本。空的 `RAY_PORT` / `SERVICE_PORT` 使用默认值 16379 / 8000。

宿主机入口按本机拥有哪个 IP 判断角色，再选择镜像：

| 角色 | 默认镜像 |
| --- | --- |
| 910B / head | `quay.io/ascend/vllm-ascend:v0.23.0-openeuler` |
| 310P / worker | `quay.io/ascend/vllm-ascend:v0.23.0-310p-openeuler` |

310P 默认使用普通 310P openEuler 镜像，不再使用 `mixed-pp` 自定义标签。
优先使用 Docker 本地已有镜像；不存在时 Docker 会按这个名称尝试拉取。
需要换镜像时，可以使用
`--image 镜像名`，或分别设置环境变量 `IMAGE_910B` / `IMAGE_310P`。

选镜像依据是用户填写的 IP 角色，不是宿主机硬件探测结果。实际型号、卡数
和两张卡的初始化由容器内 TorchNPU 检查，角色与实际型号不符时拒绝启动。
全流程不要求宿主机有 `npu-smi`、TorchNPU 或 CANN 用户态环境。

网卡通过 `ip -4 route get 对端IP from 本机IP` 自动选择并检查本机地址归属，
设置 `GLOO_SOCKET_IFNAME` 与 `VLLM_HOST_IP`。宿主机需要已有 `ip` 和 Docker。
可用 `--interface 网卡名` 明确指定；指定接口必须拥有填写的本机 IP。

默认映射宿主机卡设备文件 `davinci0` / `davinci1`。需要其他卡时使用
`--devices 0,2`；只改变 Docker 设备映射，不再设置容器的
`ASCEND_RT_VISIBLE_DEVICES`，避免宿主机编号和容器逻辑编号重复筛选。
这是选卡配置，不是空闲检测；必须确认这些卡可以供本次使用。

310P 容器添加 `--security-opt seccomp=unconfined`；910B 容器不添加此参数。
两边均使用 host 网络、1GiB 共享内存并保留镜像原始 entrypoint。
模型只读挂载，项目可写挂载以保存日志；不覆盖镜像自带 CANN。
Ray head/API 默认端口仍为 16379/8000，可通过 `--ray-port` / `--api-port` 修改；
其他 Ray/Gloo 端口仍动态分配。

310P 加入 Ray 后保持容器前台运行。测试结束时两边分别 Ctrl+C，或由容器拥有者
停止本次容器；容器退出自动删除，项目日志保留。已有同名容器时 Docker 会拒绝，
脚本不会先强制删除。没有使用 `--init`，不要求宿主机提供 `docker-init`。

在命令后加 `--dry-run` 可以先显示将执行的 Docker 命令，不启动容器或 NPU。
此入口先做 Docker 编排；下文是已经位于容器内部时使用的原入口。

## 准备

- 在两边已经能通过 NPU/CPU/Gloo 检查的容器里运行，保留成功的 Docker
  参数；310P 端添加 `--security-opt seccomp=unconfined`，910B 端不添加。
  不要为了这个脚本覆盖镜像原有 entrypoint。
- 两端更新到同一版项目；shell 入口还需要同目录下的
  `start_hetero_pp.py` 和 `check_cpu_gloo.py`，不能只复制一个 shell 文件。
- 两边各有两张真实可用的卡。脚本沿用当前容器的可见设备，默认不重新设置
  `ASCEND_RT_VISIBLE_DEVICES`。需要明确选择时才使用 `--devices`，这里指
  当前 Ascend runtime 的逻辑编号，不是直接照搬宿主机设备文件的数字。
- 两边相同路径放置相同模型配置、tokenizer 和完整权重。可挂载同一份共享文件，
  或分别放置一致的副本。脚本只检查配置，不逐文件校验大模型权重。
- 沿用每端自己的 CANN、TorchNPU 和 Ascend 插件环境。两端 Python、Ray
  版本需要兼容；当前执行器要求 vLLM、Torch、TorchNPU、Ray 的版本字符串一致。
- 当前只支持 `LlamaForCausalLM`、`Qwen2ForCausalLM`，FP16、eager、TP1/PP4。
  每端两卡必须供本次使用；脚本不会自动寻找空闲卡或停止其他任务。

## 两边使用相同命令

下面 IP 和模型路径都是示例，替换后两边使用相同值。在项目根目录执行：

```bash
bash examples/online_serving/hetero_pp/start_hetero_pp.sh \
  --head 192.168.1.10 \
  --peer 192.168.1.20 \
  --model /workspace/models/CodeLlama-34b-Instruct-hf \
  --layers 18,18,6,6
```

`--head` 填 910B 机器，`--peer` 填 310P 机器。脚本根据本机 IP 自动识别角色。
先在 910B 端执行，看到“等待两端各两张 NPU”后，在 310P 端运行相同命令：

1. 910B 端启动 Ray head，然后等待 310P 加入。
2. 310P 端加入 Ray 后返回 shell，Ray 进程继续运行；不要退出或删除容器。
3. 910B 端自动创建四个模型 worker、加载对应层并启动 API。
4. 显示“API 已就绪”之后，在 910B 端另开一个 shell 发请求。

`18,18,6,6` 对应 48 层模型，仅为初始分层，不是实测最优方案，也不保证
任意 310P 板卡显存都能容纳。层数之和须等于模型 `num_hidden_layers`。
embedding 和输出 head 仍在首末 stage，也会占用显存和计算时间。

如果要先看将执行什么，在上述命令后加 `--dry-run`。
它不导入 Torch/Ray、不初始化 NPU、不创建日志或启动服务。
需要在别的机器预览角色时，额外传 `--node-ip 192.168.1.10` 或
`--node-ip 192.168.1.20`；真实启动时仍会核对 IP 属于本机网卡。

可选参数：

| 参数 | 默认 / 用途 |
| --- | --- |
| `--python /path/to/python` | 指定已有 NPU 环境，未指定时复用通信脚本的环境探测 |
| `--cann-env /path/to/set_env.sh` | 显式加载本机 SDK 环境 |
| `--interface eth0` | 默认自动选择到对端的网卡；各端可不同 |
| `--ray-port 16379` | 两边相同；其他 Ray 和 Gloo 动态端口也需互通 |
| `--api-port 8000` | API 仅监听 910B 容器/宿主机的 `127.0.0.1` |
| `--startup-timeout 1200` | 每项启动、等待节点、模型健康检查的上限秒数 |
| `--max-model-len 512` | 第一轮功能验证的上下文上限 |
| `--max-num-batched-tokens 256` | 每轮 scheduler token budget |
| `--max-num-seqs 4` | 同时调度的请求数上限 |
| `--gpu-memory-utilization 0.8` | 各卡显存使用比例配置 |
| `--output /可写路径/新目录` | 日志目录；已存在时拒绝覆盖 |

第一次使用不需要手工启动 Ray，也不要再传 `--nnodes` 或
`--distributed-executor-backend`。已有 Ray 实例或端口被占用时会退出，
不会静默复用或执行全局 `ray stop`。

## 发请求和看结果

在 910B 容器另一个 shell 中执行：

```bash
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "hetero-model",
    "prompt": "The capital of France is",
    "temperature": 0,
    "max_tokens": 16,
    "ignore_eos": true,
    "return_token_ids": true
  }'
```

本机日志目录会打印出来，默认是 `output/hetero_pp_start_时间_标识/`。

| 文件 | 含义 |
| --- | --- |
| `environment_attempts.log`、`local_preflight.json` | 本机 NPU 与环境检查 |
| `plan.json`、`launch.json` | 分层方案、角色、实际命令和少量必要环境变量 |
| `ray_start.log`、`ray_started.json` | Ray 本机启动记录，不是模型验证通过 |
| `cluster_wait.log` | 910B 等待两端 NPU 资源的记录 |
| `api.log` | 910B 完整 API 日志，包括四个 stage 的启动报告 |
| `api_ready.json` | `/health` 返回 200，仍不是推理内容验证通过 |
| `server_exit.json` | API 正常返回后的退出码 |
| `failure.json` | 启动异常、超时或用户中断的原始原因 |

启动报告应显示 rank0/1 为 910B、rank2/3 为 310P，`world_backend=gloo`，
层区间与分层方案一致。先核对单请求完整的 16 个输出 token IDs，再测试
并发 4、长输入和另一种分层；参照使用相同权重和输入的 910B 结果。
跨 SoC 的数值分歧需要分析，不能仅凭文字看起来正常就认定正确。
本启动脚本不会生成模型正确性 `complete.json` 或自动声明通过。

当前执行器一次只执行一个 scheduler batch，不保证四个 stage 持续重叠工作。
先验收完整推理功能，再评估 TTFT、TPOT、吞吐及 CPU staging 开销。

## 停止与再次启动

在 910B 端按 Ctrl+C 停止本次 API，日志保留；Ray 服务也保留。
需要重启脚本时，先确认两边容器仅用于自己的本次测试，并由拥有者停止
本次 Ray 服务或重新创建这两个测试容器。不要在共享环境里随意执行全局清理。
本入口不会停止其他任务，也不自动重试模型失败。
