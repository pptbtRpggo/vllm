# 两边运行同一脚本，启动可配置型号的四卡 PP

入口：`examples/online_serving/hetero_pp/start_hetero_pp.sh`。
通过 `--transport gloo|hccl` 选择通信路径，默认仍为原来的 CPU/Gloo。
不自动安装依赖或重建镜像。
普通服务模式不发测试请求；显式 `--profile` 模式执行内部采集负载。
通信检查通过之后，仍需用真实模型完成下面的验证。

## 选择 Gloo 或原生 HCCL

| 选项 | 执行入口 | stage 间张量通信 | 当前范围 |
| --- | --- | --- | --- |
| `--transport gloo`（默认） | 自定义 PP 执行器＋Ray | NPU→CPU→Gloo→CPU→NPU | 下述四种型号组合；支持现有 profiling |
| `--transport hccl` | 原生 `vllm serve`＋双机 `mp` | Ascend 原生 HCCL 设备通信组 | 仅 910B＋910B；不支持现有 `--profile` |

Gloo 在线服务默认使用 vLLM 的 PP 批队列。四个 stage 可同时容纳四个调度批次，
完成一批后继续提交就绪批次。批内请求和 token 数仍由 vLLM 调度器决定。
`--profile` 用于 EdgeShard 单批逐层采集。

两台 910B、每台两张卡时，两边宿主机运行同一命令，只改各自的模型目录：

```bash
bash examples/online_serving/hetero_pp/start_hetero_pp_docker.sh \
  --head 53 --peer 54 --transport hccl \
  --model /本机完整模型目录 --layers 12,12,12,12 \
  --master-port 29501 --chat-template codellama-instruct
```

这里默认 IP 前缀仍是 `141.147.135`，也可填写完整 IPv4。先在 head 执行，
等待时再在 peer 执行，两个终端都保持运行。已经进入容器时，把脚本名换成
`start_hetero_pp.sh`，并填写完整 IP 和容器内模型路径。
`hccl` 模式默认两端型号均为 910B；仍会初始化并核对本机两张卡。

- head 原生进程使用 `--nnodes 2 --node-rank 0`；peer 使用
  `--nnodes 2 --node-rank 1 --headless`。两边均为 TP1/PP4，自动设置同一份
  `VLLM_PP_LAYER_PARTITION`；`--layers` 的层数之和必须等于模型层数。
- 不启动或加入 Ray，不使用自定义 Gloo PP worker。两边自动设置各自的
  `HCCL_IF_IP`、`VLLM_HOST_IP` 和主机通信网卡；清除继承的自定义 PP/profiling
  激活变量。`--ray-port` 在此模式不使用，`--master-port` 两边必须相同。
- head API 在 `0.0.0.0:8000` 监听，供 peer 检查启动健康状态。peer 不提供
  API，会持续等待自己的原生 worker；健康检查不代表模型请求验证通过。
  普通 Gloo 模式仍只监听 head 的 `127.0.0.1`。
- **必须已有可用的跨机 HCCL 网络。** 主机 ping、Ray 或 Gloo 成功不代表
  NPU 网络已经连通。`--interface` 选择主机网卡，不配置 NPU 网口 IP、路由
  或物理网络。原生分布式通信可能还使用动态端口，不能只开放 master 端口。
- 当前 `hccl` 拒绝 310P/异构组合及 `--profile`，不会静默退回 Gloo。
  如果比较同一 910B 拓扑的 Gloo，在两边改为 `--transport gloo`，并显式写
  `--head-soc 910b --peer-soc 910b`。两种模式同时改变执行器和张量传输路径，
  结果差异不能全部归因于 HCCL 与 Gloo 的网络耗时。

用 `--dry-run` 可预览 Docker 命令；容器内 `--dry-run --node-ip 本机IP`
可查看 `launch.json` 格式的原生命令计划，不导入 Torch/Ray。
真实运行日志仍在本机本轮目录的 `start` 下：head 为 `api.log`，peer 为
`worker.log`，`launch.json` 保存实际命令、通信设置及需要清除的环境变量。
停止时两边各自 Ctrl+C，脚本只管理本轮子进程，不执行全局 Ray stop。

参考：[vLLM v0.23 原生双机 MP](https://docs.vllm.ai/en/v0.23.0/serving/parallelism_scaling/#running-vllm-with-multiprocessing)、
[Ascend 跨机通信准备](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/features/ray.html#verify-multi-node-communication-environment)。

## 两端型号与 stage 顺序

本节及后续原有 Ray/profiling 操作针对默认 `gloo` 模式。
`head` 表示 Ray/API 主端，负责 stage 0/1；`peer` 负责 stage 2/3。
两端都可以是 910B 或 310P，不再把主从角色与设备型号绑定。
两边命令使用相同的 `--head-soc`、`--peer-soc` 和 `--layers`：

| head / stage 0、1 | peer / stage 2、3 | 两边都加的参数 |
| --- | --- | --- |
| 910B | 310P | 默认；可显式写 `--head-soc 910b --peer-soc 310p` |
| 910B | 910B | `--head-soc 910b --peer-soc 910b` |
| 310P | 910B | `--head-soc 310p --peer-soc 910b` |
| 310P | 310P | `--head-soc 310p --peer-soc 310p` |

例如，两台 910B 服务器，各两张卡，运行 48 层 CodeLlama34B：

```bash
bash examples/online_serving/hetero_pp/start_hetero_pp_docker.sh \
  --head 53 --peer 54 --head-soc 910b --peer-soc 910b \
  --model /本机完整模型目录 --layers 12,12,12,12 \
  --chat-template codellama-instruct
```

前两个 stage 改为 310P、后两个为 910B 时，两边改用：

```bash
bash examples/online_serving/hetero_pp/start_hetero_pp_docker.sh \
  --head 53 --peer 54 --head-soc 310p --peer-soc 910b \
  --model /本机完整模型目录 --layers 6,6,18,18 \
  --chat-template codellama-instruct
```

以上 IP、路径和层数均可修改；层数示例不是最优分层或显存容量保证。
已在容器内时改用 `start_hetero_pp.sh`，保留同样的型号和层数参数，模型路径
填写容器内路径。先启动 head，再启动 peer，API 只在 head 上监听。
普通服务和 `--profile` 都支持同一套型号配置，通信始终使用 CPU/Gloo。

省略 `--layers` 时，原来的 910B→310P 命令仍使用 `18,18,6,6`；其余组合
按模型实际层数均分四个 stage，余数从前面的 stage 开始分配。例如 48 层为
`12,12,12,12`，32 层为 `8,8,8,8`。显式传入层数时始终按指定值部署。

型号配置决定默认镜像和每个 stage 的 worker，真实启动还会核对本机设备。
不是把 910B 假装成 310P；型号不符会在启动 Ray 前拒绝。旧 `mixed` 和
`910b_only` JSON 方案保留原检查，新顺序使用 `validation_mode: "explicit"`，
仍要求两机各两卡、四个 stage、合法型号以及正确的模型总层数。

## 宿主机自动选择镜像并启动容器

不想手工启动 Docker 时，两边在宿主机的项目根目录使用。
以下以默认 910B→310P 组合为例；其他组合加上前述型号参数：

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

宿主机入口按本机拥有哪个 IP 判断角色，再按对应的 SoC 参数选择镜像。
默认仍为：

| 角色 | 默认镜像 |
| --- | --- |
| 910B / head | `quay.io/ascend/vllm-ascend:v0.23.0-openeuler` |
| 310P / worker | `quay.io/ascend/vllm-ascend:v0.23.0-310p-openeuler` |

310P 默认使用普通 310P openEuler 镜像，不再使用 `mixed-pp` 自定义标签。
优先使用 Docker 本地已有镜像；不存在时 Docker 会按这个名称尝试拉取。
需要换镜像时，可以使用
`--image 镜像名`，或分别设置环境变量 `IMAGE_910B` / `IMAGE_310P`。

选镜像依据是用户填写的节点型号，不是宿主机硬件探测结果。实际型号、卡数
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
模型只读挂载，项目代码保持挂载；日志另外挂载到当前宿主机用户的目录。
不覆盖镜像自带 CANN。
Ray head/API 默认端口仍为 16379/8000，可通过 `--ray-port` / `--api-port` 修改；
其他 Ray/Gloo 端口仍动态分配。
两端 Ray 显式传入 `--min-worker-port=0 --max-worker-port=0`，让系统为 worker
分配可用端口，避免默认 worker 范围 `10002–19999` 与主端口 `16379` 或
自定义的 `16389` 重叠。`Ray component worker_ports ... used by other components`
是 Ray 参数配置冲突，不等同于旧进程占用；真正的监听占用可用 `ss -lntp` 查看。

宿主机入口的日志默认写到：

```text
$HOME/vllm_hetero_pp_logs/hetero_pp_head_时间_标识/start/
$HOME/vllm_hetero_pp_logs/hetero_pp_worker_时间_标识/start/
```

实际路径会打印出来。宿主机用户先创建私有外层目录（新建时为 `700`），
容器内本轮日志目录为 `755`、普通新日志文件为 `644`，因此即使由容器 root
写入，宿主机用户也能进入自己的目录读取日志。不批量修改旧 `output` 的权限。
容器内固定路径是 `/work/hetero_logs/start`，退出并删除容器后宿主机日志仍保留。
需要更换日志根目录时，在原命令后加 `--log-dir "$HOME/其他日志目录"`，
或设置 `HOST_LOG_DIR`。现有目录不会被批量 chmod，必须对当前账号可写、可进入。
下面手动容器内入口的日志默认值不变；已有受限日志需要目录拥有者处理。

peer 加入 Ray 后保持容器前台运行。测试结束时两边分别 Ctrl+C，或由容器拥有者
停止本次容器；容器退出自动删除，宿主机日志保留。已有同名容器时 Docker 会拒绝，
脚本不会先强制删除。没有使用 `--init`，不要求宿主机提供 `docker-init`。

在命令后加 `--dry-run` 可以先显示将执行的 Docker 命令，不启动容器或 NPU。
此入口先做 Docker 编排；下文是已经位于容器内部时使用的原入口。

### 自动 profiling：不需要共享目录

310P 上需要保守计时，可在两边的 `--profile` 命令中同时加
`--profile-compute-timing host-sync`。四个 stage 统一改为逐层 NPU 同步后的 CPU
单调时钟计时，profiler 不读取 Event 时间戳；head 自动回收、统计和运行 DP。
具体命令和结果字段见 [310P 保守计时](edgeshard_v023_zh.md#20-310p-保守计时host-sync)。

两台宿主机更新到同一版本，在各自项目根目录运行（修改 IP 和本机模型路径）：

```bash
bash examples/online_serving/hetero_pp/start_hetero_pp_docker.sh \
  --head 53 --peer 54 --model /本机模型目录 \
  --layers 12,12,12,12 --profile
```

上面是 48 层模型均分示例；32 层改为 `8,8,8,8`。先运行 910B，等待时再运行 310P。
结束自己上一轮的服务容器后再开始，不要并行占用同四张卡。

脚本自动连接 Ray、采集、回收 peer 记录、打印计算/交接耗时和速率表并生成 DP 推荐。
本次 profiling 每请求只输出首个 token，随后结束，不继续 decode；
打印表和 DP 都只使用 prefill 样本，以 `latency` 目标推荐层数。
两边先写本地文件，master 在采集结束后通过已有 Ray RPC 收回 peer 原始 JSONL 和
完成回执并校验 SHA，再统计和运行 DP；不需要 SSH/SCP、密钥或共享存储。
每 rank 原始文件导出上限 64 MiB，超过会报错并保留原件，不截断数据。

`--trace-root` 现在可省略，默认结果保存在各自宿主机本轮日志目录的 `profiles/` 下；
master 的目录汇总全部四个 rank。需要换目录时用 `--trace-root /本机已有可写目录`，
两机可以是不同目录。终端和 `start/profile.log` 会打印实际结果路径。
显存默认 0.8；采集默认 KV block size=128、固定 64 blocks，可显式覆盖。
全流程上限默认 3600 秒，可用 `--profile-timeout` 修改。

采集内部加载模型，不启动 HTTP API，也不自动发 EvalScope 请求。
完成后停止自己的 worker 容器，查看 `recommended_hetero_plan.json` 的四个层数，
以 `--layers L0,L1,L2,L3` 重新运行正常部署（去掉 `--profile`）。
保持相同执行/KV 配置，例如 `--block-size 128 --num-gpu-blocks-override 64`，
先核对输出正确性，再用 [EvalScope](hetero_pp_evalscope_zh.md) 与原分区对照测性能。
DP 推荐不代表已经实测更快。详情见 [profiling 指南](edgeshard_v023_zh.md)。

### 只进入 Bash 调试

在原宿主机命令后加 `--shell`，或执行（IP 改成实际值）：

```bash
bash examples/online_serving/hetero_pp/start_hetero_pp_docker.sh \
  --head 53 --peer 54 --devices 0,1 --shell
```

仍按本机 IP 选择 910B/310P 镜像和网卡，复用设备、驱动和项目目录挂载；
可以用 `--image`、`--devices`、`--name` 覆盖。不需要另一台机器同时运行。
此模式不需要 `--model`，不挂载模型或实验日志，不创建日志目录，
不执行 NPU 预检、Ray 或模型启动。`--dry-run --shell` 可预览 Docker 命令。

调试模式强制 `--entrypoint /bin/bash`，也跳过镜像入口中的自动环境初始化；
需要时在 Bash 内手动加载该镜像的 CANN 环境。两种角色均设置
`--security-opt seccomp=unconfined`，用于复现前述调试容器配置，
这会关闭该容器的默认系统调用过滤，并不表示已确认 NPU 错误由它导致。
普通服务模式的 entrypoint 和安全配置保持原样。

默认生成独立的 `hetero-pp-debug-...` 容器名，启动时打印实际名称。
调试容器退出后保留，不使用 `--rm`，不会删除或重启原服务容器。
将下面的 `容器名` 替换为打印的名称：

```bash
# 容器仍在运行时，另开一个 Bash
docker exec -it 容器名 /bin/bash
# exit 退出后，再次启动只进入 Bash
docker start -ai 容器名
```

普通服务容器的 `docker start -ai` 会重新执行服务启动命令，
可能因 `/work/hetero_logs/start` 已存在而拒绝启动；不要为此删除旧日志。

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

以上为默认 910B→310P 示例。脚本根据本机 IP 自动识别角色，具体型号由
`--head-soc` / `--peer-soc` 指定。先在 head 端执行，看到“等待两端各两张 NPU”后，
在 peer 端运行相同命令：

1. head 端启动 Ray head，然后等待 peer 加入。
2. peer 端加入 Ray 后返回 shell，Ray 进程继续运行；不要退出或删除容器。
3. head 端自动创建四个模型 worker、加载对应层并启动 API。
4. 显示“API 已就绪”之后，在 head 端另开一个 shell 发请求。

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
| `--head-soc 910b` | head 型号，支持 `910b` / `310p`，默认 `910b` |
| `--peer-soc 310p` | peer 型号；Gloo 默认 `310p`，HCCL 默认 `910b` |
| `--transport gloo` | Gloo＋Ray；`hccl` 选择原生 HCCL＋mp，仅支持双端 910B |
| `--python /path/to/python` | 指定已有 NPU 环境，未指定时复用通信脚本的环境探测 |
| `--cann-env /path/to/set_env.sh` | 显式加载本机 SDK 环境 |
| `--chat-template codellama-instruct` | 使用项目内 CodeLlama-34B-Instruct 对话模板；也支持容器内模板文件路径，省略则沿用 tokenizer |
| `--interface eth0` | 默认自动选择到对端的网卡；各端可不同 |
| `--ray-port 16379` | 两边相同；其他 Ray 和 Gloo 动态端口也需互通 |
| `--master-port 29501` | 仅 HCCL/mp，两边相同，不使用 Ray 端口 |
| `--api-port 8000` | Gloo 在 head 回环监听；HCCL 在 head 所有网卡监听 |
| `--startup-timeout 1200` | 每项启动、等待节点、模型健康检查的上限秒数 |
| `--max-model-len N` | 普通服务省略时由 vLLM 根据模型配置推导上下文上限 |
| `--max-num-batched-tokens N` | 普通服务省略时由 vLLM 选择 scheduler token budget |
| `--max-num-seqs N` | 普通服务省略时由 vLLM 选择同时处理的序列数上限 |
| `--gpu-memory-utilization N` | 普通服务省略时使用当前 vLLM 的显存比例默认值 |
| `--output /可写路径/新目录` | 日志目录；已存在时拒绝覆盖 |

普通服务的容量参数沿用 vLLM 根据模型与设备选择的默认值，显式传入的容量参数
会覆盖默认值。Gloo、HCCL 普通服务和使用已有 Ray 集群的 `serve.sh` 默认使用
FP16、eager、PP4/TP1、V1 runner 和同步调度，开启 chunked prefill，关闭前缀缓存。
HCCL 普通服务可显式传 `--async-scheduling` 开启异步调度，或传
`--no-enable-chunked-prefill` 关闭分块 prefill。

`--profile` 是独立的小规模采集模式，省略容量参数时仍使用上下文 512、token
budget 256、序列上限 4、显存比例 0.8，并保持 chunked prefill 开启、prefix
caching 关闭。若要按采集配置验证推荐分层的性能，启动普通服务时应显式传入相同
配置，不应把采用不同默认配置的新测量与原采集直接等同。

默认值位置：本项目的参数转发在 `examples/online_serving/hetero_pp/start_hetero_pp.py`
和 `serve.sh`；vLLM 自身的默认值来源为 `vllm/config/model.py`（上下文推导）、
`vllm/engine/arg_utils.py`（模型/设备/入口相关的 batch 默认选择）以及
`vllm/config/cache.py`（显存比例）。`vllm/config/scheduler.py` 中的字段缺省值
还会被引擎配置阶段调整，不能仅凭该文件的常量判断 API 服务最终配置。

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

两端保存 NPU 利用率/显存、带 role 的请求及重复输出的短摘要诊断，见
[监控与停止原因诊断](hetero_pp_monitor_zh.md)。

## 在 NPU 上验证批次重叠

普通 Gloo 服务的 `api.log` 应打印 `Heterogeneous PP max in-flight batches=4`。
该行确认批队列已启用。实际重叠可用同一启动脚本自动采集和检查。

在两台机器上都给原 `start_hetero_pp_docker.sh` 命令追加 `--verify-overlap`，
其余 IP、模型和分层参数照常填写。容器内的 `start_hetero_pp.sh` 也支持该开关。
先启动 head，再启动 peer。head 就绪后会预热、采集并检查输出 token IDs，最后
停止本轮 API。peer 的 Ray 容器仍按启动提示手动停止。开关只用于 Gloo 在线
验证，使用 128 token 的单批预算和四条 384 token 输入，与 EdgeShard 的
`--profile` 分开运行。

在 head 本轮 `start/overlap_verification.json` 查看结果，原始时间线保存在
`start/overlap_traces/`。

| 状态 | 条件 |
| --- | --- |
| `passed` | Stage 0 的较新批次和 Stage 1 的较早批次有实际 NPU 计算 kernel 重叠，回执记录两个批次及重叠微秒数 |
| `not_observed` | 本轮可配对的 kernel 中未观测到重叠 |
| `inconclusive` | 请求、trace、批次关联或时间轴检查证据不足，回执记录原因 |

自动比较限于同一台主机上的 Stage 0 和 Stage 1。四卡均参与模型推理。

### 手工采集时间线

按[手动启动指南](hetero_pp_cpu_gloo_zh.md)在两端启动 Ray。启动 Ray 前在两端
设 `export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`，让 worker 可以记录 `forward`
区间。两端容器都准备可写的 `/tmp/pp-npu-trace`，随后只在 head 启动下面的
在线服务。`MODEL` 指向已有模型，`PLAN` 指向手动指南中的四阶段 JSON。

```bash
bash examples/online_serving/hetero_pp/serve.sh \
  --served-model-name hetero-model \
  --max-num-batched-tokens 128 --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/pp-npu-trace","torch_profiler_with_stack":false,"ignore_frontend":true}'
```

先发一轮预热。正式采集时在 head 的另一个 shell 运行下面的命令。此处以
CodeLlama 为例，四条输入各 384 token，超过单批 128 token 的预算。

```bash
tokens=$(printf ',42%.0s' {1..383})
curl -fsS -X POST http://127.0.0.1:8000/start_profile
curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"hetero-model\",\"prompt\":[[41${tokens}],[42${tokens}],[43${tokens}],[44${tokens}]],\"temperature\":0,\"max_tokens\":8,\"ignore_eos\":true,\"return_token_ids\":true}" \
  -o pp_overlap_response.json
curl -fsS -X POST http://127.0.0.1:8000/stop_profile
```

从两边容器取 `/tmp/pp-npu-trace` 下的 worker trace，查看 `trace_view.json`
中的 `forward` 和关联 NPU kernel。先在同一主机的 Stage 0 和 Stage 1 上，
按采集窗口内的非空 forward 顺序定位相同批次。验收时看 **Stage 0 的后一批
计算 kernel 是否与 Stage 1 的前一批计算 kernel 在同一时间窗口重叠**。
两个分别从零开始的时间轴不能直接对齐，比较时保留采集原始时间戳。要验证
跨机的 Stage 1 和 Stage 2，还需核对两台机器的时钟偏差小于观察到的重叠。
请求成功和 NPU 利用率可用于功能检查。TTFT、TPOT 和吞吐另在关闭采集时测量。

## 停止与再次启动

在 910B 端按 Ctrl+C 停止本次 API，日志保留；Ray 服务也保留。
需要重启脚本时，先确认两边容器仅用于自己的本次测试，并由拥有者停止
本次 Ray 服务或重新创建这两个测试容器。不要在共享环境里随意执行全局清理。
本入口不会停止其他任务，也不自动重试模型失败。
