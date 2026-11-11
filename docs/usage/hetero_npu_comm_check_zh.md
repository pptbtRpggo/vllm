# 两机通用：910B / 310P 通信检查

在两台 Linux 机器上使用同一个入口，不需要 Ray、模型权重或服务端口。
当前检查固定为每台两张 NPU，共四个 rank；复用已有
`verify_cpu_gloo.py`、`TensorDictChannel` 和 `CpuStagingTransport`。
它验证 NPU→CPU→Gloo→CPU→NPU 的内容与完成依赖，不测带宽或模型推理。

## 使用 Docker：在两台宿主机执行

如果希望直接编辑 `docker run`，使用
`examples/online_serving/hetero_pp/check_cpu_gloo_docker_manual.sh`。
它保留可编辑的 `IMAGE`、`CONTAINER_NAME`、`MASTER_ADDR`、`MASTER_PORT`、
`DEVICE_ARGS` 和 `MOUNT_ARGS`，不强制删除同名容器。
两机填相同的 `MASTER_ADDR`，各自改本机镜像/设备/驱动挂载后执行：

```bash
export MASTER_ADDR=192.168.1.10
# 910B；310P 改成 v0.23.0.post1-310p-openeuler，或使用管理员镜像。
export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0.post1-openeuler
bash examples/online_serving/hetero_pp/check_cpu_gloo_docker_manual.sh
```

模板只传递测试入口实际使用的 `MASTER_ADDR`、`MASTER_PORT`、`NPU_DEVICES`。
使用 `-e "MASTER_ADDR=$MASTER_ADDR"` 等显式传值，避免在 `docker` 前加
`sudo` 后丢失调用者环境；容器内会打印实际收到的 head/port/devices。
网卡由原入口自动选择；默认不设置 `HCCL_*`，实际 tensor 数据走 CPU Gloo。
TorchNPU 本身仍需镜像内的 HCCL 动态库。当前无模型请求，不需要模型挂载或 8000 端口。
master 端口默认 29501，协调另用 29502，仍需允许 Gloo 动态 TCP 连接。
模板不能解决 Docker 服务不可达/权限错误；先确认宿主机 `docker version` 可读取 Server。

宿主机需要已有 Docker、NPU 驱动，以及各自相同版本的本分支源码。
镜像提供 CANN/TorchNPU，不能只复制 910B 的 Python 环境给 310P 使用。
选择与本机驱动、设备及 CPU 架构兼容的镜像；管理员提供的适配镜像优先。
以下官方 310P 镜像按 Atlas 300I DUO 的设备映射举例，其他板卡以管理员配置为准。

在 **910B 宿主机** 的项目根目录执行：

```bash
bash examples/online_serving/hetero_pp/check_cpu_gloo_docker.sh \
  --soc 910b --head 192.168.1.10
```

在 **310P 宿主机** 的项目根目录执行：

```bash
bash examples/online_serving/hetero_pp/check_cpu_gloo_docker.sh \
  --soc 310p --head 192.168.1.10
```

两端的 `192.168.1.10` 均替换为同一台宿主机的真实网卡 IPv4。
先启动的一端会等待另一端；需要在两个终端分别启动，不能等第一端结束。
如果当前已在管理员启动的配套容器内，直接使用下一节的非 Docker 入口，
不要在容器里再次启动 Docker。

自定义镜像或设备号：

```bash
bash examples/online_serving/hetero_pp/check_cpu_gloo_docker.sh \
  --image registry.example.com/admin/npu-image:tag \
  --head 192.168.1.10 --devices 0,1
```

默认映射 `/dev/davinci0`、`/dev/davinci1` 和三个管理设备
`davinci_manager`、`devmm_svm`、`hisi_hdc`，所选设备须存在且空闲。
映射宿主机驱动库及项目目录，但不覆盖镜像的整个 `/usr/local/Ascend`，
不安装依赖或修改驱动。支持从常见 bin/sbin 位置映射 `npu-smi`。
其他板卡的特殊设备映射需使用管理员的容器命令，再在容器内调用原入口。

脚本使用 host 网络；两机需允许 19090、19091 和 Gloo 动态 TCP 连接。
`--port`、`--timeout`、`--interface`、`--cann-env` 可传给容器内原测试入口；
`--cann-env` 指容器内路径。测试固定使用镜像 PATH 中的 Python，
避免误用项目内从其他机器复制的虚拟环境。
`--timeout` 不包括 Docker 镜像拉取时间；首次运行可能需要下载镜像。
加 `--dry-run` 只显示命令，不拉取镜像或启动测试。

容器退出后自动删除，日志仍保留在宿主机项目 `output/comm_check_*/`。
两端都返回 0 且生成各自的 `complete.json` 才算通过；Docker 能启动不代表通信通过。
本 Docker 封装只做了本地命令生成/错误传递检查，实际异构验证以本轮硬件回执为准。

## 运行

两端同步本分支源码；每端使用适配本机 SoC 的现有 TorchNPU/CANN 环境。
源码可以复制，910B 的虚拟环境、SDK 或二进制不能直接当作 310P 配套环境。
选用的两张卡应空闲且已映射进当前容器。
入口需要现有 Bash、Python 3.9+ 和 coreutils `timeout`。

选其中一台机器真实网卡的 IPv4，例如 `192.168.1.10`。
在两台机器各打开一个终端，进入各自项目根目录，运行相同命令：

```bash
bash examples/online_serving/hetero_pp/check_cpu_gloo.sh --head 192.168.1.10
```

第一台会等待另一台加入；不要等它退出才在第二台执行。
`--head` 在两端必须相同，必须属于其中一台机器，不能填 NAT 映射地址或回环地址。
默认 master 端口为 19090，协调端口为 19091；两端还需允许 Gloo 动态 TCP 连接。
使用容器时应配置可双向直达的网络。仅映射一个 HTTP 端口不能满足这个测试。

脚本会自动：

1. 判断本机是否拥有 head IP，选择 node-rank 和对应网卡。
2. 在当前虚拟环境、仓库 `.venv`、常用环境路径和 PATH 中寻找现有 Python。
3. 优先尝试当前 SDK 环境；必要时加载一个现有 CANN `set_env.sh`。
4. 识别两张可见卡的型号，检查本地 FP16、整数、复制和事件/stream 操作。
5. 对照两端 Torch 基础版本、字节序和通信源码 SHA，再启动四 rank 测试。
6. 确认本机 torchrun 正常退出、两份本轮回执正确，再交换两端结果。

不自动安装依赖或修改驱动。需要选择特殊环境或设备时，两端可分别加本机参数：

```bash
bash examples/online_serving/hetero_pp/check_cpu_gloo.sh \
  --head 192.168.1.10 \
  --python /path/to/npu-venv/bin/python \
  --cann-env /usr/local/Ascend/ascend-toolkit/set_env.sh \
  --devices 0,1 \
  --interface eth0
```

只填需要覆盖的参数。超过两张可见卡时，用 `--devices` 明确选择本次两张。
自定义 NNAL/ATB 安装路径需提前 `source` 本机对应的 `set_env.sh`；
自动补载 ATB 仅覆盖标准 `/usr/local/Ascend/nnal/atb` 路径。
设备号遵循当前容器/Ascend runtime 的可见映射，不要照搬宿主机另一容器的卡号。
脚本记录可见列表及逻辑 0/1 的实际型号；不会根据型号猜测空闲卡。
`--port 19190` 会同时改用协调端口 19191；`--timeout 600` 指定整轮上限秒数。

## 看结果

默认每台机器独立创建全新 `output/comm_check_时间_随机标识/`，路径会打印出来。
也可用 `--output /path/to/new-directory`；目录已存在时拒绝复用。

| 文件 | 含义 |
| --- | --- |
| `environment_attempts.log` | Python/SDK 探测及失败原因 |
| `local_preflight.json` | 本机实际设备、Python/Torch/TorchNPU、网卡和源码 SHA |
| `peer_preflight.json` | 对端实际环境和设备 |
| `launch.json`、`probe.log`、`process_exit.json` | 本轮命令、原始日志和进程退出码 |
| `cpu_gloo_rank*.json` | 本机两个 rank 的完整内容检查与真实 CPU/Gloo 调用记录 |
| `peer_completion.json` | 对端本轮完成回执 |
| `complete.json` | 本节点已核对两端四 rank 通过 |
| `failure.json` | 失败阶段和原因；原始日志保留 |

只有两端命令都返回 0、各自生成 `complete.json`，才认定本轮整体检查通过。
`scope=mixed_910B_310P` 表示实际识别到 910B 与 310P；
两端同型号则标为 `homogeneous_control`，不冒充异构测试。

本地算子/导入失败属于环境或设备预检失败，不能直接称为 Gloo 网络故障。
超时、中断或子进程失败不生成成功回执，只停止本轮自建进程组。
本入口沿用现有探针的四种形状、整数/空 tensor/None、非连续数据和反向 token 检查；
没有新增 format29 或显式非默认 stream 的实机覆盖。

该自动入口的 CPU 回归与双端协调模拟不能替代真实 310P 验证。
此前硬件结果为同一物理宿主机两个 910B 容器；新两机的异构结果以实际日志为准。
