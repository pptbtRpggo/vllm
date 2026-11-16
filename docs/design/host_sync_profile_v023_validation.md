# host-sync 保守计时验证（2026-11-17）

使用方法见 [310P 保守计时指南](../usage/edgeshard_v023_zh.md#20-310p-保守计时host-sync)。
本模式是绕开异常 Event 时间戳的兼容计时后端，不是对 CANN Event 根因的修复。

## 实现与验证结果

- 显式 `--compute-timing host-sync`；双端启动入口使用
  `--profile --profile-compute-timing host-sync`，由 head 传给全部四个 Ray worker。
- 在记录每个边界前完成设备同步，以 CPU 单调时钟计时；不创建或读取 profiler Event。
- 四个 stage 同口径；逐层和 stage 区间均非负且守恒；execute/sample 之间的空档不计入。
- 原 Event 模式保留为默认。新模式的 device 时间字段为 null，通用区间另存；
  mixed mode、未知模式和 worker 忽略所选模式均拒绝生成有效 DP 输入。
- CPU 回归：444 passed、2 skipped、74 subtests passed；pre-commit 全部适用项通过。
- 实机：hw/hw2 两容器、同物理宿主机、4×910B3；CodeLlama-34B，
  PP4/TP1、12/12/12/12、FP16/eager、ACK CPU/Gloo、64 输入/1 输出。
  1 轮预热 + 2 轮正式，每轮 4 请求，共 12 请求。此处没有 310P 实机可用。
- 实测 158.81 秒正常退出，4 rank 完成回执完整；每 rank 7 条 prefill 记录
  （2 warmup + 5 formal），共 28 条 stage 记录、336 个逐层区间，全部有限非负。
  记录数是 scheduler step 数，不是请求数。
- 12 请求的完整输入 ID 和输出 token ID 与此前同配置 Event 模式逐项一致。
- 同一实测数据的数值检查、计算/交接 CSV、通信速率表和离线 prefill latency DP 均通过。
  DP 产生 21/21/5/1 候选，预测 199.31 ms；这是本次四张 910B 的同步成本估计，
  不是 310P 推荐、原生 TTFT 或推荐方案重部署的实测结果。
- 自建 Ray/模型进程均退出，四卡无运行进程，实验端口已释放。

## 回归命令与环境限制

使用项目已有虚拟环境的 Python，运行 `pytest --noconftest -q`，范围为：

- `tests/distributed/hetero_pp/`
- `tests/distributed/test_pp*.py`，排除原生 CUDA `test_pp_cudagraph.py`
- `tests/entrypoints/test_pp_profile_cli.py`

首次扩大搜索误包含原生 CUDA 的
`test_pp_cudagraph[FLASH_ATTN-2-JackFram/llama-160m]`：本机 macOS fork 子进程在
系统代理查询处触发 Objective-C abort，未通过；该项需要 CUDA 和模型下载，
不属于此次异构 PP 的 CPU 回归。没有修改该测试或放宽判据；随后按上述适用范围
完整重跑并保存 444 项通过记录。2 个 skipped 为本机缺少 Ray 时跳过的
`ray._private.parameter` 实际端口参数校验；本次实机 Ray 集群另已成功启动并完成采集。

## 证据

本地 `output/host_sync_hw_validation_20261117_d46f29b1/`；远端相同目录名位于
`/home/m7zhang/userdata/`。独立任务目录复用已有环境/模型软链接，未安装依赖，
未改 SDK、驱动、生产源码和旧实验。

- `cpu_regression.log`、`pre_commit.log`：完整软件检查。
- `evidence/real_host_sync_audit.json`：原始计时、四 rank 及 2507 个冻结文件 SHA 核对。
- `evidence/event_vs_host_output_comparison.json`：12 请求输入/输出 ID 对照。
- `evidence/profile_34b/`：原始 trace、manifest、内存证据、CSV 和 DP。
- `hw_terminal_resources.json`、`hw2_terminal_resources.json`：资源清理状态。
- `local_evidence_verification.json`：取回归档的整包及全部成员 SHA 检查。

逐层同步会增加主机开销、改变提交重叠。310P 上机时先验证此模式的完整采集流程，
候选分层的实际 TTFT 在关闭 profiling 后另测。
