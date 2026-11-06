# EdgeShard profiling 与 vLLM 0.23 的核查结果

核查日期：2026-11-06。

**结论：此前实现的 EdgeShard 式 profiling 在 vLLM 0.13.0 派生代码中；
本次迁移到干净 v0.23.0 的是新的 CPU/Gloo PP 执行与通信功能，
没有同时迁移旧 profiling。两机 PP 服务通过，不代表自动 profiling 和分层规划也通过。**

## 1. 代码版本：手动切层与自动 profiling 是两件事

本分支从官方 `v0.23.0` 开始，基线提交为
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`。

此前 profiling 所用的冻结源码提交为
`4608607c1f76704cfa10712cf87d1b5561a3b867`。
Git 祖先检查确认该提交继承自 `v0.13.0`，不继承自 `v0.23.0`。
不能因为后来在 v0.23 上成功启动了新 PP 执行器，就把旧 profiling 的版本改称 v0.23。

这里的 profiling 指采集实际执行成本、再求分层方案的整条流程。旧实现涉及：

- `vllm/distributed/pp_profile.py`：生成采样负载、运行采集、调用规划器。
- `vllm/v1/worker/pp_ascend_worker.py` 和 `pp_stage_trace.py`、
  `pp_layer_trace.py`：采集实际 microbatch 的 stage、decoder layer 和首尾模块耗时。
- `vllm/distributed/pp_comm_trace.py`、`pp_link_profile.py`：
  配对相邻 stage 的通信窗口，而非用配置带宽直接生成成本。
- `vllm/distributed/pp_layer_cost.py`、`pp_memory.py`、`pp_partition.py`：
  汇总计算与通信成本、检查内存界，并用 EdgeShard 式动态规划选择连续分层。
- `vllm/entrypoints/cli/pp_profile.py`：旧 `vllm pp-profile` 命令入口。

这些路径指旧实现的位置，不表示本分支已经包含它们。
“EdgeShard 式”描述的是分层规划方法；不表示已逐项复刻原论文的完整系统。

官方 v0.23.0 的 `vllm/distributed/utils.py` 已支持
`VLLM_PP_LAYER_PARTITION`，例如手动指定 `12,8,7,5`。
它只负责按给定层数装载模型，不负责测量哪张卡快、计算最优层数。

## 2. 历史证据：能核实旧流程，不能证明 v0.23 已迁移

保存的历史运行环境记录明确写有 `vllm_version=0.13.0`、Torch 2.8 和 CANN 8.5，
源码指向 `mixed_prefill_fixed16_20261011/frozen_source`。
其启动程序钉住上述 `4608607c1f...` 提交。

同一历史程序的 `plan()` 读取真实逐层 trace 和内存观测，调用
`measured_layer_rank_costs(..., workload='prefill')`，再调用
`partition_layers(..., objective='latency')`，保存候选穷举对照及采样文件 SHA。
这说明旧流程有实际的采集与规划接入，不能简单说成只写了一个分层公式；
但这些证据的代码和环境仍属于旧版本。

直接将旧文件复制到 v0.23 至少存在以下已确认的接入差异：

- 旧 CLI 从 `vllm.entrypoints.utils` 导入 `VLLM_SUBCMD_PARSER_EPILOG`；
  官方 v0.23.0 已没有这个文件，入口使用
  `vllm.entrypoints.serve.utils.api_utils`。
- 旧 `vllm/config/vllm.py` 调用 `maybe_override_pp_worker`，将原生 Ascend worker
  替换成 `PPAscendWorker`；官方 v0.23.0 不含该定制接入。
- 官方 v0.23.0 CLI 未注册旧 `pp-profile` 子命令，也不含上述旧 profiling 模块。

本次核查读取了旧源码、Git 历史和既存运行元数据；
没有重新执行旧 profiling，也没有把历史记录当作新的 v0.23 实机验收。

历史绝对路径证据索引如下，保留在旧目录，未改写：

| 证据 | 路径与定位 |
| --- | --- |
| 环境版本 | `/Users/dazzysy/workspace/project/vllm/output/heterogeneous_data_inventory_20261029/runtime_reuse_review.txt` 第 52 行 |
| 冻结提交与源目录 | `/Users/dazzysy/workspace/project/vllm/output/heterogeneous_data_inventory_20261029/source_review/mixed_prefill_fixed16_20261011/run.py` 第 23–30 行 |
| 实际 trace → 成本 → DP 调用 | 同一历史 `run.py` 的 `plan()`，第 259–284 行 |
| 旧实现源码工作目录 | `/Users/dazzysy/.codex/worktrees/198a/vllm` |

## 3. 新 worker 的差异：通信已接入，profiling 尚未接入

本分支的 `vllm/distributed/hetero_pp/worker.py` 在每个 Ray actor 内选择本机的
原生 Ascend worker，再通过 `StagingWorkerMixin` 组织收取中间 tensor、执行本地层、
发送中间 tensor及等待完成。CPU/Gloo 通信由独立通道承担。

该 worker 没有继承旧 `PPAscendWorker`，也没有接入旧 stage tracer、逐层计时、
`set_pp_profile_warmup` 或 `get_pp_memory_observation` RPC。
旧采样客户端需要这些能力，不能在新 worker 上直接假定它们存在。

原生 Ascend runner 仍会做自己的显存估算、`profile_run` 和模型预热；
启动日志里出现 `profile` 不等于已经接入 EdgeShard 式逐层成本采集与 DP 自动分层。

两类验证要分别理解：

- 新 PP 功能验证回答：tensor 能否正确传递、每张卡是否装载指定层、
  生成 token 是否与参照一致、服务和资源能否正常退出。
- profiling 验证回答：采集是否覆盖真实执行路径、每一项计时是否有正确的完成依赖、
  warmup 是否排除、成本和内存界是否足以产生可执行的分层方案。

前者通过不能替代后者。已有的新 PP 验证结果不应标注为 EdgeShard profiling 通过。

## 4. 尚需迁移的范围：本任务只核查，不擅自扩展实现

**本次对 profiling 的工作范围是核查和说明；没有宣称或执行完整迁移。**
如后续要在 v0.23 上恢复自动分层流程，需要单独完成并验证：

1. 迁移采集、成本、内存及 DP 模块，适配 v0.23 CLI 和环境变量接入。
2. 给新 CPU/Gloo worker 接入实际 recv、compute、send 的完成边界、
   逐层计时、独立 warmup 标记及内存观测，保留原始 trace。
3. 验证 tensor 从 NPU 复制到 CPU、Gloo 传输及复制回 NPU的计时边界。
   不能沿用旧 HCCL 采样结果作为新通信路径的成本。
4. 对时钟来源作检查。旧公式
   `recv_end - max(send_start, recv_start)` 只允许已验证共享单调时钟的两端配对；
   不能直接相减两台物理主机的时间戳。本次双实例同宿主机也需保存并校验时钟域证据，
   不能仅凭网络可达就视为满足条件。
5. 完成“真实模型采集 → trace 完整性核对 → 成本/内存检查 → DP 输出 →
   按推荐分层启动 → 与参照逐 token 对照”的端到端验收。

CPU 单元测试、一次 import 成功或能手动启动不均匀 PP，均不能单独证明这些步骤完成。
即使全部功能验证通过，也仍需独立测量候选分层的实际服务性能，才能讨论规划收益。
