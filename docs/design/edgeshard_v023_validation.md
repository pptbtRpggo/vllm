# EdgeShard 式 profiling：v0.23 验收记录

日期：2026-11-06。分支：`ysy/hetero-pp-v023`。
官方基线：`v0.23.0`，提交
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`。

**结论：最终 v5 在本轮四张 910B3 的拓扑上跑通了真实采集、离线规划、
两种推荐方案的 trace-off 完整 token 对照、HTTP 和资源清理。
验收总回执为 `output/edgeshard_v023/final_verification.json`，`passed=true`。
这是功能验收，不是稳定性能、310P 或跨物理主机网络验收。**

手动操作见[中文指南](../usage/edgeshard_v023_zh.md)。
早先的 [profiling 版本核查](edgeshard_profiling_v023_status.md)
是后续实现开始前的历史快照，其中“未迁移”不描述当前已实现的功能。
旧 v0.13 测量数据没有被改称 v0.23 数据。

## 1. 实现和验证范围

这次接入的链路为：真实模型逐层采集 → 四 rank 完整回执 → 成本与内存核对 →
固定顺序 DP 分层候选 → 关闭采集重新加载 → 完整 token 对照 → API 与退出检查。

- vLLM 0.23，匹配 vLLM-Ascend 0.23.0.post1；完整实际 runtime 取自 worker 回执。
- CodeLlama-7B-Instruct，32 层，FP16、eager、同步调度，PP4、TP1。
- Qwen2 是代码准入允许的模型类型，本轮没有 Qwen2 实机验收。
- hw 和 hw2 各两张不同物理 910B3。两端是**同一物理宿主机上的两个容器**。
- Ray 安排 worker 和设备；相邻 stage 的 tensor 经 D2H、CPU/Gloo、H2D 完成后消费。
- 固定原四设备顺序和首末角色，只调整连续层数。不做设备子集选择、重排、PD 或量化。
- 本轮模型/context/budget/KV 配置属于功能验证，不是旧研究实验的 128 请求条件。

源码分支干净地继承官方 v0.23，不依赖 v0.13 的旧 worker 自动替换钩子。
采集入口显式选 `HeteroPPExecutor` 类，并把 profiling 配置从 head 传到 actors；
默认多进程 EngineCore 不要求 driver 直接持有 executor 对象。

本轮只验收本文和操作指南列出的基础参数集合。
完整 `EngineArgs` JSON 是来源证据，不表示生成器会通用重放所有参数。
任意自定义 `additional_config`、LoRA 或其他未覆盖配置不能据此获得兼容性保证。

这不是 310P、跨物理机 NIC 或稳定性能验收。
当前单执行批次的调度方式，也不能由四张卡存在推断四张卡一直同时计算。

## 2. 测量与推导的边界

逐层计时覆盖实际 decoder 与首末端模块。`execute_model` 和 `sample_tokens`
是两段独立 RPC，分别计时后归入同一 step；两段之间的 RPC 空档不算设备计算。
LM head 与 sampling 分开理解，未执行的 endpoint 不伪造为已测零成本。

CPU/Gloo 通信使用真实 host 完成窗口，包含复制、分配、元数据、排程与同步。
它不是纯网卡发送时间，也不是从配置带宽换算出的成本。
发送前的 profiling readiness 检查用于防止计算尾部重复计入通信；
该等待已由计算计时覆盖，不另加成本。它可能改变原有重叠，因此 profile 数值
不能直接充当 trace-off 服务的 E2E 延迟。

默认 `latency` 最小化代表性 stage 计算之和与相邻通信成本。
`throughput` 给出阻塞 stage 瓶颈模型下的候选；它不代表当前单执行批次 backend
实测吞吐最优。两者都不是完整的请求到达、排队、HTTP、尾延迟或 E2E 调度仿真。

内存 fixed-block envelope 保留原始 observation，并明确以下假设：

- 每设备只观测了本机实际加载的层；用其最大 decoder storage 外推其他同结构层。
- 每层 KV 取 native page size × fixed blocks，且与实际 KV pool storage owner 字节守恒。
- 非 KV 预留从 peak reserved 中扣除本机模型与真实 KV pool，再计当前非 Torch 占用；
  另外显式留 safety margin，不重复叠加完整 KV。
- 这些是当前负载、同结构层、固定设备与首末角色下的候选约束，
  不是任意分区、任意 workload 都不会 OOM 的保证。

## 3. v4 中间成功证据：保留原件，不替代 v5

v4 的真实采集任务与随后离线规划任务均正常退出。
本地 `output/edgeshard_v023/profile_v4_summary.json` 记录四个完整 rank 回执：

| PP rank | raw step 行数 | warmup 行数 | 正式行数 | pending | 完整回执 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 97 | 32 | 65 | 0 | true |
| 1 | 97 | 32 | 65 | 0 | true |
| 2 | 97 | 32 | 65 | 0 | true |
| 3 | 97 | 32 | 65 | 0 | true |

这些是**每个 stage 的 scheduler step 记录**，不是 388 个唯一请求，
也不是 97 个用户请求。请求数量和输出 token 内容由独立 public-generate 记录器保留。

v4 采集配置为 I64/O16、每次 generate 四条 prompt、一轮 warmup、两轮正式采集；
context 512、token budget 256、max seqs 4、block 128、固定 64 KV blocks、HBM 比例 0.5、seed 7。
原始 profiling topology 使用四个相邻 stage；完整实际配置由该轮 manifest 和原参数文件绑定。

v4 的两个候选如下，**仅是该中间版本测量所得的规划结果**：

| 目标 | v4 推荐层数 | 可以支持的说法 |
| --- | --- | --- |
| latency | `1 / 29 / 1 / 1` | 该轮成本与内存模型产生了固定顺序连续分区候选 |
| throughput | `10 / 7 / 7 / 8` | 该轮阻塞 stage 瓶颈模型产生了另一个候选 |

这两组数字不证明速度更快、不证明稳定最优，也不证明同为 910B 的某个 rank
拥有恒定更高的算力。实际性能需要独立 trace-off 对照；本文没有进行此类推断。

v4 成功使用的源码另存于 `output/edgeshard_v023/frozen_runtime_v4/`，
其清单为 `source_manifest_v4.json`；远端也保留同轮冻结副本。
之后 mypy 类型检查发现的问题及修正单列，不覆盖 v4 raw 或把新类型版本冒称已实测。

## 4. 最终 v5 验收结果

验收针对 v5 冻结源：39 个本次来源文件与隔离运行树的 5,440 个官方基线文件已核对。
这不等同于重新审计整个已安装生产环境；原安装额外核对的两份核心文件在测试前后 SHA 不变。
四 rank 回执均记录 vLLM 0.23.0、Torch 2.10.0+cpu、TorchNPU 2.10.0.post4、Ray 2.48.0。

| 验收项 | 最终结果 | 主要证据（相对于本地证据根） |
| --- | --- | --- |
| 来源与运行身份 | base、39 来源文件、四 rank source/model/config/topology 绑定通过 | `final_verification.json`、`final_evidence/source_manifest_v5.json` |
| 相关 CPU/Gloo 回归 | hw、hw2 各 217 项通过 | `final_evidence/hw_v5_cpu.log`、`final_evidence/hw2_v5_cpu.log` |
| 已改文件检查 | 完整 pre-commit hooks、mypy 3.10 和 3.12 通过 | `precommit_final_all_pass.log`、`precommit_final_mypy312.log` |
| 默认多进程 CLI 采集 | exit 0，完整 profile manifest | `final_evidence/profile_v5_exit.json`、`final_evidence/profile_v5/profile_manifest.json` |
| 离线规划 | exit 0，保存两个独立目标的候选 | `final_evidence/offline_plan_v5_exit.json`、`final_evidence/offline_planning_results_v5.json` |
| 独立只读复算 | 22 runtime 源、392 行 layer timing、294 个配对 hop、KV 和两个 DP 重算一致 | `audit_v5.json`，未导入 Torch/NPU runtime |
| 两种推荐方案实机 | 均正常退出，真实层范围、设备和 Gloo 身份核对 | `final_evidence/recommended_*_validation_results.json`、`final_evidence/recommended_*_exit.json` |
| 完整 token 对照 | 每种方案 17 请求 / 272 输出 token，所有 IDs/text 相等 | `final_evidence/recommended_*_validation_raw.json` |
| 内存复验 | 两种方案的实际 KV pool 与峰值/预算检查均通过 | `final_verification.json` 的 `actual_memory_checks` 及完整 observation |
| 最后 HTTP 检查 | HTTP 200，16 输出 token，完整 IDs/text 与参照相等，API exit 0 | `final_evidence/http_complete.json`、`final_evidence/http_request.json`、`final_evidence/http_response.json`、`final_evidence/api_recommended_exit.json` |
| 最终资源 | 四卡 idle，无本次活进程和监听；原安装两份核心 SHA 不变 | `final_evidence/hw_final_resources.json`、`final_evidence/hw2_final_resources.json` |

没有运行完整仓库 pytest 或全部平台 CI；不能把这些相关回归和 pre-commit 结果称为“全 CI 通过”。

v5 真实采集有 3 次 `generate()` 调用，每次四条，共 12 条请求、192 输出 token，
包含 4 条 warmup 和 8 条正式请求；
四 rank 各 98 行：32 warmup、66 正式，pending 均为 0。
这些仍是 stage step 行数，不是唯一请求计数。

| 最终候选 | 层数 | 实际层区间 | trace-off 对照 |
| --- | --- | --- | --- |
| latency | `1 / 1 / 1 / 29` | `[0,1)` / `[1,2)` / `[2,3)` / `[3,32)` | 17 请求，272 输出 token，全部相等，exit 0 |
| stage bottleneck（CLI 名称 throughput） | `9 / 8 / 7 / 8` | `[0,9)` / `[9,17)` / `[17,24)` / `[24,32)` | 17 请求，272 输出 token，全部相等，exit 0 |

每种方案的 17 请求包含 5 个此前独立 heldout 参照请求和 12 个本轮 profile 输入。
两方案复用同一组输入；这是两次各 17 请求的执行，不是 34 个不同输入。
HTTP 是另一次 16-token 服务检查，不计入上述 17 请求。

实际内存复验中，每 rank 都核对 KV pool 与 native page geometry。
latency 方案最大的“allocator peak 加当次非 Torch 占用”为 16,583,655,424 字节；
stage-bottleneck 方案为 5,792,681,984 字节；各卡本轮预算为 32,726,056,960 字节。
这些是本轮负载下的观测，不改变前述代表性和未来 OOM 的限制。

v5 与 v4 推荐不同，说明本轮实测代表成本和状态存在波动；
不能从两轮候选推导稳定最优或把差异归因于唯一硬件原因。
v4 和 v5 的来源、raw 和候选分开保留，最终验收不使用 v4 布局冒充 v5 布局。

### 退出记录与日志边界

profile、两个推荐方案和 API 的退出回执均为 0。
本次自有 Ray CLI 在受控 SIGINT 后退出码均为 1，未伪改成 0。
原生多进程 manager 的日志还保留过 `force killing remaining processes count=1`。
因此不声称“全程没有 warning”或“所有子服务都自然 exit 0”；
最终清理由进程、端口、四卡 idle 及 source 保护的实际复查共同确认。

## 5. 前三次失败和受控修正

前三次记录没有删除或重标成功：

| 版本 | 失败点 | 受控修正与保留边界 |
| --- | --- | --- |
| v1 | controller 误把 v0.23 默认 `disable_cascade_attn=True` 当作不支持配置；模型未启动 | 增加真实默认参数准入回归，保存平台处理后的 effective 值，并显式重放 True/False |
| v2 | 独立 public-generate 记录脚本缺少 main guard，spawn 重入 CLI，触发 bootstrap 错误 | 给独立记录器加入口保护；不改 native EngineCore 的 spawn 行为 |
| v3 | profiling metadata RPC 返回 `TorchVersion` 等非原生 JSON 标量，严格序列化拒绝 | 归一化 metadata 类型并验证实际 RPC；未开启不安全 pickle fallback，未把该轮写成完整成功 |

v4 后的类型检查错误也保留原日志。类型修正没有放宽内容、完成依赖或成本/内存检查，
随后用最终 v5 源码重新完成整条实机验收。

## 6. 为什么独立保留生成 token

生产 `pp-profile` collector 负责 trace、配置和内存证据，
**本身不保存 `LLM.generate()` 返回的完整输入/输出 token 序列**。
因此本轮使用独立 public-generate 记录器，完整委托原 `LLM.generate`，
只在原调用返回后保存公开结果；不替换 model、worker 或 forward 计算。

本轮验证重放了这些实际 input IDs，并在推荐分区、tracing 关闭的条件下，
与 profile-on 返回的完整 output IDs/text 逐条核对；同时复验此前的独立 heldout
单请求和 batch=4 参照。warmup 可以从成本统计中排除，但其正确性数据不能凭空补造。

HTTP 是这条功能验收的**最后服务检查阶段**，在推荐方案的 trace-off 模型验证之后进行。
一次 HTTP 成功不能证明 profiler 已正确采集、DP 成本有效或吞吐改善。

## 7. 证据索引与保留规则

本地证据根：`output/edgeshard_v023/`。
远端本轮独立根：
`/home/m7zhang/userdata/edgeshard_v023_20261106_e359ce53d7`。
共享 userdata 在两端可见，不把同一份共享文件算成两个独立备份。

| 中间或基础证据 | 位置 |
| --- | --- |
| 本轮 base、节点和独立 root | `session.json` |
| v4 启动与四 rank 汇总 | `profile_v4_launch.json`、`profile_v4_summary.json` |
| v4 原始采集 | `final_evidence/profile_v4/`，保留原结构 |
| v4 离线规划 | `final_evidence/analysis_latency/`、`final_evidence/analysis_throughput/`、`final_evidence/offline_planning_results.json` |
| v4 原源码 | `frozen_runtime_v4/`、`source_manifest_v4.json` |
| 独立公开结果记录器 | `profile_record_driver.py` 及相应冻结版本 |
| 失败与受控诊断 | `revisions/`；原日志、源快照和诊断不删除、不重标成功 |
| v5 原始采集 | `final_evidence/profile_v5/` 及 `final_evidence/profile_v5_generate_outputs.json` |
| v5 两个分析目标 | `final_evidence/analysis_v5_latency/`、`final_evidence/analysis_v5_throughput/` |
| v5 总核验 | `final_verification.json`、`remote_verification_receipt.json` |

原始证据 ZIP 为 `edgeshard_v023_raw_evidence.zip`：716 个证据文件，
2,891,586 字节，整 ZIP SHA256 为
`c9c228d84ca4e6979d52e54db6a7b6c84b5d5a3e038a5fd20f59138eb6685189`。
本地和 HW 已核对 ZIP CRC、成员 SHA 和整包 SHA；回执为
`raw_evidence_zip_receipt.json` 与 `local_raw_zip_verified.json`。
这是原始证据包；含最终文档和提交信息的总交付包另由交付回执记录，二者不混称。

实测来源以 v5 source manifest SHA256
`c33018cd64ea91b808347d0e3656dc1637be0036d562e1465e65dae6fb0c6a86`
和 profile manifest SHA256
`47700d7a6ada7263809dabdda11b6a6488eb5237fe03f20a5cc335e9a12ba822`
绑定，最终代码提交号由交付阶段记录。

总验收 JSON SHA256 为
`9a9f0763a7075251d7b6b0f079ec37caca43c22ad92e83441692deda55f11dde`；
独立 `audit_v5.json` SHA256 为
`0e71853bd79d3c2f056b52d5f71ab3408d0c129f6cae968d617284ea56895322`。

不改驱动、SDK 或网络去迎合结果；不放宽内容正确性、完成依赖或身份/SHA 检查。
本轮证据支持“v0.23 上这条 CPU/Gloo profiling→DP→推荐部署链路在
本次四张 910B 的拓扑跑通”；不能扩大为 Qwen2/310P ready、跨物理机已验证或性能优胜。
