# 干净 v0.23 分支：两节点四卡 CPU/Gloo PP 验证

日期：2026-11-06。结论：**本次迁移后的功能回归通过**。
手动启动步骤见 [中文操作文档](../usage/hetero_pp_cpu_gloo_zh.md)。

## 1. “干净 v0.23”具体指什么

分支为 `codex/hetero-pp-v023`，直接从官方 `v0.23.0` 创建，
基线提交 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` 已与官方远端 tag 核对。
没有带入原 v0.13 派生分支的历史提交或 EdgeShard 核心修改。

功能提交为 `aada5c3571ddda44da4476407f6f8a3402a48954`。
后续提交仅完善文档，不改变已测执行代码。此次迁移保留了原工作树和旧测试证据。

两端运行的是从官方 tag 导出的**完整源码树**加本次新增模块；
5440 个官方基线文件逐 SHA 核对一致，21 个运行/测试新增文件单独钉住 SHA。
实际导入位置在新的 `source/vllm` 中，没有使用上一轮的 import 覆盖层。
生成的 `_version.py` 复用已有匹配构建；没有重新编译或替换安装包、SDK、驱动。

运行环境是 vLLM 0.23.0、vLLM-Ascend 0.23.0.post1、
Torch 2.10.0+cpu、TorchNPU 2.10.0.post4、Ray 2.48.0、CANN 9.1。
这是实测的 Ascend 配套环境，不代表任意 Torch/插件版本组合均兼容。

## 2. 本次重新验证了什么

hw 与 hw2 是同一宿主机的两个容器节点，使用四张不同的物理 910B3。
双方容器内逻辑设备号都是 0/1。模型为同一份共享 CodeLlama-7B-Instruct，
32 层；12 个模型资产文件包括全部权重都重新核对 SHA，无需再复制一份。

| 检查 | 本次结果 |
| --- | --- |
| 相关 CPU/Gloo 回归 | 本地、hw、hw2 各 27 项通过，不把重复运行当成不同测试 |
| 真实 NPU tensor 通信 | 4 rank × 4 个形状检查，共 16 项内容检查通过 |
| Gloo 调用边界 | 104 次实际 tensor 载荷调用均为 CPU tensor + Gloo group |
| 跨节点 rank1→rank2 | 16 条 payload 的发送/接收 tag、字节数与 dtype 全部匹配 |
| 新单卡参照 | batch=1 与 batch=4，共 5 请求，各输出 16 token，正常退出 |
| PP 层数 12/8/7/5 | 同样 5 请求，全部输入/输出 token ID、文本与新单卡参照一致，退出码 0 |
| PP 层数 5/7/8/12 | 同样 5 请求，全部输入/输出 token ID、文本与新单卡参照一致，退出码 0 |
| 按文档启动的 API | `/v1/completions` HTTP 200，完整 16 个输出 ID、文本与参照一致，API 退出码 0 |

每种 PP 方案各 5 请求、80 个输出 token；两方案合计 10 请求、160 token。
HTTP 是另 1 请求、16 token；单卡参照另列，不混成多轮重复实验。
输入覆盖 6、11、61、251 token，并包含触发 chunked prefill 的请求批次。

固定 FP16、eager、TP1/PP4、同步调度、context=512、token budget=256、
max_num_seqs=4、chunked prefill 开、prefix cache 关、HBM 比例 0.5。
功能检查固定 64 个 KV blocks；这与此前研究矩阵的参数不同。

此次审查还复现并修复了同步 RPC 越过已接收后台模型 RPC 的问题：
先记录失败回归，再将同步和异步 public RPC 都提交到同一个队列。
同步等待位于准入锁之外；关闭时先排空已接收工作，再拆通信组。
本次两套模型和 API 均使用这一修复后的源码。

## 3. 为什么能判断 tensor 经过 CPU/Gloo

证据由三个部分组成：

1. 源码路径明确：worker 的中间 tensor 进入 `TensorDictChannel`，
   再进入 `CpuStagingTransport`；发送前完成独立 CPU buffer 的 D2H，
   `dist.isend/irecv` 使用显式 Gloo group，接收完成后 H2D 并等待复制结束。
2. 独立真实 NPU 探针在实际 Gloo 调用边界记录 tensor 的 device、backend、
   peer、tag、dtype 和字节数，全部观察到 CPU/Gloo；不是只打印环境变量。
3. 探针完整内容检查、非连续 view、源缓存覆盖、空 tensor、None、整数 token、
   反向 token 返回均通过；真模型的实际层归属、同一通道源码 SHA、输出 ID 和退出也通过。

探针只在自己的诊断进程中包裹调用以观察实参，不修改服务中的通信方法。
单看 WORLD 为 Gloo、Ray 显示四张卡，或 HTTP 能返回文字，都不足以单独证明载荷路径。

## 4. 退出、环境与验证边界

API 与模型进程均成功退出。Ray head/worker 在本次主动 SIGINT 停止后 CLI
返回码均为 1，CLI 日志为 `Aborted!`；主动停止记录保存了对应 PID 和 SIGINT。
这不是模型请求失败，也没有被改写成 0。
两端重新检查均无本次子进程/Ray 遗留，API/Ray/TCP store 端口关闭，四卡无运行进程。
原安装中核对的 `worker_base.py`、`parallel_state.py` 两个核心文件 SHA 前后相同。
不把这两项核对描述成 SDK 或所有生产文件的重新审计。

本地相关 pytest 为 27 passed，有一条未生成 `_version.py` 的源码环境告警；
远端使用匹配的生成版本文件。全仓 pytest 尝试在 60 秒上限内未完成，不能称全仓通过。
离线环境没有缓存 ruff，本轮未安装额外依赖，未声称 lint 全套通过。

这些结果证明本配置下的功能检查通过，**不是**吞吐/延迟基准、最佳分层结论、
跨物理主机网络验证或 310P 混合部署验收。没有加入软件等待或设置网络限速。
当前一次只执行一个 scheduler batch；不承诺四张卡持续同时满载。

旧 EdgeShard profiling 仍在 v0.13 派生实现中，未在本次迁移；
详见 [版本核查](edgeshard_profiling_v023_status.md)。
原生 runner 的 `profile_run`/显存预热日志不等于 EdgeShard 逐层采集与 DP 自动分层。

## 5. 证据位置

本地工作树的 `output/hetero_pp_v023_validation/` 保存：

- `branch_provenance.json`、官方 tag 核对、基线/新增文件 SHA 清单；
- `final_verification.json` 及 hw/hw2 两份一致的最终审计；
- 单卡参照、两方案完整 token 结果、HTTP 回执与资源退出检查；
- `v023_regression_evidence.zip`：268 个原始/源码/日志证据文件，全部成员 SHA 与 CRC 核验；
- `hetero-pp-v023.patch` 和最终交付包；
- EdgeShard 原始版本出处、迁移及 RPC 顺序问题的失败/通过回归记录。

远端独立目录为 `/home/m7zhang/userdata/hetero_pp_v023_20261106_d5a6b52b99`。
userdata 是共享存储，不把两个容器看到同一目录当作两个独立备份。
原始运行日志中的 UTC 时间与本文的北京时间属于同一次运行。
