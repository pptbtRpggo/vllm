# 310P 与 910B：Event 负计时的底层路径对照

调研日期：2026-11-16。结论来自当前项目源码、已完成的 910B 实机证据、
本线程用户提供的 310P 异常字段，以及 CANN Runtime v9.1.0 开源源码。
本轮没有启动新的 NPU 实验，也没有修改计时实现或 SDK。

## 核心结论

310P 和 910B 使用相同的 Python Event 接口，但设备侧调度和时间戳回填路径不同。
CANN 的耗时计算允许时间戳逆序时返回负数；时间戳为无效哨兵时则返回 timestamp
invalid。这里已经找到两类现象的代码级产生机制，尚未找到现场 310P 时间戳异常的
具体触发条件。不能由此直接判定驱动、CANN、torch_npu 或采集器中的某一项已被定罪。

## 1. 实际区别

| 项目 | Ascend 310P | Ascend 910B |
| --- | --- | --- |
| Runtime 芯片配置 | `CHIP_DC` | `CHIP_910_B_93` |
| 调度引擎 | `HWTS`，`isStars=false` | `STARS`，`isStars=true` |
| 普通任务结果回填 | 从 TS 结果的两个 32 位值拼成 64 位 timestamp | 从逻辑完成队列 `logicCq.u1.timeStamp` 读取 |
| 非 Stars 的额外读取分支 | `GetDisableThread()` 为真时使用 `UpdateTimeline()` / stream timeline | 不走该非 Stars 分支 |
| Event 计时换算 | `RT_DEFAULT_TIMESTAMP_FREQ` | `RT_CHIP_CLOUD_V2_TIMESTAMP_FREQ` |
| Runtime 平台库 | `libruntime_v100.so` | 同为 `libruntime_v100.so` |

因此不能将区别简单说成两者加载完全不同名字的 Runtime 库；共同库内仍然有芯片分派。
上述特殊 timeline 分支有运行条件，当前没有现场证据证明该条件在用户 310P 容器里启用。

一手源码，固定版本：

- [CANN Runtime 官方项目](https://gitcode.com/cann/runtime)，本地取证采用其 GitHub 镜像
  `hicann/runtime` 的 `v9.1.0`，commit `0b4af8a31c75a1f1f43638a8f75b9e3c6182e031`。
- [310P 配置](https://github.com/hicann/runtime/blob/0b4af8a31c75a1f1f43638a8f75b9e3c6182e031/src/runtime/config/dc/dev_info_reg.cc#L95)。
- [910B 配置](https://github.com/hicann/runtime/blob/0b4af8a31c75a1f1f43638a8f75b9e3c6182e031/src/runtime/config/910_B_93/dev_info_reg.cc#L148)。
- [HWTS 回填](https://github.com/hicann/runtime/blob/0b4af8a31c75a1f1f43638a8f75b9e3c6182e031/src/runtime/core/src/task/task_info/event/event_task_v100.cc#L26)。
- [Stars 回填与共同完成处理](https://github.com/hicann/runtime/blob/0b4af8a31c75a1f1f43638a8f75b9e3c6182e031/src/runtime/core/src/task/task_info/event/event_task.cc#L92)。

下载的源码及逐文件 SHA 在
`output/event_timestamp_hw_validation_20261116_cecba771/research_310p_910b/runtime_v910/`。
这份开源 tag 是实现依据，不是对用户已安装 Runtime 二进制的逐字节核验。

## 2. 负数到底怎么产生

`ApiImpl::EventElapsedTime(start, end)` 委托 `end->ElapsedTime(start)`。
`Event::ElapsedTime` 在读取两侧时间戳后，先检查是否为 `UINT64_MAX`，
再将两个值各自转换为 double 后相减，除以设备频率，结果转 float 返回。
没有检查结束时间戳是否小于开始时间戳。

其等价计算为：

```text
elapsed_ms = (double(end_timestamp) - double(start_timestamp)) / positive_frequency
```

- 两侧有 `UINT64_MAX`：返回 `RT_ERROR_EVENT_TIMESTAMP_INVALID`。
- 两侧均不是哨兵，但 end < start：这段计算可以正常返回负数。
- 频率不同只会缩放结果，单独不能把正常正区间变成负数。

依据：[v9.1.0 Event::ElapsedTime](https://github.com/hicann/runtime/blob/0b4af8a31c75a1f1f43638a8f75b9e3c6182e031/src/runtime/core/src/event/event.cc#L1033)。
同文件 `UpdateLatestRecord` 在 RECORDING 时将 timestamp/timeline 置为哨兵，
在匹配 stream/task 的完成回填时写入结果。无效值可能出现在未正确回填等状态下，
但仅凭错误码还不能确定是哪一种状态转换导致。

本项目 `profiling.py` 的 NPU 路径直接返回 `float(a.elapsed_time(b))`。
诊断字段 `lm_head.start` 直接调用 `origin.elapsed_time(lm_head_start)`，
没有使用两台机器的 CPU 时间，也不是由 DP 或通信耗时相减产生。
如果同一 stream 上按 origin → head_start 顺序记录，却返回 -179.43 ms，
说明采集到的设备时间轴已不满足该记录顺序；它不是正常计算耗时。

`runner_overhead_ms` 则属于第二类、上层派生负数：

```text
overhead = 外层执行和采样总区间 - 所有层耗时 - endpoint 耗时
```

用户提供的早期一轮记录中，外层约 99.834 ms、层之和约 93.460 ms，
head 约 280.44 ms，其他 norm 约 0.004844 ms。相减自然会得到约 -274.071 ms。
这里的负 residual 是分项 Event 区间已经不一致之后的结果，DP 校验只是发现它。
不同轮的 -179.43 和 -9691.91 不合并为同一条原始记录。

## 3. 不能拿来直接解释本次故障的差异

官方文档支持在 Atlas 推理系列上使用 Event 计时，不能说 310P 没有此能力。
Event SYNC 资源在推理系列为 1023，A2 系列为 65536；这是实际资源区别，
但文档描述资源不足时可能阻塞，并未说会返回正常状态的负耗时。
本次没有资源耗尽证据，故不能把 1023 上限当根因。

[CANN 9.1 Event 创建及资源约束](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0081.html)。

`torch_npu` upstream v2.10 的 `elapsed_time` 已会排空 CPU task queue、同步两侧 Event、
再调 CANN。因此，重复加一个 Python synchronize 不是已经找到原因的证明；用户现场
也已经报告同步读取后仍然失败。
[torch_npu NPUEvent.cpp](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/NPUEvent.cpp#L148)。

310P 还有独立的 model runner、NZ 数据布局和部分 CPU/stream 处理路径；这些会改变
任务提交及同步行为，但目前没有证据把某一个具体 sampler/算子确定为异常时间戳来源。
910B 同模型通过说明这套采集器在此次 910B 配置下正常，不能用它跳过 310P 的路径核对。

## 4. 当前最短定位路径

下一步应对齐失败 Event 的 native handle/marker 与 Runtime 的
`curNs/baseNs/curEventId/baseEventId`，再对齐对应 `event_record` 的
`stream_id/task_id/timestamp/timeline`：

1. 若 Runtime 报告结束值小于开始值，进一步追该 Event 的完成回填是否匹配当前 task。
2. 若仍为无效哨兵，追这次 record 是否产生并回填了 timestamp/timeline。
3. 若 Runtime 两侧均合理而 Python 配对错误，则查采集器对象/marker 归属。

优先使用已有失败诊断和 plog；若日志级别未包含这些字段，才需要一次最小现场采集。
不改 DP、不修改 Gloo、不取绝对值或伪造有效样本。

## 5. 已完成 910B 对照的来源

`output/event_timestamp_hw_validation_20261116_cecba771/evidence/real_profile_audit.json`：
CodeLlama-34B、4×910B3、PP4/TP1、12/12/12/12，32 条 stage 记录、384 个 layer interval
均通过有限非负与区间守恒检查，四个 rank 采集完整结束；stage1→2 CPU/Gloo 另有真实内容验证。
本轮只是重新阅读该完成证据，没有重复执行模型。

## 6. 2026-11-17：由芯片差异提出的具体故障假设

优先调查的假设是：HWTS 真实模型任务流中的 Event 结果回填或 stream/task 关联异常，
使本次 Event 的时间读数没有正确对应本次记录。这个假设能同时容纳“非哨兵但逆序”
与“哨兵无效”两类现象，但二者尚未通过现场原始时间戳证明来自同一个缺陷。

依据有三层：

1. 时间戳不是 Python 调用 record 时取得的 CPU 时间。非 STARS 通过 TS 结果回传，
   STARS 通过逻辑 CQ 的时间戳字段回传；随后都要进入 `RecordComplete`，
   再由 `UpdateLatestRecord` 按 stream/task 身份匹配后更新当前 Event。
2. `QueryEventTask` 有一处实际可见的分支差异：STARS 在进度已到、且为 TIME_LINE
   Event 时，若 timestamp 仍为无效哨兵，会回到 INIT 并触发 TaskReclaim；
   非 STARS 分支主要比较当前 task ID 与 CQ 共享内存里的最后完成 task ID。
   这不是通用同步缺陷的证明：该查询路径受 `GetDisableThread()` 等条件限制，
   `Event::Synchronize` 另有 WaitTask/Notifier 路径，torch_npu 的 elapsed_time
   本身也会同步两个 Event。不能直接据此宣称“310P 同步不等回填”。
3. 如果某次关联拿到了旧值、错误值或尚未产生有效值，则分别可能表现为负时间差
   或 timestamp invalid。真实模型的逐层密集 Event、任务回收及末 stage 的额外工作
   是比小矩阵测试更复杂的触发条件；这是对“为什么小测试通过、模型失败”的待证解释，
   不证明资源已耗尽或某个算子已被定位。

相关源码位置：`event.cc:247-264`（按 stream/task 回填），
`event.cc:678-723`（QueryEventTask），`event.cc:874-963`（Synchronize），
`event_task.cc:197-216`（完成回填及日志）。均来自上述固定 v9.1.0 快照。

排除不成立的推导：两个 uint32 在转为 uint64 后拼接本身是正常实现，不能因此宣称
310P 有 32 位溢出；两个正频率配置不同不会单独改变时间差的符号；只要 Event
确实在同一 stream 正序记录、身份和时间戳均正确，设备执行慢或其他 stream 的工作
也不能让该顺序上的结束时间戳小于开始时间戳。范围遗漏可能造成分项守恒失败，
却不能单独解释同一捕获内原始 origin→module_start 的大幅负偏移。

最有区分力的检查是对齐某个失败 Event 的 Python marker/handle、Runtime
stream_id/task_id 和回传 timestamp/timeline。若 Event 身份关联错，修关联或生命周期；
若身份及记录顺序正确而设备返回值仍逆序/无效，则继续定位 HWTS 回报及实际 Runtime/
驱动/固件组合。当前证据不足以把某个具体环节确定为唯一根因。
