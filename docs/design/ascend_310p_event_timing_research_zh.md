# Ascend 310P Event 计时：官方要求与当前 profiler 的边界

调研日期：2026-11-16。本文调研仅作源码／文档审查，没有执行新的 NPU 实验，
没有修改 SDK 或驱动；调研后的采集路径改动见文末。当前用户镜像是定制环境；以下 upstream v2.10.0
源码不能代替已安装 `torch-npu 2.10.0.post4` wheel 的实际文件／二进制核对。

## 结论

使用 `torch_npu.npu.Event(enable_timing=True)`、在目标 stream 的工作前后 record、
等待完成后调用 `elapsed_time`，是有官方 API 和 upstream 实现依据的设备计时方法。
不能说“310P 不支持 Event 计时”。但是，没有找到官方对我们这套
“逐模块 forward hook 创建大量 Event、后台读取、把各层之和与外层区间相减”
的完整方案作过正确性保证。它是本项目自定义的 profiler，需要在真实模型路径验证。
本轮没有找到可以直接解释已报告 `-179.43 ms` 或 `-9691.91 ms` 的官方已知问题或修复。
这表示根因尚未找到，不表示这类问题不存在。

## 1. elapsed_time 与 recorded_time 不能混为一谈

[CANN 9.1 的 aclrtEventElapsedTime 文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0090.html)
列出 Atlas 推理系列产品支持；功能是求两个 Event 的耗时，单位 ms。
文档给出的调用顺序是：创建 Event → 同一 stream 记录起点和终点 →
`aclrtSynchronizeStream` → `aclrtEventElapsedTime`。
本轮抓取上述正式版页面间歇失败；相同要求亦在可读取的
[CANN 9.1 beta1 页面](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/API/runtimeapi/aclcppdevg_03_0090.html)
列出。beta 文档可作交叉核对，不能冒充对用户实际安装二进制的核验。

`recorded_time()` 对应获取单个 Event 的绝对设备时间戳，不能用它的产品支持范围
推导 `elapsed_time()` 是否支持。我们当前 profiler 调用的是 `elapsed_time()`。

## 2. torch_npu 的同步比 Python 调用外观多一层

[upstream v2.10.0 NPUEvent.cpp](https://raw.githubusercontent.com/Ascend/pytorch/v2.10.0/torch_npu/csrc/core/npu/NPUEvent.cpp)
中 `NPUEvent::elapsed_time` 实际执行：

1. 检查两个 Event 已创建；
2. `emptyAllNPUStream()`，排出相关 CPU task queue；失败时记录错误；
3. 分别 `aclrtSynchronizeEvent` 同步这两个 Event，并检查错误码；
4. 调用 `aclrtEventElapsedTime`，并检查错误码。

因此，不能仅凭 Python 代码没有显式 `stream.synchronize()`，就断言读取时工作未完成。
故障版本的首次后台读取还检查了 capture 末 Event 的 `query()`；故障后重读又显式同步了
末 Event。用户已报告同线程、同 stream、同步后重读仍出现负偏移，
所以“少一次显式 stream 同步”目前只是可对照的差异，不能宣称它已是根因。

## 3. Event flags 与 310P 资源限制必须按 API 分开

[upstream v2.10.0 Event.cpp](https://raw.githubusercontent.com/Ascend/pytorch/v2.10.0/torch_npu/csrc/npu/Event.cpp)
显示：若 runtime 提供 `aclrtCreateEventExWithFlag`，`enable_timing=True`
使用 `ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC`；否则使用 `ACL_EVENT_TIME_LINE`。
普通默认 Event 的创建 flags 与启用计时的 Event 不相同。

[AclInterface.cpp](https://raw.githubusercontent.com/Ascend/pytorch/v2.10.0/torch_npu/csrc/core/npu/interface/AclInterface.cpp)
显示 wrapper 在 Ex 符号可用时优先调用 Ex API，不能只看 wrapper 的函数名
`AclrtCreateEventWithFlag` 就认定实际调用了旧接口。

[CANN 9.1 ExWithFlag 文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0081.html)
同时说明了两个容易混淆的事实：

- **创建 Event 对象本身**不受硬件 Event 资源数限制。
- 若含 `ACL_EVENT_SYNC`，在 **record 时**才申请硬件资源，可能阻塞等待释放；
  该页对 Atlas 推理系列仍列出每 Device 最多 1023 个资源。

因此，“Ex API 创建不受限”不能简化成“整个生命周期绝无 1023 上限”。
反过来，也不能据此断定本次已经耗尽资源：当前没有用户进程的实际在途 Event 数、
native 其他 Event 占用或资源失败证据；文档描述的是资源申请／阻塞约束，
没有说到达上限会正常返回负耗时。

以本轮最新负载作条件算术：12 层末 stage 的逐层 Event 24 个，final norm 2 个、
execute 外层 2 个、sample 外层 2 个，约每 step 30 个。若四条 64-token prompt
在 budget 256 下每轮恰好一个 prefill step，warmup 1 轮加正式 2 轮约创建 90 个。
这不是现场实测的资源峰值，也不含 native 内部 Event；仅凭 1023 这个上限无法定罪。

旧版文档还直接点名 Ascend 310P 1023、910 65535：
[CANN 6.0.1 旧 CreateEventWithFlag](https://www.hiascend.com/doc_center/source/en/canncommercial/601/inferapplicationdev/aclcppdevg/aclcppdevg_03_0046.html)。
这条可作历史背景，不能代替当前已安装 CANN 9.1 的实际行为。

## 4. record、复用与销毁

[CANN 9.1 RecordEvent 文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0083.html)
把 record 列为异步接口。对 Ex API 的 Event，重复 record 会覆盖保存的任务；
首次 record 前查询可能返回 COMPLETE。不能把未记录、被覆盖的 Event 当成旧时间点使用。

upstream `NPUEvent::record` 会核对 stream 与 Event 所属设备；`query` 还检查
task queue 的已记录状态。析构经 `LaunchLazyDestroyEventTask` 延迟销毁。
Ex 文档禁用的是 `aclrtQueryEvent` 和 `aclrtQueryEventWaitStatus` 等具体接口，
upstream Python `Event.query()` 走的是 `AclQueryEventRecordedStatus`，该 wrapper
再调用 `aclrtQueryEventStatus`；不能只因为 Python 方法名叫 query 就认定它违反禁用条款。
本项目每次 mark 都新建 Event，并保留 capture 引用到后台读取；
从当前 Python 源码没有发现主动重新 record 同一对象的行为。
是否存在底层生命周期交互仍需真实句柄和实际 wheel 核对，不能从 Python 对象新建
直接证明所有底层资源在整个时间窗口内正确。

调研时的 `b92f762284` 版本 pending queue 使用无界 `queue.Queue()`，没有给待解析的
capture 数量设置上限。这是需要核对的资源管理风险，不是已证实的耗尽原因。
在新的一 token、少量 prefill 负载下，尤其不能直接用旧长输出的 Event 总量推断峰值。

## 5. forward hook 和后台读取的证据边界

[streams.py 的 Event API](https://raw.githubusercontent.com/Ascend/pytorch/v2.10.0/torch_npu/npu/streams.py)
规定 record 的 stream 设备必须匹配 Event，并给出 record／query／synchronize／elapsed_time 语义。
已查文档没有给出“310P 必须只在记录线程读取 elapsed_time”的规则；
也没有对本项目的后台读取和模块 hook 组合给出背书。
不能把“没找到禁止条款”当作“完整方案已获官方验证”。

forward 的 Python 返回只表示工作已提交，不能单靠前后 `perf_counter` 当设备执行时间。
在真正承载所测工作、且有正确依赖的 stream 上放 Event，才可能量到该设备工作。
记录处 stream ID 相同，是必要的排查信息；它不自动证明复杂算子内部的所有工作、
隐式 stream 或完成依赖都已被该区间完整覆盖。

## 6. 对当前故障的判断

已知实机证据来自本线程用户：小矩阵 Event 检查正常；串行／队列读检查正常；
真实末 stage 的 lm_head 偏移却在同步后仍为负，同一 capture 只记录到一个 stream，
记录和重读线程 ID 相同。这不足以把问题归因到 Python 跨线程读或两台机器时钟不同。

可审慎回答：**底层 Event 方法有依据；当前自定义 profiler 在 310P 真实模型上的
组合行为没有通过验证。** 应保留异常原值，核对实际非法字段、实际二进制和 Event
生命周期／提交顺序。不能通过取绝对值、把大负值归零、删掉失败样本、或关闭校验
来宣称已修复。本轮调研没有验证任何新的修复方案。

## 7. 对照检查的参考流程

以下是依据 CANN 调用顺序写出的 torch_npu 对应示意，不是官方逐层 profiler 代码：

```python
start = torch.npu.Event(enable_timing=True)
end = torch.npu.Event(enable_timing=True)
stream = torch.npu.current_stream(device)
start.record(stream)
result = fn()  # 所测工作必须提交到该 stream，或有完整的依赖关系。
end.record(stream)
stream.synchronize()
elapsed_ms = start.elapsed_time(end)
```

在实际 310P 模型中用该流程作有限对照，才能继续比较记录位置、生命周期和读取路径。
同步会影响提交重叠和主机侧开销；该控制不能不加说明地替代原生性能结果。
不应在生产路径直接逐层同步，也不应从一个成功的小矩阵测试推断整个模型计时正确。

后续实现已改成 RPC 结束时同步目标 stream，并在记录线程读取和释放 Event，
后台只写数值快照（队列最多 8 条）。这是一条待真实 310P 验证的保守采集路径，
不是已经确认负值根因或已经实机修复的结论；不对每层进行同步。

## 8. 同步采集路径的实机反馈与后续诊断

用户随后报告同一路径在 pp3 的 `profiling.py:254 elapsed` 中抛出
`rtEventElapsedTime ... event timestamp invalid`，不再只是返回负值。
这确认异常发生在本项目读取设备 Event 耗时的调用，而不是离线 DP 的非负检查；
也说明上述同步路径没有解决实机问题。该行是所有配对共用的读取函数，
仅凭此行和 pp3 仍无法区分总区间、某个 decoder layer、final norm 或起止偏移。
LM head 已取消独立打点，不能因故障仍在最后 stage 就再次认定是 head 的独立计时。

[CANN 9.1 错误码](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1345.html)
将 `ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID` 定义为 107006，建议检查 Event 是否创建。
这是通用排查提示，并非证明本项目没有创建 Event；当前 Python 源码确实每次 mark
创建启用计时的 Event、record 后保留引用，读取循环结束前不清空列表。
底层资源、实际 wheel 和 runtime 行为仍不能由 Python 引用关系单独证明正确。

本次只补充失败配对的可读错误：rank、step、execute/sample、测量字段、
两侧 marker/原生句柄、已完成的读数和原始异常链。失败后不再查询、重录或重试，
不发布失败记录；同步方式、计时来源及数值门限不变。
相关回归使用主动注入异常的 CPU Event fixture，只验证诊断行为，不冒充真实 NPU 复现。
