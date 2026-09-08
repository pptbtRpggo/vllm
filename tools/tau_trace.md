# Tau trace 开关与写入

```bash
# 正常业务 / 测吞吐与 SLO：关闭 trace
bash serve_tau.sh /path/to/model --no-trace

# 采集耗时拟合样本：开启 trace（保持原有默认行为）
bash serve_tau.sh /path/to/model --trace
```

也可以设置 `TRACE_ENABLED=0` 或 `1`，命令行开关优先。开关在服务启动时生效，切换需要重启。直接使用原生 `vllm serve` 时，不设置 `--tau-batch-trace` 和 `TAU_BATCH_TRACE` 即默认不记录；Ascend 采集还需使用 `TauAscendWorker`。

关闭时，脚本不传 trace 路径、不加载采集专用 worker，并清除继承的 `TAU_BATCH_TRACE`。调度器不生成 trace forward ID，worker 跳过采集计时及额外同步。`server_meta.json` 中 `trace` 为 `null`。`bench_tau.sh` 仍校验请求完成数、保存压测指标和逐条 SLO 结果，但跳过 trace 覆盖检查，不生成虚假的“trace 检查通过”文件。

开启时，原始记录仍集中写到一个 JSONL。默认路径在本机临时目录，启动输出及 `server_meta.json` 给出确切路径。建议使用本机 `/tmp`，不要把高频原始 trace 直接写到 NFS；采集完成后可复制到长期保存目录。

写入使用持久文件句柄，每条记录编码后通过一次 `os.write` 追加，不再经过文本缓冲与逐行 `flush`。已识别的本地 Linux 文件系统省去跨进程文件锁；NFS 和未知文件系统保留锁。进程内线程锁仍保留。写入返回后记录可供其他进程读取，不需要等待后台队列；不执行 `fsync`，不承诺断电持久性。短写会报错，不能把不完整 trace 当作有效样本。

这里没有加入异步写队列：它还需要处理队列积压、丢记录、进程退出排空和压测边界 flush。当前同步单次追加保留已有的记录顺序和按字节区间校验/拟合接口。

写文件优化不会移除采集模式中的 `torch.npu.synchronize()`。`compute` 仍为主机计时，并在结束前等待 NPU 完成；这项同步本身可能改变流水线重叠。性能结论必须区分 trace 关闭、trace 开启和写入方式，不能把启用 trace 的全部开销归因于文件锁。
