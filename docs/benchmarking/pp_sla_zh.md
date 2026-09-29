# CodeLlama34B 异构 PP 实验

## 实验范围

- 单机4张910B，PP=4、TP=1；计算耗时分别模拟为原来的1、1、2、4倍。
- 保留原生 HCCL 通信。本轮不模拟跨机100 Gbps网络，结果不能称为真实“两机四卡”结果。
- CodeLlama-34b **base**，48层，FP16。均分为12/12/12/12。
- 比较均分、Latency DP、Throughput DP；相同切分只测试一次。
- 最终指标：P99 TTFT不超过150 ms或500 ms时，实测能支持多少并发用户；相对于均分是否提升50%。
- DP预测的是平均microbatch成本，不是P99 TTFT。是否达到SLA必须实际请求服务验证。

## 数据与请求

使用ShareGPT首轮user/assistant文本对。用本地模型的tokenizer计算长度：

- 输入4～2048 tokens，输出4～1024 tokens，超出范围的记录过滤，不截断。
- 输入使用user文本，输出长度使用参考回答的token数。`ignore_eos=True`保证生成指定长度。
- 去除重复的tokenized prompt，再按输入、输出长度分组抽样。
- warmup 512条、profiling 1024条、测试8192条；三组不重叠。
- 这是性能测试，不评价CodeLlama回答ShareGPT问题的质量。
- 并发C表示C个模拟用户，每个用户收到完整回答后立即发送下一条请求，无思考时间。
- 所有方案使用相同测试请求及长度；记录每条请求的TTFT、总耗时和失败情况。
- P99直接对逐请求TTFT计算。存在失败请求时不判为SLA达标。

```bash
python benchmarks/pp_sla.py prepare \
  --dataset datasets/sharegpt.json \
  --model /path/to/CodeLlama-34b/model \
  --output output/pp_sla_34b/data
```

## 服务和内存

固定服务参数：`max_model_len=4096`、`max_num_seqs=256`、
`max_num_batched_tokens=2048`、chunked prefill开启、prefix cache关闭、eager执行。
模拟减速使用主代码逐层计时和额外等待；profiling与实际对照服务使用同一设置。

所有切分固定1024个KV blocks，每个128 tokens，即131072 tokens容量。
该模型FP16 KV每层约512 MiB。并发较高、累计context超过容量时可能出现preemption；
不能把它描述为支持256个满4096-token context同时驻留。

DP内存检查使用实际加载的逐层权重、endpoint权重和设备总内存；
每卡预算为总内存的85%。从allocator peak中扣除已计入的权重和KV后，
将剩余值与6 GiB取较大者作为runtime等开销预算，另留2 GiB安全余量。
这部分余量是保守设定，不能保证所有未运行过的切分都一定可加载；候选还要实际启动检查。

`profile/memory.json`仅用于该脚本的离线DP，必须与同目录的`protocol.json`
一起使用，其中记录固定KV blocks。不要把它传给不支持固定KV override的engine自动切分入口。

## 先验证流程，再做正式对比

先加载Ascend环境，并从已提交的代码运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
python benchmarks/pp_sla_experiment.py \
  --model /path/to/CodeLlama-34b/model \
  --data output/pp_sla_34b/data \
  --output output/pp_sla_34b/pilot --mode pilot
```

该命令依次完成：

1. 均分启动，8条独立warmup请求；正式32条请求、并发8采trace。
2. 排除warmup，平均实际逐层耗时；通信使用匹配send/recv的trace。
3. 用上述实测成本和新的内存预算，计算两个目标的DP切分。
4. 分别启动均分和DP方案，关闭trace输出，先warmup，再验证请求。
5. 每个方案测32条单输出token请求，检查单请求TTFT；另测32条原始输出长度请求、并发8。

第5步的单token请求只是TTFT检查，不能用其throughput代表原始数据。
这些少量请求只验证流程，不足以认定P99达标或确定最大并发数。

正式测试可复用已生成的DP方案：

```bash
python benchmarks/pp_sla_experiment.py \
  --model /path/to/CodeLlama-34b/model \
  --data output/pp_sla_34b/data \
  --output output/pp_sla_34b/pilot --mode sweep \
  --requests 2048 --concurrencies 1 2 4 8 12 16 24 32
```

每个并发点独立warmup后再测。默认并发列表可到256，但应先用小规模测试排除明显不适用的范围。
结果包含有限请求列表的开始和结束阶段；正式测试应有足够样本，边界附近重启服务重复测量。
`capacity.json`只报告**已测试并发点中**通过SLA的最大值；要找精确边界需补测中间点。
若均分连并发1都未达标，“并发提升50%”没有可用分母，应如实报告，不能放宽SLA后仍称达标。
若正式并发明显不同于profiling的并发8，应补采对应负载的trace，再验证切分。

脚本保留启动命令、commit、参数、逐请求结果、trace、DP输入与方案、内存观察和服务日志。
不要覆盖既有结果；独立重复实验使用新的output目录。
