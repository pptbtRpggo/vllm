# ShareGPT 分组 SLO 压测

在 `configs/bench.yaml` 中配置数据集和负载，将 `SLO.enabled` 改为 `true`，
在同一文件修改 `SLO.profiles` 和 `SLO.ratios`。服务启动后直接执行：

```bash
bash bench_tau.sh
```

需要一份独立实验配置时，复制完整的 bench YAML，再使用 `--config /path/to/bench.yaml`。
无需单独配置 SLO JSON；`--slo-config` 仍保留为兼容原生 benchmark 的临时覆盖。

`configs/bench.yaml` 中的 SLO 示例 定义两个档位：

| 档位 | TTFT 上限（ms） | TPOT 上限（ms/token） | 请求比例 |
| --- | ---: | ---: | ---: |
| tight | 1000 | 50 | 0.5 |
| loose | 3000 | 150 | 0.5 |

档位数量不限。可以修改档位名称、阈值和比例，也可以增加、删除档位；增删时同时修改 `profiles` 和 `ratios`。
比例非负且总和为 1；名称必须一一对应。将某组比例设为 0 可暂停为该组分配请求，仍保留其定义。
这里的动态配置是指**修改 YAML 后，下次启动 bench 生效**；一次运行开始时配置和请求 SLO 已固定，运行中不热更新。
`SLO.enabled: false` 且未显式传入外部 SLO 配置时不分配档位。

## 固定阈值和按分布采样

每个组的 TTFT、TPOT 都可以独立配置为固定正数或分布，两种形式可以混用。
下面将 `configs/bench.yaml` 的 `SLO` 段改为三组，其他配置保持原样：

```yaml
SLO_SEED: 42
SLO:
  enabled: true
  profiles:
    tight:
      ttft_slo_ms: 1000
      tpot_slo_ms: 50
    standard:
      ttft_slo_ms:
        distribution: normal
        mean: 2000
        std: 300
        min: 1000
        max: 3000
      tpot_slo_ms: {distribution: uniform, min: 80, max: 120}
    loose:
      ttft_slo_ms: {distribution: uniform, min: 3000, max: 5000}
      tpot_slo_ms: 150
  ratios: {tight: 0.3, standard: 0.5, loose: 0.2}
```

然后执行 `bash bench_tau.sh`。例如正式请求数为1000时，三组分别分配300、500、200条请求。
同组中使用固定值的字段相同；配置为分布的字段则对每条请求分别抽样。每条请求只在流量发送前采样一次，不会在Decode迭代中反复改变SLO。

| 写法 | 含义与参数 |
|---|---|
| `1000` | 固定阈值，必须为有限正数 |
| `{distribution: uniform, min: 1000, max: 3000}` | 区间内均匀采样；有限且 `0 < min < max` |
| `{distribution: normal, mean: 2000, std: 300, min: 1000, max: 3000}` | 在区间内对正态分布做截断采样；全部参数为有限正数，`min < max` |

TTFT阈值及其分布参数使用ms；TPOT阈值及其分布参数使用ms/token。
`normal` 的 `mean`、`std` 指**截断前**正态分布的参数，截断后的样本均值/标准差未必等于它们。
采样使用区间条件分布的逆CDF，避免负值，也不把越界样本截成端点值。
若上下限处于过远的尾部，或区间窄到概率质量小于 `1e-12`，配置会报错，需调整参数。
当前TTFT和TPOT在各组内独立采样；没有指定二者相关性，也不按prompt长度调整分布。

如不需要分组，只希望所有请求按分布生成SLO，保留一个组并将其ratio设为1即可。
删除某组时同时删掉对应ratio并重新调整剩余比例；只修改比例不需要修改代码。

## 分配与发送

- 先完成 ShareGPT 过滤、采样和必要的重复采样，再对实际请求列表分配档位。数据集文件无需修改。
- 每档请求数先取 `实际请求数 × 比例` 的整数部分，剩余名额按小数余数从大到小分配，同余数按档位名排序。例如 7 条请求、各 50%，示例配置得到 loose 4 条、tight 3 条。
- 用独立随机数生成器打乱档位分配，不改变数据采样和到达间隔的随机状态。`--slo-seed` 默认等于 `--seed`；复现实验需要保持数据、采样参数和两个种子一致。
- 阈值采样使用与档位打乱隔离的随机数流，二者均由 `SLO_SEED` 控制。仅把固定阈值改成分布不会改变相同比例、种子下的档位分配；配置、请求列表和种子相同时实际阈值可复现。
- 每条 HTTP 请求通过 `vllm_xargs.ttft_slo_ms` 和 `vllm_xargs.tpot_slo_ms` 传递自己的实际数值阈值，不把分布定义发给服务。全局 `--extra-body` 中其他字段保留，同名 SLO 字段以该请求已分配的值为准。
- 比例适用于每次 benchmark 调用。脚本的预热和正式压测分别采样、分配和统计；正式 goodput 不包含预热请求。

原生 `vllm bench serve` 也支持 `--slo-config` 和 `--slo-seed`，目前限定 `--dataset-name sharegpt` 与 `--backend vllm` 或 `openai` 的 completions 请求。现有 Tau 接口能接收这些字段；是否达到 SLO 仍取决于调度策略和负载。

## 逐条 goodput

请求必须成功、至少生成一个 token，并同时满足自己的 TTFT 和 TPOT 上限，才计为达标。失败请求计入总请求数，但不计入达标数。
对于分布采样，同组请求也可能具有不同阈值：逐条使用实际抽到的阈值判断，不使用分布均值或组平均阈值。

- `TTFT` 使用压测客户端测得的首 token 延迟。
- `TPOT = (请求完成耗时 − TTFT) / (实际输出 token 数 − 1)`，即平均每 token 耗时，不是每个相邻 token 间隔的上限。只有一个输出 token 时沿用现有口径，TPOT 记为 0。
- `goodput = 达标请求数 / 正式压测总耗时`，单位 req/s。
- `attainment_rate = 达标请求数 / 总请求数`，失败请求包含在分母中。

未分配档位的请求仍可使用原生 `--goodput` 全局阈值。两者同时设置时，请求档位覆盖全局 TTFT/TPOT 阈值；若指定了全局 E2EL 阈值，仍需同时满足 E2EL。

## 结果文件

脚本默认把每轮压测放在 `output/<服务运行目录>/bench/<本轮标识>/`：

- `bench_meta.json`：服务引用、数据集路径/大小/SHA256、SLO 配置快照与 SHA256、采样/分配种子和各阶段命令。
- `requests.jsonl`：发流量前保存正式请求 ID、ShareGPT 原始下标、长度和 SLO；正常测量结束后原子替换为逐条结果，含 TTFT/TPOT/E2EL、success、attained。中途退出保留 planned 行，success 为 null，不推断实际完成情况。
- `summary.json`：正式统计、总体及 by_profile goodput；trace 字段保存正式区间与校验，warmup 字段仅保存预热摘要。
- `error.log`：仅失败时保存控制台输出。

`summary.json` 的 `slo_evaluation.attainment_rates` 包含 `ttft`、`tpot`、`all`
三项达成率，取值为 0–1（0.8 表示 80%）。`by_profile.<组名>.attainment_rates`
提供同样的分组统计。TTFT/TPOT 分别以配置了对应阈值的请求数为分母，包含失败请求；
对应的分子、分母保存在 `ttft_good_requests` / `ttft_total_requests` 和
`tpot_good_requests` / `tpot_total_requests` 中。没有配置某项 SLO 时，该项达成率为
`null`。`all` 以至少配置一项 SLO 的请求数为分母，要求全部已配置条件同时达成；
它与保留的 `attainment_rate` 字段相同。总体统计直接累计请求，不对分组达成率取平均。

顶层 `request_goodput` 与 `slo_evaluation.request_goodput` 相同，都是全部达标请求数
除以正式压测时长，单位 req/s。分组 goodput 也使用同一个正式压测时长，因此分组值
相加等于总体值。未启用任何 SLO 时，不生成 `slo_evaluation`，`request_goodput` 为
`null`；预热统计单独在 `warmup` 中。

`requests.jsonl` 的 `attained` 使用三个子字段（原来的布尔值改为对象）：

```json
"attained": {"ttft": true, "tpot": false, "all": false}
```

- `ttft`：是否满足该请求的实际 TTFT 阈值。
- `tpot`：是否满足该请求的实际 TPOT 阈值。
- `all`：是否满足全部已配置 SLO；总体和分组 goodput 都以该值为准。
- planned 状态下三个字段均为 `null`；未配置某个指标时该指标为 `null`，完全未配置 SLO 时三个字段均为 `null`。
- 已配置 SLO 的失败请求或零输出请求，对应判定项为 `false`；不是把失败的零耗时当作达标。
- 如果同时指定全局 E2EL 上限，`all` 也要求 E2EL 达标，所以可能出现 `ttft=true, tpot=true, all=false`。

原生 benchmark 的逐请求 SLO evaluation 使用同样的对象结构。读取新结果的分析脚本需使用 `attained.all`，不能再把 `attained` 对象直接当布尔值。

不保存生成文本；不再额外保存配置、分配表、预热明细和正常运行日志。原始 trace 继续在本机临时目录的单个文件中，路径记录在 `server_meta.json`。

脚本使用新增的原生 `--request-output <JSONL路径>` 输出逐条记录。直接运行 `vllm bench serve` 而不传此选项时，原有结果和 `.slo_assignment.json` 格式保持兼容。
