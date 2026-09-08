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

可以修改档位名称、阈值和比例，也可以增加档位。阈值必须为有限正数，比例非负且总和为 1；`profiles` 和 `ratios` 的名称必须对应。`SLO.enabled: false` 且未显式传入外部 SLO 配置时不分配档位。

## 分配与发送

- 先完成 ShareGPT 过滤、采样和必要的重复采样，再对实际请求列表分配档位。数据集文件无需修改。
- 每档请求数先取 `实际请求数 × 比例` 的整数部分，剩余名额按小数余数从大到小分配，同余数按档位名排序。例如 7 条请求、各 50%，示例配置得到 loose 4 条、tight 3 条。
- 用独立随机数生成器打乱档位分配，不改变数据采样和到达间隔的随机状态。`--slo-seed` 默认等于 `--seed`；复现实验需要保持数据、采样参数和两个种子一致。
- 每条 HTTP 请求通过 `vllm_xargs.ttft_slo_ms` 和 `vllm_xargs.tpot_slo_ms` 传递自己的阈值。全局 `--extra-body` 中其他字段保留，同名 SLO 字段以该请求的档位为准。
- 比例适用于每次 benchmark 调用。脚本的预热和正式压测分别采样、分配和统计；正式 goodput 不包含预热请求。

原生 `vllm bench serve` 也支持 `--slo-config` 和 `--slo-seed`，目前限定 `--dataset-name sharegpt` 与 `--backend vllm` 或 `openai` 的 completions 请求。现有 Tau 接口能接收这些字段；是否达到 SLO 仍取决于调度策略和负载。

## 逐条 goodput

请求必须成功、至少生成一个 token，并同时满足自己的 TTFT 和 TPOT 上限，才计为达标。失败请求计入总请求数，但不计入达标数。

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

不保存生成文本；不再额外保存配置、分配表、预热明细和正常运行日志。原始 trace 继续在本机临时目录的单个文件中，路径记录在 `server_meta.json`。

脚本使用新增的原生 `--request-output <JSONL路径>` 输出逐条记录。直接运行 `vllm bench serve` 而不传此选项时，原有结果和 `.slo_assignment.json` 格式保持兼容。
