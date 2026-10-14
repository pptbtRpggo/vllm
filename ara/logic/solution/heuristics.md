# 下次实验的防错清单

这些规则为本次复盘提出的工作建议，Provenance均为ai-suggested；不表示用户已逐条确认。

适用范围：同类LLM推理、模拟异构、profiling/DP、SLO及策略对照实验。

## H01: 先把评价目标写成完整合同
- **Rationale**: 先写模型、硬件/网络、输入输出分布、到达过程、基础和扩展并发、执行模式、TTFT统计量、毫秒门槛、覆盖比例、观察窗口与失败计入方式。P95/P85门槛来源、P99/P90统计值、100%观测目标分别列字段。性能可达性回推与业务可接受性另行说明。
- **Provenance**: ai-suggested
- **Sensitivity**: low
- **执行检查**: 正式结果文件、图、PPT逐字段一致；协议变更有新ID；扩展并发沿用基础门槛明确标为压力测试。
- **Claim refs**: C06/C07
- **Code / evidence refs**: E07, E08, E10

## H02: 计时区间必须有边界和包含项
- **Rationale**: 对CPU准备/提交、device event区间、算子active time、通信区间和请求TTFT分别定义。每个值记录在哪个线程/stream、起止位置、前置依赖及是否含空档。相同比较按step+batch_id对齐，并区分prefill/decode/mixed；embedding和lm head单独列项。
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **执行检查**: 不把stage区间直接称纯计算，不把event替换当作自动消除空档；计时字段与边界图相互对应。
- **Claim refs**: C01/C02
- **Code / evidence refs**: E01, E12, E13

## H03: 模拟机制先通过独立验收再做全量比较
- **Rationale**: 分开检验原生对照、带载体/插桩的零等待、开启等待的基础/额外/总区间。选择同输入、microbatch形状、layer数量、phase和模式的参考量；检查上游加等待时基础值是否漂移。先明确模拟的是stage服务时间还是设备算子工作时间，再验证目标。
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **执行检查**: 容差预先定义；倍率验算使用独立测量；eager/prefill/Graph decode各自通过。未通过的实现作为受限软件模型报告。
- **Claim refs**: C02/C03/C08
- **Code / evidence refs**: E01, E02, E03

## H04: 工作负载包括到达过程，不只有数据集与并发
- **Rationale**: 保存逐请求输入长度、输出规则、计划到达/实际发出/首token/结束时间。闭环输出变短会改变补发节奏，不能默认更快或复用原TTFT门槛。输出、机制、budget、chunked、Graph或cache改变时重新判定校准适用性。
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **执行检查**: 校准与正式协议diff明确；固定到达率/时间线和固定并发两类实验按各自目标选择，重放迟发等待不得隐藏。
- **Claim refs**: C04/C07
- **Code / evidence refs**: E05, E06, E10

## H05: 用端到端留出验证检查DP模型的有效性
- **Rationale**: 先列成本模型的目标和假设：层成本是否随形状/phase/切分改变、通信与端点成本是否充分、平均成本与P99目标如何对应。prefill、decode、mixed分别统计；不用一个被大量decode支配的平均默认代表TTFT。少量候选比较预测值、实测值和排序，再决定是否扩大全量。
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **执行检查**: DP完整枚举检查算法；独立serving候选检查模型。均分、推荐切分、预先选定邻近切分均保留；调参集与最终测试集分离。
- **Claim refs**: C05/C09
- **Code / evidence refs**: E04, E07, E12

## H06: 异常用定点干预，不先给强因果故事
- **Rationale**: 出现异常先核对数据身份、配置和计时边界，再提出能区分多个原因的干预。保持其余变量固定，采用配对请求，必要时两轮正反顺序。用只改上游等待、只改输出或只改到达节奏的实验排除解释。
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **执行检查**: 结论注明直接观测/干预支持/未验证推测；诊断同步不会未经验证成为生产修复。
- **Claim refs**: C01/C04
- **Code / evidence refs**: E01, E05

## H07: 统计设计按尾部和相关性决定
- **Rationale**: 记录请求数、microbatch区间数、独立服务运行次数。512请求的P99由约5个最慢尾部请求附近决定；增加同请求decode步数不能视为增加独立请求。必要时跨运行重复，按请求或运行层级重采样，不独立重采样相关layer/steps。
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **执行检查**: 同时给P50/P90/P99、超门槛条数、最大值；100%只能称本轮样本全部达标，不承诺未来绝无违约。
- **Claim refs**: C07/C09
- **Code / evidence refs**: E01, E02, E06, E09

## H08: 先小而有辨别力，再决定全量实验
- **Rationale**: 按机制检查、代表性负载试跑、成本预测验证、SLO校准/冻结、正式测量、交付核对依次推进。时间预算按服务加载、warmup、profile、正式请求、整理复制分项估算并用试跑修正。减少decode是负载变更，独立标识并重审SLO。
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **执行检查**: 模拟/模型关口未通过时停止扩大全量；不为更低数字盲目重采或重跑同类诊断。
- **Claim refs**: C03/C04/C05
- **Code / evidence refs**: E02, E03, E04, E05, E07

## H09: 把原始数据到图表做成可复算链条
- **Rationale**: 保存代码commit、冻结运行脚本、runtime server_info、实际层切分、库SHA、数据ID/token/hash、独立warmup、原始请求/trace和绘图字段。成功点身份与输出校验后跳过；恢复保留故障与协议记录；light快照与最终全量备份区别标识。
- **Provenance**: ai-suggested
- **Sensitivity**: low
- **执行检查**: 图表和百分比由canonical请求计算；JSON/JSONL含trace逐文件SHA；ZIP CRC；图标题分位数、门槛和达标率逐格核对。
- **Claim refs**: C06
- **Code / evidence refs**: E08, E10, E11, E14, E15

## H10: 探索更优与独立证明分开
- **Rationale**: 可以用探索数据选择更好切分或调整成本模型；正式评价用未参与选择的样本或测量轮次，保留全部候选和失败。报告按已测范围说最好、预测最优或观测提升，避免把它们写成全局/真实硬件保证。AI解释先查实现与证据，推测明确标注，不用可信故事代替验证。
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **执行检查**: 不挑最小半轮P99，不混换协议、数据或来源，不据验证集反复抬门槛；纠正后的图表同步重算结论。
- **Claim refs**: C05/C08/C09
- **Code / evidence refs**: E04, E08, E09
