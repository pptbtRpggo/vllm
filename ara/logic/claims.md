# 有证据支持的结论与待验证假设

证据编号见[证据索引](../evidence/README.md)。supported均附适用范围，未表示普遍定律。

## C01: 上游等待会改变下游被测基础区间
- **Statement**: 在已测CodeLlama均分、eager控制实验中，Stage3自身不等待，仅Stage2多等20 ms，Stage3短decode基础区间由24.77 ms变为14.61 ms；CPU准备/提交时间没有同步减少。支持异步提交与设备执行时序改变区间空档的解释。
- **Status**: supported
- **Provenance**: ai-suggested
- **Scope**: 只直接覆盖该受控配置；未逐kernel测出active time，也不能精确分解旧trace的每一毫秒。
- **Falsification criteria**: 用同负载独立CPU/NPU时间线重复干预；若基础区间不随上游改变，当前配置中的解释需修正。
- **Proof**: E01
- **Dependencies**: 无
- **Tags**: measurement, profiling, protocol, reproducibility

## C02: 等待公式自洽不能独立验证硬件等效精度
- **Statement**: Stage3的14.61 ms基础区间追加43.83 ms得到58.44 ms，总计/本次基础区间约4；相对24.77 ms无等待对照仅约2.36。内部公式正确与相对稳定参考的减速精度是不同检验。
- **Status**: supported
- **Provenance**: ai-suggested
- **Scope**: 整个stage的经过时间和纯设备算力不是同一物理量；整stage没有变4倍也不能单独反证真实四分之一算力硬件会如何。
- **Falsification criteria**: 需以不复用等待自身计时的独立观测检验指定目标：stage经过时间或算子活跃时间。
- **Proof**: E01, E03
- **Dependencies**: C01
- **Tags**: measurement, profiling, protocol, reproducibility

## C03: 逐层零等待路径存在可测开销
- **Statement**: Qwen逐层等待kernel原型中，零等待平均TTFT相对原生路径在eager增加5.6%，FULL_DECODE_ONLY Graph增加6.4%；Graph decode的逐层倍率较接近设置，prefill和stage总区间并未同样通过。
- **Status**: supported
- **Provenance**: ai-suggested
- **Scope**: 特定原型及负载；不能推断所有插桩均有相同开销，或Graph已修复全部prefill问题。
- **Falsification criteria**: 在匹配原生/零等待路径、模式及形状下重测，若开销落入预定容差，则该新实现可通过。
- **Proof**: E02, E03
- **Dependencies**: 无
- **Tags**: measurement, profiling, protocol, reproducibility

## C04: 固定输出会通过闭环到达过程影响TTFT
- **Statement**: Qwen并发64的控制实验：固定16闭环在420 ms门槛下达标率17.38%/17.58%；自然输出为86.33%/86.33%；固定16重放旧到达时刻为84.18%/84.38%，并将迟发时间纳入评价。
- **Status**: supported
- **Provenance**: ai-suggested
- **Scope**: 直接控制证据只覆盖Qwen C64及Code C16代表负载，不能为全部环境分配原因占比。
- **Falsification criteria**: 固定同样到达过程后比较输出规则；若差异仍同等巨大，则需要其他机制解释。
- **Proof**: E05
- **Dependencies**: 无
- **Tags**: measurement, profiling, protocol, reproducibility

## C05: DP推荐切分的端到端表现需要验证
- **Statement**: Qwen272/8负载两批重采的旧平均成本DP仍推荐1/27，其512请求P99为737.8 ms；预设27/1候选为395.9 ms。证明该次预测选择未得到已测候选的最低P99。
- **Status**: supported
- **Provenance**: ai-suggested
- **Scope**: 仅对已测候选和该协议成立；不能声称27/1为实测全局最优，也不能将全部偏差归因于单一模拟或phase混合问题。
- **Falsification criteria**: 在匹配负载与留出测试中验证候选排序；比较成本聚合、约束、执行模式及切分变化。
- **Proof**: E04, E12
- **Dependencies**: 无
- **Tags**: measurement, profiling, protocol, reproducibility

## C06: 门槛来源、场景分位数和覆盖目标应分别定义
- **Statement**: 用户明确P95/P85仅用于选取毫秒门槛，P99/P90仍为实测TTFT场景指标，实测覆盖目标为100%。评价澄清在测量开始后记录，测量数据及冻结协议未据此改写。
- **Status**: supported
- **Provenance**: user-revised
- **Scope**: 这是本项目已确认的评价定义；不是通用业务SLO，也不等于项目原P99/P90验收要求已达成。
- **Falsification criteria**: 任何报告、图标题、统计字段或验收语句将三个概念混用，应判为口径不一致。
- **Proof**: E07, E08, E14
- **Dependencies**: 无
- **Tags**: measurement, profiling, protocol, reproducibility

## C07: 匹配协议校准仍不能强制独立验证比例
- **Statement**: 固定16新校准与验证各512请求，门槛在验证前冻结；Code25G C16的P95来源门槛下新验证89.5%，两个P85来源门槛下新验证81.8%；不存在强制凑成95%/85%的处理。
- **Status**: supported
- **Provenance**: ai-suggested
- **Scope**: 有限样本、输入长度分布和闭环时序均有影响；新门槛是当前协议的基线延迟参考，业务可接受性另需定义。
- **Falsification criteria**: 在真正同协议独立运行、相应长度分布下重复验证；报告波动和区间，不用验证反复重定门槛。
- **Proof**: E06, E10
- **Dependencies**: 无
- **Tags**: measurement, profiling, protocol, reproducibility

## C08: 当前软件等待已等效真实项目异构硬件
- **Statement**: 未建立完整的真实硬件等效验证，不能依据软件scale或同次基础区间内部倍率证明TFLOPS/HBM/CPU等效。
- **Status**: untested
- **Provenance**: ai-suggested
- **Scope**: 保留为未验证假设，不得写成实验结论。
- **Falsification criteria**: 需要真实异构硬件或有明确物理目标、独立观测的匹配验证。
- **Proof**: E01, E02, E03
- **Dependencies**: 无
- **Tags**: measurement, profiling, protocol, reproducibility

## C09: 当前DP在所有负载上给出实测全局最优
- **Statement**: 完整枚举验证的是给定输入成本与约束下的预测最优；现有实测有限候选及单轮正式结果不足以证明所有负载、真实硬件或长期P99全局最优。
- **Status**: untested
- **Provenance**: ai-suggested
- **Scope**: 现有66点结果支持软件设置下的配对策略表现，不能替代实测全局最优证明。
- **Falsification criteria**: 需要覆盖候选空间的实测验证，或经独立验证有误差保证的模型与目标对应论证。
- **Proof**: E04, E07, E09, E11
- **Dependencies**: 无
- **Tags**: measurement, profiling, protocol, reproducibility
