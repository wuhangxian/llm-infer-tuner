# SGLang 调优原则

这份文档记录跨模型、跨硬件的调优经验和判断方法。内容属于启发式知识
（`heuristic`），不能替代目标镜像的 `--help`、服务启动结果和 benchmark 实测。

## 最小可行卡数优先

在模型显存、运行时开销和目标上下文都能容纳，并且仍能预留足够 KV Pool
的前提下，优先选择能稳定运行的最少 GPU 数量。机器有更多 GPU，不代表
应该无条件扩大通信域。

这里的“能跑起来”不能只看模型权重是否装得下，还必须考虑：

- 权重和激活的运行时显存；
- 目标 `context_length` 下的 KV Cache；
- 目标并发下可预留的 KV Pool；
- CUDA Graph、通信 buffer 和其他运行时开销；
- 服务能否满足 JobSpec 的 SLA。

如果较少卡数虽然能够启动，但 KV Pool 过小、并发不足或无法满足 SLA，
则不能算作可行方案。

## TP 不是 GPU 数量

机器有 8 张 GPU，不代表最优部署方案就是 `tp_size=8`。TP 是需要测试和比较的
部署变量，而不是根据 GPU 数量直接推导出的硬约束。

### 原因

- 较小 TP 可能已经能够容纳模型和目标上下文，且跨卡通信更少；
- PCIe、NVLink 等互联拓扑会影响 TP 通信成本；
- 扩大 TP 可能提高总吞吐，也可能降低每卡吞吐、恶化 TTFT/TPOT；
- “使用全部 GPU”不等于性能最优，也不等于成本最优。

## 根据互联拓扑选择 TP/PP

当最小可行卡数大于 1 时，必须把卡间互联纳入并行策略选择：

### PCIe 卡间互联

PCIe 卡间通信带宽和时延通常弱于 NVLink。跨卡部署时，应优先评估能够
减少高频同步通信的 Pipeline Parallelism（PP），并与 TP 方案在相同 workload
下比较。PP 不是无条件更优，还要记录 pipeline bubble、首 token 延迟和负载均衡。

### NVLink 卡间互联

NVLink 卡间通信性能较好时，Tensor Parallelism（TP）对延迟通常更友好，
应重点评估 TP 的切分效率、通信开销和 KV Pool 容量。仍然不能仅凭 NVLink
就直接假定 TP=GPU 数量最优。

### SearchPlan 要求

除非 JobSpec、HardwareSpec、ModelSpec 或已有实测证据明确要求，否则：

1. 不得仅因为机器有 N 张 GPU，就把 `tp_size=N` 标记为 `hard`；
2. 应先判断模型权重、运行时开销、目标上下文和 KV Cache 对显存的要求；
3. 优先寻找满足显存、KV Pool 和 SLA 的最小卡数方案；
4. PCIe 拓扑下的多卡方案应优先纳入 PP 候选；NVLink 拓扑下应重点纳入 TP 候选；
5. 候选数量不足时，先做并行策略和卡数粗筛，再搜索 backend、prefill 等参数；
6. 每个部署方案必须使用相同 workload 和 SLA 进行比较。

### 评价指标

TP 方案至少应比较：

- 总 token throughput；
- 每卡 token throughput；
- TTFT 和 TPOT（包括尾延迟，如果测试方法提供）；
- 显存占用和 KV Cache 容量；
- KV Pool 预留大小和目标并发下的可用容量；
- GPU 卡数、TP/PP 划分和卡间互联类型；
- 服务启动、OOM、通信和调度失败情况。

### 证据等级

- 没有显存或实测证据：只能作为候选或启发式约束；
- 已确认模型切分/显存无法满足：可以作为 hard constraint 排除；
- 已完成相同 workload 的实测：可以作为 measured recipe 记录，并注明镜像、模型、硬件和版本。

## 上下文长度不是本次请求长度

WorkloadSpec 的输入/输出长度只描述本轮 benchmark 发送的请求。例如 32K 输入和
1K 输出表示测试请求长度，不代表服务的 `context_length` 应该被设置成 33792。

如果 ModelSpec 提供 `native_context_length`，SearchPlan 默认应保留该模型能力，或
明确让 SGLang 继承模型配置。只有 JobSpec 显式指定 serving cap，或者已有实测证据
证明目标硬件/运行时无法承载时，才允许缩短服务上下文，并说明依据。
