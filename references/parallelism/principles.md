# 并行策略原则

并行策略不是 `tp_size` 和 `pp_size` 的独立参数笛卡尔积，而是一个完整部署方案。

## 核心约束

对单机部署，每个方案必须满足：

```text
gpu_count = tp_size × pp_size
gpu_count <= HardwareSpec.gpu.count
```

先确定候选总卡数，再生成该卡数的 TP/PP 因子组合。例如 8 卡只生成：

```text
TP=8, PP=1
TP=4, PP=2
TP=2, PP=4
TP=1, PP=8
```

不能把 `TP=1, PP=1` 当作 8 卡部署方案。

## 搜索顺序

1. 根据权重、运行时开销、目标上下文、KV Pool 和 SLA 估计最小可行卡数；
2. 为每个卡数生成合法的 TP/PP 因子组合；
3. 结合 PCIe/NVLink 对候选排序；
4. 固定部署方案后，再搜索 attention backend、prefill 和调度参数。

机器有更多 GPU 只代表可用预算更大，不代表必须全部使用。
