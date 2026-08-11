---
name: parallelism-search
description: Generate bounded LLM serving deployment candidates by total GPU budget and valid TP/PP factorization.
---

# Parallelism Search

用于在固定单机 GPU 预算下生成模型服务的并行部署候选。

## Inputs

读取：

- `HardwareSpec`：GPU 数量和互联拓扑；
- `ModelSpec`：层数、attention/KV heads、权重精度和上下文能力；
- `WorkloadSpec`：请求长度、并发和 SLA；
- `references/parallelism/*.md`；
- 目标 SGLang 镜像的 `--help` 和已有启动日志（如果提供）。

## Required reasoning

1. 估计或读取每个候选的权重、运行时和 KV Pool 约束；
2. 从较少 GPU 数量开始寻找满足可启动和 SLA 的候选；
3. 对每个 `gpu_count` 只生成满足 `gpu_count = tp_size * pp_size` 的方案；
4. PCIe 多卡优先评估 PP，NVLink 多卡重点评估 TP；
5. 区分 `official`、`measured` 和 `heuristic` 证据；
6. 不因为机器有 N 张卡就默认使用 TP=N；
7. 选择并行方案后，再生成 backend、prefill 和调度参数搜索空间。

## Output contract

SearchPlan 使用：

```json
{
  "parallelism_candidates": [
    {
      "gpu_count": 4,
      "tp_size": 2,
      "pp_size": 2,
      "source": "...",
      "reason": "...",
      "evidence_level": "heuristic"
    }
  ]
}
```

禁止把 `tp_size`、`pp_size` 作为独立 search axes 与其他参数做无约束笛卡尔积。
