# SGLang 参数寻优规则索引

这里不再放一整本混杂的经验文档，而是规则库的入口。具体知识按主题拆在
`references/rules/*.yaml`，文件中的 `rules` 数组可以一条一条追加。

## 读取工作流

1. 读取本 skill 的 `SKILL.md`，确定输入、输出和候选生成步骤。
2. 读取项目根下 `catalogs/gpu.yaml`、`models.yaml`、`workloads.yaml` 和
   `sglang-images.yaml`，拿到这次 JobSpec 对应的事实卡片。
3. 读取 `references/rules/README.md`，理解规则字段和状态。
4. 根据 JobSpec 的 GPU、模型、镜像、负载读取下表中相关主题的规则；默认先读
   `active` 规则，`experimental` 规则可以作为带风险的探索候选，`deprecated` 不主动采用。
5. 候选的 `reasons` 引用命中的规则 ID；如果规则与当前模型或镜像不匹配，不要强行套用。

## 主题文件

| 文件 | 主要问题 |
|---|---|
| `references/rules/attention.yaml` | attention 后端、SM/CUDA 门槛、page size 联动 |
| `references/rules/parallelism.yaml` | TP/PP/EP、量化整除、精度准入 |
| `references/rules/memory.yaml` | context、KV cache、显存比例、hybrid mamba 状态池 |
| `references/rules/speculative.yaml` | 模型与镜像共同决定投机解码算法和参数 |
| `references/rules/scheduling.yaml` | 调度策略、搜索轴、候选预算、SLA/goodput |
| `references/rules/fairness.yaml` | 公平压测、预热、输出健康、证据分层 |

## 新增规则原则

- 一个主题一个文件，文件内一条条追加；不要再把新经验塞回本文件。
- 每条规则都必须有全局唯一 `id`、适用范围 `applies_when`、可执行的 `guidance`、
  证据 `evidence` 和状态 `status`。
- `official`/`source`/`measured`/`judgment` 只表示证据来源，不把人的判断伪装成源码事实。
- 真机失败可以记录为 `measured` 或把规则标成 `experimental`，但不因为规则不确定就
  把所有候选在生成阶段拦掉；执行器实测才是最终裁决。
- 规则只约束“如何选择和解释候选”，不承担模型权重、镜像和机器事实；事实放 catalogs。

规则文件可用 `uv run python scripts/validate_knowledge.py` 检查格式、重复 ID 和证据字段。
