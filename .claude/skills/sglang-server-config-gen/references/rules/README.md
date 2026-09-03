# 可扩展规则库

这里是 sglang-server-config-gen 的可维护知识库。一个主题一个 YAML；每个 YAML 的
rules 数组里一条条追加规则。knowledge.md 只负责目录和工作流，不再承载具体经验。

## 规则格式

示例：

schema_version: 1
topic: attention
description: 这个文件解决什么问题
rules:
  - id: attention.example
    title: 简短标题
    status: active          # active / experimental / deprecated
    confidence: high        # high / medium / low
    applies_when:
      engine: sglang
    guidance: |
      给配置生成器的可执行建议。
    evidence:
      kind: source           # official / source / measured / judgment
      ref: "源码、文档或实测记录"
      verified_with: "sglang-v0.5.16"
    tags: [attention, startup]

字段含义：

- id 必须全局唯一，建议用 主题.对象.结论 命名；以后只追加，不复用旧 ID。
- applies_when 描述适用范围；暂时无法结构化的条件写在 guidance，不要猜测。
- evidence 让每条规则可追溯；没有证据的判断标成 judgment 或 experimental。
- status: deprecated 表示保留历史记录，但生成候选时不要主动采用。

## 新增一条规则的工作流

1. 先确认主题：attention、parallelism、memory、speculative、scheduling、fairness。
2. 在对应 YAML 的 rules 末尾新增一条，填写唯一 ID、适用条件、建议、依据和版本。
3. 运行 uv run python scripts/validate_knowledge.py，只检查知识文件格式和可追溯性。
4. 运行 ./gen_configs.sh <job.json> 观察候选是否引用了新规则；不要求规则预先保证每个候选都能启动。
5. 真机结果写入规则的 evidence 或新增一条 measured 规则，不覆盖原始官方/源码事实。

规则校验只保护知识库不损坏，不是候选配置的硬闸；experimental 规则仍可进入搜索，最终由执行器实测决定。

