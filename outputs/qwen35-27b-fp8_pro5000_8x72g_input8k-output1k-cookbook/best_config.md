# 最佳启动参数报告

- 任务：`qwen35-27b-fp8_pro5000_8x72g_input8k-output1k-cookbook`
- 模型：`Qwen3.5-27B`
- GPU：`G24_pro5000` × `8`（单卡 `72 GB`）
- 镜像：`hai-beijing.tencentcloudcr.com/ai/sglang:v0.5.16-cu129`
- 负载：`W10_input-8k-output-1k`
- 排名状态：`FINAL`

## 结论

没有找到同时满足 SLA 和输出健康检查的候选配置，不能推荐可上线参数。

## 本次 SLA

- 平均 TTFT：≤ `2,000.00` ms
- 平均 TPOT：≤ `80.00` ms
- 成功率：≥ `99.00%`

## 未通过候选的实测点

| 候选 | 并发 | 平均 TTFT | 平均 TPOT | 成功率 | 平均输出 tokens | 状态 |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 1 | 2,200.85 ms | 55.32 ms | 100.00% | 1,024 | ok |

## 候选排名对比

| 排名 | 候选 | TP | 实例/机 | 最佳并发 | goodput/host | TTFT | TPOT | 成功率 | 状态 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | baseline | 1 | 8 | None | 0.00 | 0.00 ms | 0.00 ms | 0.00% | completed |

## 未通过候选

- `baseline`：没有找到满足 SLA 的并发点

## 报告来源

- 排名：`/home/ubuntu/llm-infer-tuner/outputs/qwen35-27b-fp8_pro5000_8x72g_input8k-output1k-cookbook/results/ranking.json`
- 候选结果：`/home/ubuntu/llm-infer-tuner/outputs/qwen35-27b-fp8_pro5000_8x72g_input8k-output1k-cookbook/results/candidate_results.jsonl`
- 本报告由确定性脚本生成，没有调用 AI，也不会修改原始排名文件。
