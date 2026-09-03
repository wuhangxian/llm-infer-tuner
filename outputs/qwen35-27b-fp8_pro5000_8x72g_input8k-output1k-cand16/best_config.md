# 最佳启动参数报告

- 任务：`qwen35-27b-fp8_pro5000_8x72g_input4k-output1k-cand16`
- 模型：`M41_qwen3-5-27b`
- GPU：`G24_pro5000` × `8`（单卡 `72 GB`）
- 镜像：`I03_sglang-v0.5.16`
- 负载：`W10_input-8k-output-1k`
- 排名状态：`PROVISIONAL`

## 结论

推荐候选 **`c011`**（排名第 `1`）：在满足 SLA 的候选中，它的整机 goodput 最高，为 **1,629.85**。

## 最终启动参数

本次候选启动命令：

```bash
python -m sglang.launch_server --tp-size 4 --attention-backend flashinfer --chunked-prefill-size 4096 --mem-fraction-static 0.78 --disable-radix-cache --reasoning-parser qwen3 --tool-call-parser qwen25 --model-path ${MODEL_PATH} --host 0.0.0.0 --port 30000
```

> 说明：执行器会替换 `${MODEL_PATH}` 和端口，并强制加入 `--disable-radix-cache`。
> 参数差异中的“未显式指定”表示该参数没有传给 `launch_server`，实际测试采用 SGLang/镜像默认行为；报告不会擅自猜测默认值。

生效参数：

```json
{
  "attention_backend": "flashinfer",
  "chunked_prefill_size": 4096,
  "disable_radix_cache": true,
  "mem_fraction_static": 0.78,
  "reasoning_parser": "qwen3",
  "tool_call_parser": "qwen25",
  "tp_size": 4
}
```

- TP 大小：`4`
- 每台机器实例数：`2`
- 最佳并发：`2`

## 实测指标与 SLA

| 指标 | 最佳实测 | 目标 | 结果 |
|---|---:|---:|---|
| goodput / host | 1,629.85 | 越高越好 | 排名依据 |
| 平均 TTFT | 1,896.25 ms | ≤ 2,000.00 ms | 通过 |
| 平均 TPOT | 20.25 ms | ≤ 80.00 ms | 通过 |
| 成功率 | 100.00% | ≥ 99.00% | 通过 |
| 平均输出 tokens | 1,024 | ≥ 922 | 通过 |

## 为什么它比其他候选好

- 相比第二名 `c003`，goodput / host 高 `4.11`（`0.25%`）。
- 与第二名的有效参数差异：
  - `mem_fraction_static`：最佳=0.78；对比=0.86
- 相比 baseline，goodput / host 高 `1,629.85` （`—`）。
- 排名使用的是 `goodput_per_host`，即候选在 SLA 约束下的单实例吞吐，再结合 TP 大小折算整机可放置的实例数（当前最佳为 `2.0` 个）。
- 该候选生成时记录的配置依据：
  - 中影响 mem-fraction 轴(下界): mem=0.78 最保守
  - 并发上限最低但启动最稳,作为对照下界

## 候选排名对比

| 排名 | 候选 | TP | 实例/机 | 最佳并发 | goodput/host | TTFT | TPOT | 成功率 | 状态 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **c011** | 4 | 2 | 2 | 1,629.85 | 1,896.25 ms | 20.25 ms | 100.00% | completed |
| 2 | c003 | 4 | 2 | 2 | 1,625.74 | 1,894.00 ms | 20.31 ms | 100.00% | completed |
| 3 | c016 | 4 | 2 | 2 | 1,619.68 | 1,922.68 ms | 20.36 ms | 100.00% | completed |
| 4 | c010 | 4 | 2 | 2 | 1,617.90 | 1,912.41 ms | 20.40 ms | 100.00% | completed |
| 5 | c009 | 4 | 2 | 2 | 1,472.79 | 1,990.75 ms | 22.51 ms | 100.00% | completed |
| 6 | c012 | 2 | 4 | 1 | 1,149.95 | 1,363.52 ms | 30.00 ms | 100.00% | completed |
| 7 | c002 | 2 | 4 | 1 | 1,124.82 | 1,383.91 ms | 30.67 ms | 100.00% | completed |
| 8 | c008 | 2 | 4 | 1 | 1,081.73 | 1,421.02 ms | 31.91 ms | 100.00% | completed |
| 9 | c015 | 4 | 2 | 1 | 907.67 | 1,036.83 ms | 18.83 ms | 100.00% | completed |
| 10 | c004 | 8 | 1 | 2 | 863.36 | 1,989.74 ms | 18.92 ms | 100.00% | completed |

## 报告来源

- 排名：`/home/ubuntu/llm-infer-tuner/outputs/qwen35-27b-fp8_pro5000_8x72g_input8k-output1k-cand16/results/ranking.json`
- 候选结果：`/home/ubuntu/llm-infer-tuner/outputs/qwen35-27b-fp8_pro5000_8x72g_input8k-output1k-cand16/results/candidate_results.jsonl`
- 本报告由确定性脚本生成，没有调用 AI，也不会修改原始排名文件。
