# SGLang 压测(客户端)知识库

> 这是 **AI 生成压测命令时要读的唯一经验文件**。SKILL.md 是流程入口(读什么、按什么步骤),
> 本文件是「逻辑与判据」——每条经验带出处等级,你据此拼分并发命令、定并发梯度、判哪档数据作废。
>
> 出处等级(越靠后越该被质疑):
> `official`(官方文档/源码原文) · `source`(读 sglang 源码/方法定义) · `measured`(实测,标硬件) · `judgment`(人的判断,最该质疑)
>
> **加新经验 = 改本文件,不改任何代码。**

---

## 0. 最高红线:压测命令绝不写 `--context-length`

`source`(读 sglang 源码坐实)+ 团队定论。

**压测复刻生产**——生产不写 `--context-length`,压测就一字不写,回落模型 config 默认。

**机制精简版**(完整机制见 server 侧 `sglang-server-config-gen/knowledge.md §0`):KV 是「按剩余显存整体切的共享分页池」(`max_total_num_tokens`),`context_length` 只当单请求长度上限,**不按 context 给每个请求预留 KV**。所以写不写对显存/并发槽/吞吐**没有区别**;写贴 workload 的人造紧值(如 4608)只会制造乐观 SLA 假象。

**推论**:压测命令里不出现 `--context-length` 这个 flag(它属于**服务端**启动参数,本就不在 bench_serving 的 `argument_mapping` 里)。压测侧要做的只是「读值」——从服务端启动日志读回落 context(qwen3.6 = 262144)和 `max_total_num_tokens` 备查,不在压测命令里「写值」。

---

## 1. `--backend` 恒为 `sglang`(不许改)

`measured`(团队两条独立记录都踩过)。

**压测必须 `--backend sglang`**,走服务端 `/generate` 原生端点。

**踩坑机制**:thinking 类模型走 `/v1/chat/completions`(OpenAI 兼容端点)时,流式 `delta.content` 可能为空(推理内容走 `reasoning_content` 或被吞),bench 按 `content` 统计 token 会把 **decode 吞吐算成 0**,整档数据废掉且看不出原因。

- `--backend sglang` 是 benchmark_method 的 `fixed_args.backend`,**固定值**,任何 workload/模型都不改。
- 这也是数据体检的一部分:见 §5——若发现某档 output token 统计异常为 0,先查是不是 backend 走错端点。

---

## 2. 并发档怎么选(直接决定结论对不对)— `judgment`

(迁移自 server 侧 knowledge.md §9)

- **第一轮用贴近 SLA 边界的并发,不要拍一个数。** 太高全 fail SLA(白跑),太低测不出配置差异。
- 有同卡同场景历史数据取「已知刚好过 SLA」那档;没有则先 C=1、C=2 各跑一次基线定边界。
- 第二轮只对前 3-5 名扫并发梯度,二分找各自 SLA 饱和点,在饱和点比 goodput(不同配置饱和点可能不同)。
- 前 3 名各重跑 2-3 次算标准差(差异小于噪声的并列不硬排名次)。
- 梯度取值:优先用 workload 卡的 `traffic.values`;缺则用 benchmark_method 的 `traffic.concurrency_values`(默认 `[1,2,4,8,16,32]`)。

---

## 3. num_prompts 推导 — `source`

(读 benchmark_method json 的 `derived_args` / `traffic` 坐实)

**每档 `num_prompts = concurrency × num_prompts_multiplier`**(multiplier 默认 4,取自 benchmark_method 的 `traffic.num_prompts_multiplier`)。

- **为什么必须等比**:每档请求量随并发同比放大,每档才跑「够久、够满」才能反映该并发下的稳态吞吐;若所有档固定同一 `num_prompts`(比如都 128),低并发档请求太多跑很久、高并发档几秒跑完,**两档不可比**。
- 例(qa-chat multiplier=4):并发 `[1,2,4,8,16,32]` → num_prompts `[4,8,16,32,64,128]`。
- 别把 multiplier 写死在脑子里——从 json 读,不同 method/workload 可能不同。

---

## 4. 目标函数 goodput — `judgment`

(迁移自 server 侧 knowledge.md §10)

```
goodput(cfg) = max_C  total_token_throughput(cfg, C)
               s.t.  Pxx_TTFT(cfg,C) ≤ sla.max_avg_ttft_ms
                 AND Pxx_TPOT(cfg,C) ≤ sla.max_avg_tpot_ms
                 AND success_rate     ≥ sla.min_success_rate
                 AND avg_output_tokens ≈ output_len  (未被截断,见 §5)
```

**不能用裸吞吐排序**——峰值吞吐那档几乎必然违反 SLA,拿它当最优上线就翻车。percentile(Pxx)从 job 的 SLA 读,**不写死**(前人无共识:vLLM P99 / Vidur P90+P99 / BBuf P50)。本 skill 只出命令;goodput 的计算与排序是下游执行器的事,但命令必须覆盖梯度、让下游有数据可算。

---

## 5. 数据体检(不过则该档数据作废)— `measured`

(迁移自 server 侧 knowledge.md §11)

1. **backend 必须 `sglang`**(见 §1):走 `/v1/chat/completions` 会把 thinking 模型 decode 吞吐算成 0。
2. **`avg_output_tokens ≥ output_len × 0.9`**:高并发下网关超时/并发限制会提前截断,截断的档位吞吐虚高,**数据作废**。
3. **丢弃第一条**:首次运行有 kernel 编译尖峰(warmup),会让第一个候选看起来极差——丢第一条或单独 warmup。
4. **高并发网关超时提前终止的档作废**:与第 2 条同源,任何被提前终止/请求数不足的档不进 goodput 比较。

---

## 6. 命令拼装铁律 — `source`

(读 benchmark_method json 的 `argument_mapping` / `notes` 坐实)

- **严格照 benchmark_method json 的 `argument_mapping` 落 flag**——一个映射项对一个 flag,mapping 里没有的 flag **一律不生成**(别发明)。当前 `sglang_bench_serving.json` 落地的 flag 集:`--backend` `--dataset-name` `--random-range-ratio` `--host` `--port` `--model` `--dataset-path` `--max-concurrency` `--random-input-len` `--random-output-len` `--output-file` `--num-prompts`。
- **`--random-range-ratio 1.0`**(fixed_args):固定输入长度(range ratio=1 → 输入长度不抖动,每条都是 `--random-input-len` 的精确值),使各档可比。
- **`--dataset-name random-ids`**(fixed_args):纯随机 token id 合成数据集,**完全离线、不依赖外部语料**。⚠️ 注意 `random`(不带 `-ids`)在 sglang ≥0.5.16 已改为「从 ShareGPT 采样 token 再截断」,会触发 `hf_hub_download` 去 HuggingFace 下 ShareGPT 语料——离线/内网环境必崩(`RuntimeError: Cannot send a request, as the client has been closed`),且不写 `--output-file`,报告为空。故本项目一律用 `random-ids`。配 `--random-range-ratio 1.0` + 固定 seed(默认 42),各候选拿到完全相同 workload,goodput 横向可比。
- **续行反斜杠后不能有空格**(notes 第三条):若把长命令折多行,`\` 必须是行尾最后一个字符,后面一个空格就断行失败。**首选给单行命令**规避此坑。
- **`--context-length` 不在 mapping 里**——它是服务端 flag,压测命令里本就不该出现(呼应 §0)。
- host/port/model/dataset_path 是 `runtime_args`,用占位 `${BENCHMARK_HOST}` / `${BENCHMARK_PORT}` / `${MODEL_PATH}` / `${DATASET_PATH}`,运行期注入。
