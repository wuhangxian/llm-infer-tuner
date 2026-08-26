# SGLang 参数寻优知识库

> 这是 **AI 生成启动配置时要读的唯一经验文件**。SKILL.md 是流程入口(读什么、按什么步骤),
> 本文件是「逻辑与判据」——每条经验带出处等级,你据此派生候选、决定哪些轴铺哪些档、哪些必崩要排。
>
> 出处等级(越靠后越该被质疑):
> `official`(官方文档/源码原文) · `source`(读 sglang gate 条件) · `measured`(实测,标硬件) · `judgment`(人的判断,最该质疑)
>
> **加新经验 = 改本文件,不改任何代码。**

---

## 0. 最高红线:绝不写 `--context-length`

`source`(读 sglang 0.5.13 源码坐实)+ 团队定论。

**压测和上线都一字不写 `--context-length`**,回落模型 config 默认(qwen3.6 = native `262144`,不是 max 1M——1M 要手动开 YaRN + 显式设值 + override 环境变量)。

**机制**:KV 是一个「按剩余显存整体切的共享分页池」(`max_total_num_tokens`),`context_length` 只当单请求长度上限,**根本不按 context 给每个请求预留 KV**。所以写紧值 vs 回落默认,对显存/并发槽/吞吐**没有区别**。一个 3.5k 请求就只占 3.5k slot。

**推论**:不写 → ①不会启动 OOM ②不会并发槽变少 ③候选天然可比(默认值是模型 config 常数、对同 job 全体候选相同,不需靠 pin 保证可比)。

**这条推翻了旧 skill 的 bug**:旧 gen_spec 把「一条测试 case 的 input+output 向上取整」当成 context_length 并 pin,那是把测试负载误当服务上下文窗口,制造乐观 SLA 假象。**绝不写贴 workload 的人造紧值(如 4608)。** 旧规则里 `context_length_explicit`/`ctx_covers_workload` 两条硬约束**已废弃,不要再生成**。

**唯一要做的是「读值」不是「写值」**:每个 job 读一次启动日志,把实际回落 context(262144)和 `max_total_num_tokens` 记进结果备查。

**唯一例外**(需用户明确拍板):若生产给 context 设了业务上限 cap,压测才写生产那个精确值。本 case 生产不写 → 压测不写。

---

## 1. 按算力查 attention 后端(SM120 / Pro5000 的核心)

`source`(读 v0.5.10 源码)。这是「引擎相关」的算力知识,故在 skill 内、不在 catalogs/gpu.yaml。

**最终 attention 候选 = 本短名单[sm_major] ∩ 镜像菜单(sglang-images.yaml.attention_backends)∩ CUDA 达标。**
三者取交才是真正能上的 attention 轴。短名单是「阶段中性」的一份,`--prefill-attention-backend` / `--decode-attention-backend` 两轴共用它,所以**只放两侧都能起的后端**。

| sm_major | 架构 | 短名单 | 备注 |
|---|---|---|---|
| 7 | Volta/Turing | `[triton]` | flashinfer 覆盖不全(V100 不支持),只稳 triton |
| 8 | Ampere/Ada | `[flashinfer, triton]` | flashinfer 首选,triton 兜底 |
| 9 | Hopper | `[fa3(min_cuda 12.3), flashinfer, triton]` | fa3 需 CUDA≥12.3 |
| 10 | SM100(B200) | `[trtllm_mha(min_cuda 12.8), flashinfer, triton]` | trtllm_mha 两侧都起 |
| **12** | **SM120(Pro5000/50xx)** | **`[flashinfer, triton]`** | 见下方 SM120 专段 |

**SM120 专段(Pro5000 必读):**
- **fa3 排除**:fa3 在 SM120 硬失败(attention_registry gate 只放 major∈{8 非 MLA, 9})。所以 Pro5000 上 attention 轴**只有 flashinfer / triton**,没有 fa3。
- **trtllm_mha 必须分 prefill/decode 两侧看,别混**:
  - `prefill` **起不来**:`server_args.py:2373-2375` `if prefill_backend=="trtllm_mha" and not is_sm100_supported(): raise`。SM120≠SM100 → 启动即 ValueError。**永远别给 SM120 生成 prefill=trtllm_mha 的候选。**
  - `decode` **技术上能起**:门是 `is_sm90_supported() or is_sm100_supported() or is_sm120_supported()`,SM120 过关,走 XQA 内核。但短名单是阶段中性的、暂不放它;是否把 decode=trtllm_mha 加为候选,**待上机实测**吞吐/稳定性后再定。
  - auto-select 不会选它(`:2224` 原话 "trtllm_mha does not support SM120, which will fall back to flashinfer")。
- 该 prefill/decode 门 v0.5.10 ↔ v0.5.13 `git diff` 为空(两版一致,**不是** 0.5.13 后才放行)。

**换卡自动适配**:H200(sm9)attention 轴变 `[fa3, flashinfer, triton]`;这全靠查表,不写 `if sm==12` 死分支。

---

## 2. 并行度推导(TP / PP / EP)

### TP 推导 — `judgment`
候选 TP = 所有满足「**单卡权重 ≤ 单卡显存**」**且**「**块量化整除**」的 2 的幂,上限 `gpu_count`。

- **准入门槛只看「权重放得下」**(`weight_gb[precision] / tp ≤ memory_gb`)。KV 余量**不参与准入删除**——「并发多高、要不要逐出」交压测 + SLA 兜底。
- **块量化 × TP 整除硬约束(`measured` v0.5.10,不满足直接不生成该 TP 候选)**

  **一句话**:权重按「块」量化时,TP 把权重矩阵切开,**每卡分到的分片维度必须刚好是整数个量化块**;切进半块,SGLang 在**加载权重、创建量化张量的那一刻**就抛 `ValueError: output_size ... not divisible by block_n` 并对子进程 sigquit——**服务在启动早期就崩,`/health` 从不返回,压根进不到压测**。这不是显存不够(那是压测期 OOM),是维度切错。

  **① 什么量化方式会触发(受影响 vs 不受影响):**
  | 量化方式 | 是否受此约束 | 原因 |
  |---|---|---|
  | **块量化 / 细粒度**(`scheme: fine-grained`,FP8 block-wise,`block_size=128×128`,Qwen/DeepSeek 官方 FP8) | ✅ **受限** | 每块共享一个 scale,分片必须含完整块 |
  | **组量化**(AWQ / GPTQ,`group_size=128`) | ✅ **受限**(同一堵墙,非 FP8 独有) | 与块量化同理 |
  | **per-tensor FP8**(整张量一个 scale) | ❌ 不受限 | 无「块」概念,任意维度可切;代价:精度差一截 |
  | **per-channel FP8**(每输出通道一个 scale) | ❌ 不受限 | 等效块=1 |
  | **不量化 / bf16 / fp16** | ❌ 不受限 | 无量化块 |
  → **判据**:只看 `catalogs/models.yaml` 卡片 `quantization.scheme` 是否为 `fine-grained`(或 block/group 类)。**不是「用了 FP8 就受限」,是「用了块/组量化才受限」。**

  **② 什么模型结构容易撞墙:**
  - **细粒度 MoE 最危险**:专家多、每专家小,`moe_intermediate_size` 常只有几百(本模型 =512)。被切分的是专家 FFN 的 gate/up 输出维度 = `moe_intermediate_size / tp`,须 `% block_size == 0` → **`tp_max = moe_intermediate_size / block_size`**。
  - **dense 模型基本不撞**:`intermediate_size` 动辄上万(如 27B=17408),`17408/128=136`,切到 tp=8 每卡 2176 仍是 128 整数倍,远不触顶。
  - 一句话记:**「小专家 + 块量化」= TP 天花板很低**,与卡数无关。

  **③ 具体排除哪些启动配置(生成期就删,不留给压测):**
  - 计算 `tp_max = floor(moe_intermediate_size / block_size)`(dense 用 `intermediate_size`)。
  - **只保留 `tp ∈ {2 的幂} 且 tp ≤ tp_max 且 moe_intermediate_size % (block_size × tp) == 0`** 的候选;其余 TP **直接不生成**(不是降优先级,是删)。
  - 连带:该 TP 下所有 `--attention-backend`、`--mem-fraction-static` 等组合**一并不生成**(它们再合法也起不来,是 TP 先崩)。避免像本次那样白跑 4 次启动(c001–c004 四条 tp=8 组合全崩在同一个错)。
  - **实例**:M02_qwen36-35b-a3b-fp8,`moe_intermediate_size=512`、`block_size=128` → `tp_max=4`。**合法 TP={1,2,4}**(每卡 512/256/128,整除);**排除 TP=8**(64<128)。故 8 卡机上此模型最多切 4 路,`gpu_count=8` **不等于** tp 可取 8。

  **④ 卡片缺字段时的保守兜底**:`moe_intermediate_size`(或 `intermediate_size`)或 `quantization.block_size` 任一缺失 → **不得假设可切到 `tp=gpu_count`**;按已知能起的最小安全档(如 tp≤4)生成,并在候选 `reasons` 明确标注「未核对块量化整除性,TP 上限存疑,需上机核 config.json」。**宁可少生成,不生成注定崩的候选。**

  **⑤ 不要为跑满卡去改量化方式**:换 per-tensor 能绕过约束但掉精度、破坏候选可比性;且本模型实测 tp=4 已是吞吐最优(2346 tok/s,tp 越小越低),tp=8 即便能起也因 all-reduce 更重而更慢。**正解永远是砍掉非法 TP,不是改 checkpoint。**
- KV 估算(`kv_gb_per_token`)仍算、仍进 notes,但**只作候选排序/展示**,不删任何能放下权重的 TP。真值以启动日志 `max_total_num_tokens` 为准。
- **KV 余量排序公式**(`judgment`,**仅排序/展示,不删 TP**):
  ```
  kv_headroom_gb(tp) = max(min_gb, input_len × target_concurrency × kv_gb_per_token / tp)
  ```
  - `min_gb = 8`(地板,估算再小也按 8G 保留余量意识)。
  - `kv_gb_per_token`:优先取 model 卡的值;model 卡没有时用全局兜底 `kv_gb_per_token_default = 0.00012`(稠密标准注意力量级,混合 GDN 会显著更小)。
  - `target_concurrency`:优先取 workload 的目标并发;没有时用 `target_concurrency_default = 16`。
  - `input_len`:取 workload 的输入长度(input+可忽略的 output)。
  - 用途:同样放得下权重的多个 TP,按 headroom 估算从大到小提示优先级(headroom 越大越抗高并发);**绝不因 headroom 小而删 TP**,删只走本节的权重门槛(单卡权重 ≤ 显存),降序只走下方长输入规则。真值仍以启动日志 `max_total_num_tokens` 为准。
- 例:M01_qwen36-27b-fp8(27G)在 72G 卡:TP1 权重 27✓ / TP2 13.5✓ / TP4 6.8✓ / TP8 3.4✓ → 可行 TP `[1,2,4,8]`,**默认全保留**。
- **排序提示(不删候选)**:
  - 输入超 32768 时把 TP1 排到末尾(`measured`:本地 Pro5000 Qwen3.5-27B 64k 输入下 TP1 吞吐从并发 1 到 32 完全平坦≈778 tok/s。但证据模型≠目标模型,外推不确定 → 只降序不删)。
  - 无 NVLink 的卡 TP 通信走 PCIe,开销高于 NVLink → 只影响排序,不排除任何 TP。
- **不因为机器有 N 张卡就默认 TP=N。**

### PP 推导 — `judgment`
`--pipeline-parallel-size > 1` **排除**:单机流水线气泡,严格劣于纯 TP/EP。所以单机场景 `pp_size` 恒 = 1,`gpu_count = tp_size × pp_size` 退化为 `gpu_count = tp_size`。

### EP(专家并行)推导 — `source`
**仅 MoE 模型生成此轴**(`arch` 含 "moe" 或有 `num_experts`)。dense 模型(如 qwen36-27b `dense_hybrid_gdn`)**不生成 ep 轴**——给了也非法。

- 候选:`ep ∈ {1} ∪ { tp 的 2 的幂因子 d | num_experts % d == 0 }`。ep=1 恒为候选(等价「不开 EP」的基线,必须能与 ep>1 对照)。
- **两条硬约束**(不满足就起不来,`source` v0.5.10 坐实,非法组合直接不生成):
  - `tp_size % ep_size == 0`(`parallel_state.py:1887` moe_tp_size = tp//ep//moe_dp)
  - `num_experts % ep_size == 0`(`expert_location.py:204` assert num_physical_experts % ep == 0)
- 例:tp=8、num_experts=256 → ep ∈ {1,2,4,8}(256=2^8,全整除)。
- 生成期**只保证「起得起来」**;ep-vs-tp 谁吞吐高、某 ep 是否 OOM,交④实测。SM120 上 moe_runner_backend 保持 auto,ep>1 起得来,故 ep 本身在 SM120 可启动,不作生成期硬约束。

### 量化诚实(权重精度准入)— `judgment` + `source`
不是每种精度都能端到端真跑通。**只在「端到端诚实集」内出候选,越界的精度要么拒绝、要么降级并标注**,绝不假装某精度能跑而生成注定失败/静默降级的候选。

- **端到端诚实集 = `{none, bf16, fp8}`**:这几种在本项目目标环境(NVIDIA + SGLang)上是真跑得通的权重精度。
- **`nvfp4`(fp4)有算力门槛**:仅当 `gpu_model 查表得 sm_major ≥ model.nvfp4_requires_sm`(qwen3.6 系列 = 10,即 B200/B300 SM100+)才允许。**SM120(Pro5000)< 不满足** → 生成期直接**拒绝** nvfp4 候选,不降级(降级会偷改精度、破坏可比性)。
- **其它未列入诚实集的格式**(如 gptq/awq/int4 等本项目未验证的):**降级到 `fp8`(模型有 fp8 权重)或 `bf16`,并在候选 `reasons` 里明确标注「原请求精度 X 未验证,已降级到 Y」**,让下游知道这不是用户原意。
- 精度决定用哪份 `weight_gb[precision]` 去过本节 TP 推导的权重门槛(单卡权重 ≤ 显存)—— 降级后必须用降级后精度的 weight_gb 重算准入。
- 出处:诚实集来自旧 sglang-param-search 的实测边界(`_QUANT_END_TO_END`);nvfp4 算力门来自 cookbook「NVFP4 仅 B200/B300」并由 `model.nvfp4_requires_sm` 承载(见 catalogs/models.yaml)。

---

## 3. 搜索空间(各轴铺哪些档)

`job` 没显式给同名字段时用这些默认;显式给了则覆盖。

| 轴 | 默认档 | 出处 | 说明 |
|---|---|---|---|
| `--mem-fraction-static` | `[0.80, 0.84, 0.88, 0.92]` | judgment | 上界 0.92 来自 compute-mamba-ratio skill 建议 |
| `--chunked-prefill-size` | `[4096, 8192, 16384]` | official | 官方长上下文建议从 4K 起逐步加大;必须 `% page_size == 0` |
| `--max-running-requests` | `[8, 16, 32]` | judgment | 适用长输入(60k+);BBuf 的 [64,96,128] 是 8k dataset,不可直接迁 |
| `--kv-cache-dtype` | `[auto, fp8_e4m3]` | measured | 长输入 KV 是主瓶颈,fp8 腰斩 kv/token;需配精度验证 |
| `--schedule-conservativeness` | `[0.3, 1.0, 1.3]` | official | 官方唯一给出闭环判据的调度参数(见 §6) |
| `--page-size` | `[1, 32, 64]` | official | KV cache 分页大小,调大减少页表管理开销(尤其 MoE 高并发)。**按 attention 后端联动生成,非独立铺**:flashinfer/triton → 只生成 [1](调大无收益,SGLang 不强制改写);mamba `no_buffer` → 只生成 [1](`no_buffer`+radix cache 强制 page_size=1);mamba `extra_buffer` → 只生成 [64](`FLA_CHUNK_SIZE(64) % page_size == 0` 硬约束);trtllm_mha → [16,32,64](但 SM120 用不了 trtllm_mha)。cookbook 证据:Qwen3-Coder MoE 推荐 32,DeepSeek-V3.2 用 64/128。⚠️ **版本核对**:换 SGLang 版本时需核对 `server_args.py` 中 `_handle_page_size` 及后端约束是否变化 |
| `--mamba-radix-cache-strategy` | 暂不搜索 | official | **TODO: 暂用 SGLang 默认值 `no_buffer`,不生成此轴的候选**。原因:① SGLang 默认 `auto`→`no_buffer`,源码注释说 extra_buffer "needs more verification";② cookbook 用的参数名 `--mamba-radix-cache-strategy` 与 SGLang CLI 实际参数名 `--mamba-scheduler-strategy` 不一致,需确认;③ extra_buffer 几乎全好处(cookbook: 非KV-bound场景严格优于no_buffer),但显存代价和 KV-bound 场景的并发下降需实测。**后续启用时**:确认 CLI 参数名 → 改成首选 extra_buffer + no_buffer 对照 → 补 `--disable-overlap-schedule` 约束(no_buffer必须配) |
| `--speculative-algorithm` | `[无, EAGLE]` | official | **仅 `capabilities.supports_mtp: true` 模型生成**。「无」是对照基线(不写这组 flag);EAGLE 从 `mtp_params` 取配套值(`num-steps`/`eagle-topk`/`num-draft-tokens`)。cookbook 5/7 Qwen 模型推荐,延迟可降 2-3 倍。放 §6 F 阶段搜,先跑通非投机基线 |

**kv-cache-dtype 按算力过滤**(`source` v0.5.10):choices=`[auto, fp8_e5m2, fp8_e4m3, bf16, fp4_e2m1]`;`fp8_*` 无 SM 门槛(CUDA 11.8+),`fp4_e2m1` 需 CUDA≥12.8 + PyTorch 2.8.0+。默认池保守取 `[auto, fp8_e4m3]`,更激进精度要搜在 job 里显式给。

---

## 4. 固定不搜的参数(pin)

- `--schedule-policy lpm` — `official`。官方原文 "If the workload has many shared prefixes, try `--schedule-policy lpm`"。**仅当 workload 有大量共享前缀时 pin**(AI-coding 同 codebase / 同 system prompt 场景符合)。**若请求前缀不共享(如 random 独立请求),用默认 fcfs,不要 pin lpm。**
- 模型专属 flag(`reasoning-parser` / `tool-call-parser` / `trust-remote-code`)从 `catalogs/models.yaml` 的 `default_flags` 原样取,别自己编 parser 名。

---

## 5. 排除项(不生成这些候选)

| 排除 | 出处 | 理由 |
|---|---|---|
| `--enable-torch-compile` | official | 官方限定「small models on small batch」;27B/35B 不属该域,CUDA graph 已覆盖 small-bs decode |
| `--pipeline-parallel-size > 1` | judgment | 单机流水线气泡,严格劣于 TP/EP |
| `--speculative-eagle-topk > 1` | source | overlap scheduler 与 trtllm_mha 都只支持 topk=1,固定为 1 |
| `--enable-mixed-chunk` + 投机解码 | source | 源码里 assert 冲突 |
| `--speculative-algorithm` + `supports_mtp: false` | official | 模型没有 MTP 权重(`mtp.safetensors`),启动直接报错 |
| `--mamba-radix-cache-strategy` + `hybrid_mamba: false` | official | 非 GDN 模型没有 mamba 层,参数被忽略,等于白跑 |
| `--mamba-radix-cache-strategy extra_buffer` + `--page-size != 64` | source | `FLA_CHUNK_SIZE(64) % page_size != 0` → 启动报错 |
| `--mamba-radix-cache-strategy no_buffer` + `--page-size != 1` | source | `no_buffer`+radix cache 强制 `page_size=1`,写了别的会被忽略或报错 |
| `--page-size` 值不在 attention 后端的允许集 | source | SGLang 不报错但自动改写并打 warning(如 flashinfer+page_size=64 不会报错但无收益;trtllm_mha+page_size=1 会被改成 64)。生成期只放后端认可的值,避免被改写后行为不可预期 |
| `--chunked-prefill-size -1` | judgment | 关闭 chunked prefill,长输入峰值激活会爆(有生产用过,非普适,可手工 A/B 一次) |
| `--tp-size N`(块量化下 `moe_intermediate_size/N` 或 `intermediate_size/N` 不是 `block_size` 整数倍) | measured | **启动即崩**(加载权重时 `ValueError: output_size not divisible by block_n`),非压测 OOM。细粒度 MoE 专家小最常撞;判据与算法见 §2「块量化 × TP 整除硬约束」。例:35B-A3B(moe_int=512/block=128)排除 tp=8 |

**关于 torch-compile 的两条辟谣**:①"官方标 out of maintenance" 是张冠李戴(那话出自 server_arguments 文档非调优文档);②"torch-compile 与 EAGLE MTP 互斥、接受率 100%→12%" 查无依据,别当剪枝规则。

---

## 6. 候选选择策略(baseline_first_bounded_product)

**前提**:执行器一次性生成所有候选并行实测排名,不做分阶段负反馈。候选选择必须均衡覆盖各维度,不按阶段顺序把重要维度推到最后。

### 6.1 维度重要性分级

| 等级 | 维度 | 影响幅度 |
|---|---|---|
| 高影响 | TP、投机解码、attention | 投机解码延迟降 2-3 倍;TP 决定通信开销和显存利用率;attention 影响吞吐 10-30% |
| 中影响 | mem-fraction、mamba 策略 | mem-fraction 撑高并发上限;mamba 策略影响 overlap 吞吐 |
| 低影响 | chunked-prefill、schedule-conservativeness、kv-cache-dtype | 影响 5-15%,有剩余名额才铺 |

### 6.2 最优候选数计算

生成候选前算各维度的独立值数，再算符卡约束后的最优候选数:

基线 = 1 条(固定)
高影响独立值 = (TP数-1) + (投机解码数-1) + (attention数-1)  # 减1因基线占了一个组合
交叉组合 = (TP数-1) × 投机解码数  # TP×投机解码交叉点(扣除基线占的)
中影响独立值 = mem-fraction数 + mamba数(仅 hybrid_mamba=true)
低影响独立值 = chunked-prefill数 + schedule数 + kv-cache数

最优候选数 = 基线 + 高影响独立值 + 交叉组合 + 中影响独立值 + 低影响独立值

**示例**(M01_qwen36-27b-fp8, SM120, hybrid_mamba=true, supports_mtp=true):
- TP=[1,2,4]=3 (TP8 块量化约束砍掉)
- 投机解码=[无,EAGLE]=2
- attention=[flashinfer,triton]=2
- mem-fraction=[0.84,0.88,0.92]=3
- mamba=[no_buffer,extra_buffer]=2
- chunked-prefill=[4096,8192,16384]=3
- schedule=[0.3,1.0,1.3]=3
- kv-cache=[auto,fp8_e4m3]=2

基线 1 + 高影响独立 (2+1+1=4) + 交叉 (2×2=4) + 中影响 (3+2=5) + 低影响 (3+3+2=8) = 22

如果 max_candidates=16 < 22,按§6.3 截断。

### 6.3 名额分配规则

**如果 JobSpec 有 baseline 字段**:基线不算在 max_candidates 里,总候选数 = max_candidates + 1。基线放第一条(id="baseline"),AI 生成 max_candidates 条候选排在后面。预览时基线显示在第一行。

**如果没有 baseline 字段**:基线占第 1 条,剩余名额(max_candidates-1)按比例分配。

| 类别 | 占比 | 填充顺序 |
|---|---|---|
| 高影响 | 60% | 先 TP(不同值各1条)→ 再投机解码(开/关×不同TP交叉)→ 再attention(triton) |
| 中影响 | 30% | mem-fraction(上下界各1)、mamba(两分支各1) |
| 低影响 | 10% | 有剩余才铺,每个维度 1-2 个值 |

### 6.4 交叉优先级

名额不够时,高影响维度的交叉组合(TP×投机解码)优先于低影响维度的单独铺设。宁可砍掉 chunked-prefill/schedule/kv-cache 的候选,也要保留 TP×投机解码的交叉组合。

### 6.5 不重复

参数完全相同的候选只留 1 条。两条候选只差一个低影响参数而高影响参数完全相同的,合并为1条(取默认值)。
---

## 7. 混合架构(mamba/GDN)专属 — `official`

qwen3.6(`hybrid_mamba: true`)适用。

**约束**:
- `no_buffer` + radix cache → 强制 page_size=1 且关 overlap scheduler
- `extra_buffer` 必须配 `--page-size 64`
- `extra_buffer` + `--disable-radix-cache` → ValueError(extra_buffer 依赖 Radix Cache 存 Mamba 状态,关了没地方存。用 `no_buffer` 替代)
- 投机解码 + `no_buffer` + `--disable-radix-cache` → **合法**(不冲突,源码无此约束)
- 投机解码 + `extra_buffer` → **合法**(但 §4 pin 了 `--disable-radix-cache`,所以 extra_buffer 不能用,投机解码候选用 `no_buffer` 即可)

**为什么两分支都要测**:extra_buffer 在非 KV-bound 场景更优;KV-bound 场景要权衡 overlap 收益 vs 并发下降。长输入正是 KV-bound,两分支必须实测对冲。但 §4 pin 了 `--disable-radix-cache` 后,extra_buffer 不可用,只保留 `no_buffer`。

**与投机解码的交互**(§3 `--speculative-algorithm`):
- 投机解码 + `no_buffer` + `--disable-radix-cache` → **合法组合**(源码 server_args.py 无冲突)。
- 投机解码 + `extra_buffer` + `--disable-radix-cache` → ValueError(§5 已排除)。
- 因此:在 §4 pin 了 `--disable-radix-cache` 的情况下,投机解码候选用 `no_buffer`,正常生成,不受限制。

**mamba ratio(`--mamba-full-memory-ratio`,默认 0.9)不该盲扫——有公式**:`r* ≈ S · token_equiv · dcp_size / L`。参考:L=64K→r≈0.31,L=128K→r≈0.16。S 由 cache strategy 决定(extra_buffer overlap 开→S=5,extra_buffer_lazy→S=4)。**strategy 和 ratio 不正交,别当独立笛卡尔积扫。** L≈60k 时 r*≈0.31,盲扫 0.5/0.9 会整段落在 KV-bound 区。
- 逃生:L 很长时 r* 会低到状态池装不下一个请求,改 pin `--max-mamba-cache-size = 目标并发 × S`,别用 sub-0.15 的 r。
- 免费杠杆优先:启动日志显示大量闲置显存时,先提 mem-fraction(→0.92)再动 split。

> 本 case(qa-chat-3.5k-1k,输入仅 3500)**不是长上下文**,mamba 池压力小;混合架构缓存策略仍值得 A/B,但 ratio 公式的极端逃生场景用不到。

---

## 8. 调度层 — `official`(唯一给出闭环判据的维度)

`--schedule-conservativeness`(默认 1.0,搜 [0.3, 1.0, 1.3]):
- token usage < 0.9 且 #queue-req > 0 → 降到 0.3
- 频繁 "KV cache pool is full. Retract requests" → 升到 1.3(retract ~1 次/分钟内可接受)
- **P99 TTFT 炸掉的常见杀手就是 retract 重跑**:任何频繁 retract 的候选,即使 p50 好看 P99 也炸 → 应作自动否决条件。

---

## 9. 并发档怎么选(直接决定结论对不对)— `judgment`

- **第一轮用贴近 SLA 边界的并发,不要拍一个数。** 太高全 fail SLA(白跑),太低测不出显存类参数差异。
- 有同卡同场景历史数据取「已知刚好过 SLA」那档;没有则先 C=1、C=2 各跑一次基线定边界。
- 第二轮只对前 3-5 名扫并发梯度,二分找各自 SLA 饱和点,在饱和点比 goodput(不同配置饱和点可能不同)。
- 前 3 名各重跑 2-3 次算标准差(差异小于噪声的并列不硬排名次)。

---

## 10. 目标函数 — `judgment`

```
goodput(cfg) = max_C  total_token_throughput(cfg, C)
               s.t.  Pxx_TTFT(cfg,C) ≤ sla.max_avg_ttft_ms
                 AND Pxx_TPOT(cfg,C) ≤ sla.max_avg_tpot_ms
                 AND avg_output_tokens ≈ output_len  (未被截断)
```
**不能用裸吞吐排序**——峰值吞吐那档几乎必然违反 SLA,拿它当最优上线就翻车。percentile 从 job 的 SLA 读,不写死(前人无共识:vLLM P99 / Vidur P90+P99 / BBuf P50)。

---

## 11. 数据体检(不过则该档数据作废)— `measured`

1. **压测必须用 `--backend sglang`(走 `/generate`)**:thinking 类模型走 `/v1/chat/completions` 时 `delta.content` 可能为空,按 content 统计会把 decode 吞吐算成 0。团队两条独立记录都踩过。
2. **`avg_output_tokens ≥ output_len × 0.9`**:高并发下网关超时/并发限制会提前终止,截断的档位数据必须作废(否则吞吐虚高)。
3. **丢弃第一条**:首次运行有 kernel 编译尖峰,会让第一个候选看起来极差(warmup 或丢第一条)。

---

## 12. A 阶段跑完必须读的启动日志字段 — `official`

- `available_gpu_mem`:5-8GB 最佳;10-20GB 说明 mem-fraction 太保守,可上调。
- `max_total_num_tokens`:÷ 单请求 token 数 = 并发天花板,E 阶段取值不要超过它。
- `max_num_reqs`:混合架构模型这个数可能远小于上面算出的值,说明卡的是 mamba state 池而不是 KV,此时调 `--mamba-full-memory-ratio` 而不是加显存。
- **顺带记录回落的 context(应为 262144)和 max_total_num_tokens 备查**(呼应 §0)。
