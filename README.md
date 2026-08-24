# llm-infer-tuner — 大模型推理参数寻优 Agent

在**指定模型 / 卡型 / 工作负载 / SLA** 下,自动生成一批 SGLang 启动配置候选,拿到目标机器上真机压测,
在 SLA 约束内按 **goodput(有效吞吐)** 排名,输出可复现的最佳部署配置。

核心理念一句话:**AI 定考什么题,确定性代码当阅卷老师。**
参数决策(读知识库、算并行度、拼启动命令)交给 AI;执行、压测、SLA 判定、排名全是确定性代码,可复现、可审计。

---

## 两阶段架构

```
        input/jobs/<job>.json          input/targets/<machine>.json
        (测什么:模型/卡/负载/SLA)      (在哪测:SSH/模型路径/镜像)
               │                                  │
               ▼                                  │
   ┌───────────────────────┐                      │
   │ 阶段一  gen_configs.sh │  AI 读 skill+knowledge+catalogs
   │ (本地,几分钟)         │  → 生成候选启动配置
   └───────────────────────┘                      │
               │                                  │
               ▼                                  ▼
     outputs/<job>/configs.jsonl   ──►  ┌───────────────────────┐
     (每行一候选:params/cmd/reasons)    │ 阶段二  run_executor.sh │
                                        │ (远程,较久)           │
                                        │ 起服务→压测→采指标→排名 │
                                        └───────────────────────┘
                                                  │
                                                  ▼
                                    outputs/<job>/results/ranking.json
                                    (SLA 内 goodput 排名 + 最优配置)
```

- **阶段一(生成)**:`gen_configs.sh` 用 `claude -p` 让 AI 读 `.claude/skills/` 里的知识库和 `catalogs/`,
  产出一批**机器无关**的启动配置候选(命令里模型路径是 `${MODEL_PATH}` 占位符)。
- **阶段二(执行)**:`run_executor.sh` 从 `input/targets/` 读目标机器信息,SSH 过去 `docker run` 起容器,
  逐候选起服务、做**自适应并发 sweep**(指数扩张 + 二分找 SLA 饱和点)、采集指标、算 goodput、排名。

全流程 AI 只被调用两次:①阶段一生成候选;②阶段二启动时生成压测命令模板。其余全是确定性代码。

---

## 快速开始

### 0. 前置

- Python 3.13 + [uv](https://github.com/astral-sh/uv)
- `jq`、`claude` CLI(第三方网关也可)
- 目标 GPU 机器:SSH 免密可达、已装 docker、已备好模型权重和 sglang 镜像

```bash
uv sync            # 装依赖
```

### 1. 配置 claude 网关 / api-key

凭证走环境变量,不写进代码:

```bash
cp .env.example .env
vim .env           # 填 ANTHROPIC_BASE_URL 和 ANTHROPIC_AUTH_TOKEN
chmod 600 .env
```

`.env` 已被 `.gitignore` 排除,不会进仓库。

### 2. 写两个输入文件

**`input/jobs/<job>.json` —— 测什么**(字段都是查表用的 ID):

```json
{
  "job_id": "qwen36-35b-a3b-fp8_pro5000_8x72g_qa-chat-3.5k-1k_cand2",
  "engine": "sglang",
  "gpu_model": "pro5000",
  "gpu_count": 8,
  "gpu_memory_gb": 72,,
  "model": "qwen36-35b-a3b-fp8",
  "image": "sglang-v0.5.16",
  "workload": "qa-chat-3.5k-1k",
  "benchmark_method": "sglang-bench-serving",
  "sla": { "max_avg_ttft_ms": 2000, "max_avg_tpot_ms": 80, "min_success_rate": 0.99 },
  "search": { "max_candidates": 2, "max_runtime_minutes": 180 }
}
```

> `gpu_model / model / workload` 三个 ID 必须能在 `catalogs/*.yaml` 里查到,`image` 能在
> `catalogs/sglang-images.yaml` 里查到;查不到就先去对应文件补一张卡
> (每个文件底部都有「缺了怎么补、去哪查」的注释)。

**`input/targets/<job_id>__<ip>.json` —— 在哪测**:

> 文件名规则:`{job_id}__<ip>.json`,job_id 段与 `input/jobs/` 对齐,ip 为目标服务器 IP(点换成横杠,如 `122.51.115.16` → `122-51-115-16`)。
> GPU 字段(`gpu_model`/`gpu_count`/`gpu_memory_gb`)用于和 job.json 校验硬件匹配,防止跑错机器。

```json
{
  "gpu_model": "pro5000",
  "gpu_count": 8,
  "gpu_memory_gb": 72,
  "ssh_target": "ubuntu@122.51.115.16",
  "ssh_password": "",                     // 选填: 密码登录(留空走 key 免密)
  "model_host_dir": "/data/autotune/models/Qwen3.6-35B-A3B-FP8",
  "model_container_path": "/data/autotune/models/Qwen3.6-35B-A3B-FP8",
  "image_ref": "hai-beijing.tencentcloudcr.com/ai/sglang:v0.5.16-cu129",
  "port": 30000
}
```

> SSH 默认走 key 免密(`remote.py` 用 `BatchMode=yes`);若 target.json 填了 `ssh_password`,则用 `sshpass` 走密码登录。

### 3. 跑两阶段

```bash
# 阶段一:AI 生成候选 → outputs/<job>/configs.jsonl
./gen_configs.sh input/jobs/qwen36-35b-a3b-fp8_pro5000_8x72g_qa-chat-3.5k-1k_cand2.json

# 阶段二:真机压测 + 排名 → outputs/<job>/results/ranking.json
./run_executor.sh input/jobs/qwen36-35b-a3b-fp8_pro5000_8x72g_qa-chat-3.5k-1k_cand2.json input/targets/qwen36-35b-a3b-fp8_pro5000_8x72g_qa-chat-3.5k-1k_cand2__122-51-115-16.json
```

### 4. 看结果

```bash
jq . outputs/qwen36-35b-a3b-fp8_pro5000_8x72g_qa-chat-3.5k-1k_cand2/configs.jsonl          # 阶段一:AI 生成的候选
jq . outputs/qwen36-35b-a3b-fp8_pro5000_8x72g_qa-chat-3.5k-1k_cand2/results/ranking.json   # 阶段二:goodput 排名
```

---

## 目录结构

```
gen_configs.sh          # 阶段一入口:注入 job → 调 claude → jq 拆成 configs.jsonl
run_executor.sh         # 阶段二入口:读 target → 拼参数 → 跑 runners.executor

.claude/skills/         # AI 读的知识库(改经验只改这里,不改代码)
  sglang-server-config-gen/   # 服务端:出 launch_server 启动配置
    SKILL.md            #   流程入口:读什么、推导步骤、输出契约、3 道硬闸
    knowledge.md        #   全部调优判据(attention/并行度/搜索空间/pin/排除…)
  sglang-client-config-gen/   # 客户端:出 bench_serving 分并发压测命令
    SKILL.md  knowledge.md

catalogs/               # 引擎无关的共享事实 + 引擎相关镜像事实
  gpu.yaml              #   显卡型号 → 算力(sm)/NVLink(显存/卡数在 JobSpec)
  models.yaml           #   模型 → 架构/是否 MoE/专家数/量化块/权重大小/专属 flag
  workloads.yaml        #   负载 → 输入输出长度/并发梯度/采样参数
  sglang-images.yaml    #   sglang 各版镜像事实(CUDA/attention 菜单/flag 白名单)

runners/                # 阶段二执行器(全确定性,无 AI 判断)
  executor.py           #   主循环编排:读 configs → 逐候选 起服/压测/采集/排名
  remote.py             #   SSH + docker 薄封装(BatchMode 免密)
  container.py          #   容器生命周期:docker run 起 / exec 跑 / 清理
  readiness.py          #   轮询 /health 判就绪
  bench_runner.py       #   调 client skill 出压测命令模板 → env 替换 → 跑
  concurrency_search.py #   自适应并发 sweep(指数扩张 + 二分找饱和点)
  metrics.py            #   解析 bench_serving 输出 → RunResult
  ranker.py             #   数据体检 + SLA 约束下算 goodput + 排名

schemas/job_spec.py     # JobSpec 的 Pydantic 契约(严格校验)
references/benchmark_methods/   # 压测方法契约(sglang_bench_serving.json)
input/jobs/  input/targets/     # 你的输入
outputs/                        # 产物(gitignore)
tests/                          # pytest
```

---

## 设计原则

1. **加经验 = 改文本,不改代码**。新模型 / 新卡 / 新镜像 / 新调优判据,只改 `catalogs/*.yaml`、
   `.claude/skills/**/knowledge.md`、`catalogs/sglang-images.yaml`,生成器代码零改动。
2. **配置机器无关**。阶段一产出的 `configs.jsonl` 里模型路径是 `${MODEL_PATH}` 占位符,
   同一份配置可在不同机器间搬运,换机器不必重新调 AI。
3. **AI 出题、代码阅卷**。参数决策交给 AI(灵活);执行、SLA 判定、goodput 排名交给确定性代码(可复现、可审计)。
4. **goodput 而非峰值吞吐**。排名基于 SLA 约束(TTFT / TPOT / 成功率)内的最大有效吞吐,不是不管延迟的极限吞吐。
5. **不存在脱离工作负载的通用最优**。最优配置永远对应具体的 模型 × 卡型 × 负载 × SLA。

---

## 测试

```bash
uv run python -m pytest tests/ -q
```

---

## 说明

- `input/targets/` 里存的是**目标机器的部署事实**(IP / 模型路径 / 镜像),不含密码 —— SSH 走 key 免密。
- 真正的凭证(网关 token 等)只放在 `.env`(已 gitignore),绝不进仓库。
- 当前落地引擎为 SGLang;架构上 `catalogs/` 引擎无关,未来 `engine: vllm` 可复用同一套 catalog,
  新建 `.claude/skills/vllm-config-gen/` 即可。
