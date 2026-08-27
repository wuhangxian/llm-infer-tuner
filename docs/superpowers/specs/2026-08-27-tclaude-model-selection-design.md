# tclaude 模型选择设计

## 目标

`gen_configs.sh` 生成 SGLang 候选配置时统一通过 `tclaude` 调用模型。调用者可以在命令行选择 tclaude 暴露的任意模型；没有指定模型时，稳定地使用 `claude-hy3`，不依赖 tclaude 自身 Default 将来如何变化。

## 命令行接口

保留现有两个位置参数，并新增可选的 `--model`：

```bash
./gen_configs.sh [--model MODEL] <job.json> [out.json]
```

示例：

```bash
# 默认 claude-hy3
./gen_configs.sh input/jobs/job.json

# 临时切换模型
./gen_configs.sh --model claude-opus-4-8 input/jobs/job.json
./gen_configs.sh --model 'claude-glm-5.2[1m]' input/jobs/job.json
```

`[1m]` 含 Shell 通配符字符，文档示例使用引号保护；引号不属于模型名。

## 行为设计

- 默认模型常量为 `claude-hy3`。
- `--model MODEL` 和 `--model=MODEL` 均覆盖默认值；参数顺序不影响现有 job/output 位置参数。
- 不在项目内维护模型白名单。模型是否存在由 tclaude 校验，避免网关新增模型后还要修改项目。
- 流式和非流式两条生成路径都使用同一个 Bash 参数数组调用 `tclaude`，不使用 `eval`，确保 `[1m]` 等字符原样传递。
- 前置检查从 `claude` 改为 `tclaude`；启动日志明确打印最终模型以及它是默认值还是命令行覆盖值。
- tclaude 启动器会为其子 Claude Code 进程设置自己的 `ANTHROPIC_BASE_URL`，因此保留项目现有 `.env` 加载逻辑也不会把请求导回旧网关。
- 本次只修改第一阶段入口 `gen_configs.sh`，不改变 `planner/claude_code_client.py` 等其他调用点。

## 错误处理

- `--model` 缺值、未知选项、位置参数过多时，在调用模型前打印用法并非零退出。
- tclaude 不在 `PATH` 时立即报错。
- 不可用模型由 tclaude 返回非零退出码，沿用现有 stderr 日志和退出码处理。

## 测试

通过临时目录和假的 `tclaude` 端到端执行脚本的非流式公共接口，验证：

1. 未指定模型时实际传入 `--model claude-hy3`。
2. `--model 'claude-glm-5.2[1m]'` 原样覆盖默认值。
3. 现有 `<job> [out]` 位置参数保持兼容，并能解析结构化输出。
4. 缺失 model 值或未知选项会在 tclaude 调用前失败。

