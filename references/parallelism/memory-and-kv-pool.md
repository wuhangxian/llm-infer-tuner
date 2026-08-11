# 并行方案的显存与 KV Pool 可行性

判断卡数和并行方案时，不能只看权重能否加载，还要考虑：

- 权重和激活显存；
- CUDA Graph、通信 buffer 和框架开销；
- 目标 `context_length` 下的 KV Cache；
- 目标并发下可预留的 KV Pool；
- TP 下 KV head replication；
- PP 各 stage 的显存不均衡。

启动日志和 `nvidia-smi` 是优先证据。没有实测时只能生成 heuristic 候选，不能
把显存估算写成 hard constraint。
