# 互联拓扑与 TP/PP

## PCIe

PCIe 卡间通信通常弱于 NVLink。跨卡部署时优先评估 PP，减少高频同步通信；
同时记录 pipeline bubble、stage 负载均衡和 TTFT 影响。

## NVLink

NVLink 卡间通信性能较好时，TP 通常更适合低延迟路径，但仍需实测通信开销、
KV Pool 和每卡吞吐，不能直接假定 TP 等于 GPU 数量。

拓扑只决定候选优先级，不直接产生 hard constraint。
