# 第一章
## 并行计算
### 并行
先决条件:  判断数据依赖关系, 

> 任务并行

各任务间独立
多核分布式函数

> 数据并行

划分数据

- 按块划分
- 循环划分


### 计算架构

按数据(单/多)和指令(单/多)组合分为四种, 主流为SIMD

判断一个架构的优劣
- 时延
- 带宽 gigabytes/sec
- 吞吐量 gflops (billion fl oating-point operations per second)

按内存管理分为两种
- 集群
多节点分布式内存, 每个处理器拥有局部内存分配, 不同处理器间可以通信

- 多处理器共享内存
pcie

## 异构计算
### 异构架构
衡量GPU计算能力的两个重要特征:
- CUDA core
- 内存大小

另一种衡量机制:
- 计算性能峰值 gflops
- 内存带宽

### CUDA
管理GPU设备和分配thread的API, 两API互斥
- CUDA Driver API
- CUDA Runtime API





