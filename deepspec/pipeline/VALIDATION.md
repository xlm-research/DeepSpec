# DSpark 流水线验收记录

## 整机分离全十六卡（2026-09-16）

**整机分离的小模型探针、真实 4K/3 步与 128K/3 步均已通过。**
第一台 `172.20.1.195` 八卡全部用于原生 vLLM DP2×TP4，第二台 `172.20.5.39`
八卡全部用于原生 TorchTitan DP(shard)2×TP4/GAS2/global batch 4，第二台提供唯一 Mooncake CPU 池。
用户要求按整机分离；下文历史十二卡混布结果不作为当前布局验收。RDMA 跳过，十二卡 TCP20 未重复。

生产使用一个 AsyncLLM frontend 和两个本地 EngineCore；原生 Ray DP placement 仅允许第一台，
`data_parallel_size_local=2`。消费由第二台单个 `torchrun --standalone --nproc-per-node=8` 启动。
单 dispatcher 全局按序预留、按 `position % 2` 显式路由，每组最多一个生成调用。
两个 TP0 writer 可以乱序 READY；buffer 在写前和发布时校验唯一归属，全部 Store 写入结束才完成生产。
故障取消其他请求及阻塞准入；清理按完整 Run ID 和进程身份执行。验收会拒绝生产/消费共享节点，
即使 GPU 编号不重叠。推理区间先取并集再计算与消费计算的重叠。

| 项目 | 4K | 128K |
| --- | --- | --- |
| Run ID | `dspark-7637338bd9cc` | `dspark-6f0a5195f37b` |
| 启动器退出码 | 0 | 0 |
| 真实样本 | 12 × 4096，互不重复 | 12 × 131072，互不重复 |
| 读取与训练 | 48 次 SHA256；八 rank 各三次更新、context 梯度通过 | 同左 |
| 释放 | 12 批均在所属四 ACK 后释放，剩余 0 | 同左 |
| Checkpoint | 独立读取 fc optimizer step=3；native cursor6、八分片与 metadata 通过 | 同左 |
| 池 / 预留峰值 | 4 GiB / 1.875 GiB | 64 GiB / 45.012 GiB |
| 第 2/3 次更新间隔 | 1.90 / 2.10 秒 | 64.60 / 66.00 秒 |
| 首次推理至消费退出，含 checkpoint | 107.51 秒 | 299.84 秒 |
| 清理 | 本轮模型/GPU 进程清空，原生 DP PG 移除，Ray 两节点和 16 GPU 保留 | 同左 |

4K metadata SHA256：`dcf0fced4db990ab7a17045a3c79742820b6b13d08730485fdbd1e803227219e`。
128K metadata SHA256：`8be16a5eb81a8aa20b43b47b7b87579b76d2a545d5ed1f65bee3c3d689f47a55`。
两轮使用同一份冻结训练源码，各自输入 identity 与历史相同长度短测一致。
128K 三步日志 loss 为 3.91829 / 3.09029 / 3.89470；每个 rank 的样本顺序精确匹配计划。
SHA256、有限值检查、原生 DSpark K/V 数学路径和优化器均保留；不宣称 checkpoint 逐位等价。

128K 有 140 次容量背压，无内存压力或其他 GPU 任务。每批 writer 中位 34.65 秒，
其中转换 6.68 秒、Store put 20.42 秒；两组消费本机 get 中位约 26.94 秒。
计时在 FeatureBuffer 主机上统一记录，包含校验及共享资源上的并发调用；不作为纯网络带宽，
也不从三步观测推断长期吞吐收益。全十六卡 20 步延长、跨机 RDMA、远端 GPU 直传和
完整训练配置仍未验证；目标 CUDA 节点的本机 GPU 目标缓冲区探针已通过，记录见
[目标 CUDA 节点传输验证](../../scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/TARGET_CUDA_TRANSPORT_VALIDATION.md)。

整机分离探针为 `native_probe4_separated`，使用可导入 worker extension 与字符串 RPC；
八个 worker 记录实际接收请求，退出和清理通过，无需打开不安全序列化。
初版组件 27 passed，Ray 事件循环入口/失败清理修复后定向 4 passed；
整机分离改动后组件 24 passed，另有同节点拒绝定向 2 passed。检查有重叠，不累加成独立用例数。
保留两个未验收混布轮：run1 在推理前因嵌套事件循环失败，已修复并清理；
run2 在用户指出整机分离要求后停止，当时 0 READY、0 更新，清理通过。

- [4K 独立核验](../../outputs/dspark_two_node_20260916_4k_full16_separated1/verification-status.json)
- [128K 独立核验](../../outputs/dspark_two_node_20260916_128k_full16_separated1/verification-status.json)
- [最终资源与结果汇总](../../outputs/dspark_full16_20260916/final-summary.json)
- [整机分离探针](../../outputs/dspark_full16_20260916/native_probe4_separated/verification-status.json)
- [启动方法](../../scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/README.md)

## 十二卡 DP2 的 128K TCP20（2026-09-16）

**20 次更新的延长测试已正常退出并完成独立核验。**
运行 [dspark_two_node_20260916_128k_dp2_tcp20](../../outputs/dspark_two_node_20260916_128k_dp2_tcp20)，
Run ID `dspark-124899ba1b40`。一个 vLLM TP4 生产者、一个 TorchTitan TP4 × DP(shard)2 / GAS2
消费者，共十二卡；第二台提供唯一 64 GiB 池，Mooncake TCP/CPU，跨机 NCCL Socket。
用户要求继续跳过 RDMA，本轮未修改训练源码或并行加载其他 GPU 模型。

| 检查项 | 结果 |
| --- | --- |
| 输入与退出 | 80 个不同的 131072-token 样本，单个 epoch，与 DP1 TCP20 的 input identity/顺序相同；启动器退出码 0 |
| 训练与校验 | 八个 rank 各完成 20 次更新、首次 context 梯度检查通过；320 次读取 SHA256 与每 rank 样本顺序通过 |
| 释放与容量 | 80 批全部在所属四个 TP rank ACK 后释放，剩余 0；预留峰值 45.012 GiB，1266 次容量背压 |
| Checkpoint | 完整 step-20；实际读取 fc optimizer step=20，DP 微步游标 40、全局样本数 80，metadata 与八个分片引用范围通过 |
| 内存与进程 | 两机各 381 次监控记录，无内存压力或其他 GPU 任务；清理后本轮标识/GPU 进程均为空，Ray 两节点和 16 张空闲 GPU 保留 |
| 源码与回归 | 两端源码 hash 与之前 DP2 短测一致，保留其 20 项组件回归；本轮只新增分析工具和文档，Ruff 通过 |

Metadata SHA256：`2d96759a0512dadfade64644598a3f8889dfa6bbf2270ea13e6e1d107b123440`。
全部更新的 loss 与日志梯度范数有限；前三步与 DP2 短测一致，最后一步为 2.67902 / 1.3828。
本轮不以不同样本的更新组推断收敛，不宣称与 DP1 loss/checkpoint 等价。
累计特征逻辑写入 600.156 GiB，八个 rank 读取合计 2400 GiB。

| 两轮 20 步观测 | DP1，八卡 | DP2，十二卡 |
| --- | --- | --- |
| 第 2–20 步间隔中位数 | 93.56 秒 | 80.36 秒 |
| 第 2–20 步间隔均值 | 93.19 秒 | 80.33 秒 |
| 输入 token 吞吐 | 5626 /秒 | 6527 /秒 |
| 首次推理至消费者结束，含 checkpoint/退出 | 1958.07 秒 | 1719.01 秒 |

本轮活动区间为 **28.65 分钟**，不含模型初始化；更新间隔范围 70.68–87.98 秒。
输入吞吐观测比为 1.160×，使用相同 80 样本及特征转换实现，但 DP 布局、GPU 数量与运行时间不同。
这是两次顺序运行的观测，不代表统计保证或长期扩展效率；输入 token 也不是 draft 预测 token。

每样本主机计时中位数：转换 6.57 秒、描述符/SHA256 7.66 秒、put 5.32 秒、writer 总耗时 19.70 秒；
DP0 本机 get / DP1 跨机 get 为 10.46 / 9.59 秒，读取 SHA256 为 5.51 秒。
单 writer 每样本平均 20.13 秒，四样本的供给周期约 80.51 秒，接近更新间隔均值 80.33 秒。
结合当前单 writer 的实现，这是生产端仍限制整体吞吐的证据；十六卡阶段将优先扩展生产并发。
这些耗时包含同步、校验和其他并发读写的影响，不代表纯网络带宽。

第 4–20 步之间选取 251 个监控序号，比较前/后四分位窗口：

| 节点 | RSS 中位数前→后 | 增量 | 全轮 RSS 峰值 | 最低准入 headroom |
| --- | --- | --- | --- | --- |
| 第一台，生产+四卡消费 | 160.892→160.918 GiB | 26.34 MiB | 207.62 GiB | 3404.66 GiB |
| 第二台，池+四卡消费 | 126.136→126.179 GiB | 44 MiB | 172.87 GiB | 3431.72 GiB |

十二张 GPU 的显存前/后中位数都没有增长。区间由第二台本机的事件时间选择，第一台按
同轮监控序号对齐，没有直接相减两机时钟；最终 checkpoint 不纳入稳态趋势窗口。
RSS 求和可能重复计算共享页，准入使用节点/cgroup headroom。
约半小时活动区间未出现明显累积增长，但不能排除慢泄漏或代替小时/天级验证。
全十六卡与 GPU 直传仍未运行。

- [完整独立核验与 checkpoint](../../outputs/dspark_two_node_20260916_128k_dp2_tcp20/verification-status.json)
- [时序、内存、显存与背压分析](../../outputs/dspark_two_node_20260916_128k_dp2_tcp20/stability-analysis.json)
- [DP1/DP2 观测对比](../../outputs/dspark_two_node_20260916_128k_dp2_tcp20/performance-comparison.json)
- [清理检查](../../outputs/dspark_two_node_20260916_128k_dp2_tcp20/cleanup-verification.json)
- [更新间隔 CSV](../../outputs/dspark_two_node_20260916_128k_dp2_tcp20/step-intervals.csv)、[内存趋势 CSV](../../outputs/dspark_two_node_20260916_128k_dp2_tcp20/memory-trend.csv)

## 两机十二卡消费者 DP2（2026-09-16）

用户要求暂时跳过 RDMA。新增 `--consumer-dp 2`，生产仍为一个 vLLM TP4，消费改为
一个 TorchTitan TP4 × DP(shard)2 / GAS2 实例，使用原生 FSDP2。第一台 `172.20.1.195`
使用四卡生产与四卡消费，第二台 `172.20.5.39` 使用四卡消费及唯一 Mooncake CPU 池。
两端 torchrun 组成一个 world size 8 的消费实例；DP 通信跨机，TP 组留在各自节点。
Mooncake 为 TCP/CPU，NCCL 显式使用 Socket、禁用 IB；RDMA 不作为本阶段前置条件。

全局更新批量仍是四个样本，DP0 / global rank 0–3 读取偶数位置，DP1 / rank 4–7 读取奇数位置。
各样本由所属四个 TP rank 校验并确认后释放，其他 DP 组不读取或 ACK 该样本。
三次更新是 12 个不同样本、48 次 rank 读取；原生同步 DP 微步游标为 6，全局样本计数为 12。
pool/window 继续保证全局四样本更新组容量，第一台预算合并生产与消费暂存。

| 检查项 | 4K 实际结果 |
| --- | --- |
| 运行 | `outputs/dspark_two_node_20260916_4k_dp2_run1`，Run ID `dspark-c54bccc6a704`，启动器退出码 0 |
| 输入与生命周期 | 12 个不同样本，均为 4096 token；48 次 SHA256，12 批全部在所属四个 ACK 后释放，剩余 0 |
| 训练 | 八个 rank 的 DP/TP 坐标和 GAS2 正确，各完成三次更新；fc、第一层 context K/V 梯度有限且非零 |
| Checkpoint | 独立读取 fc optimizer step=3，原生游标 6；metadata hash 与八个分片引用范围通过 |
| 池与计时 | 4 GiB 池，预留峰值 1.875 GiB；第 2/3 次更新间隔 5.63 / 5.55 秒 |
| 清理 | 本轮标识进程与两机 GPU 进程清空；Ray 保留两节点、16 张空闲 GPU |

4K checkpoint metadata SHA256：`acfb17408ee6257f61f525d994de6c58516e783bd7ba5cb2b38de99e5993027e`。
首次推理到全部消费者结束为 84.85 秒，含 checkpoint/退出，不含模型初始化。
完整有限值检查、逐字节 SHA256、两端源码 hash 检查均保留。

**同拓扑 128K 也已正常退出并完成独立核验**：
`outputs/dspark_two_node_20260916_128k_dp2_run1`，Run ID `dspark-cf5513a417f8`。

| 检查项 | 128K 实际结果 |
| --- | --- |
| 输入与退出 | 12 个不同的 131072-token 样本，单个 epoch；与 DP1 短测样本及 input identity 相同，启动器退出码 0 |
| 样本顺序与释放 | 八个 rank 仅按顺序读取所属 DP 样本，48 次 SHA256 校验；12 批全部在所属四个 ACK 后释放，剩余 0 |
| 原生训练 | 八个 rank 的 TP/DP 坐标与 GAS2 正确，context 梯度有限且非零，各完成三次更新 |
| Checkpoint | 实际读取 fc optimizer step=3，原生 DP 微步游标 6、全局样本 12；metadata hash 与八个分片引用范围通过 |
| 容量 | 唯一 64 GiB 池，预留峰值 45.012 GiB；140 次容量背压，无节点内存压力 |
| 源码与清理 | 运行期间未修改 Python 源码，两端 hash 匹配；两机本轮标识/GPU 进程清空，Ray 两节点和 16 张空闲 GPU 保留 |

128K checkpoint metadata SHA256：`1ece50d0e830c4e99d06e13f957ee477a639adf788d0169d6076dccf7bd2fff6`。
三步日志 loss 为 3.91829、3.09029、3.89470，日志梯度范数为 182.0000、10.4375、12.7500；
八个 rank 的全局指标一致且有限。DP1→DP2 改变样本和累积布局，未控制跨布局的随机采样轨迹，
不宣称与 DP1 的 loss/checkpoint 逐位相同，也不从三个不同更新组的 loss 推断收敛。

| 128K 主机计时阶段 | 观测值 |
| --- | --- |
| 第 2/3 次更新间隔 | 70.69 / 78.98 秒，中位 74.83 秒 |
| 首次推理至全部消费者结束 | 344.33 秒，含 checkpoint/退出，不含模型初始化 |
| 特征转换 / writer 总耗时 | 中位 6.52 / 20.57 秒/样本 |
| 跨机 Store put | 中位 6.36 秒/样本 |
| DP0 本机 get / DP1 跨机 get | 中位 11.78 / 11.88 秒/样本/rank |

跨组件区间采用第二台 FeatureBuffer 的统一时间戳，各调用耗时来自各进程的本机时钟，
没有直接相减两台机器的时间。日志中的 GPU/通信调用和 SHA256 含同步与校验开销，
这些数字不等于纯网络带宽或 GPU kernel 时序。
第一台进程 RSS 峰值 208.12 GiB（同时包括生产和消费），第二台 172.78 GiB，
最低准入 headroom 分别为 3389.33 / 3447.02 GiB；RSS 求和可能重复计算共享页。
这轮仅三次更新，随后 DP2 的 20 步延长测试见本文开头；全十六卡和 GPU 直传仍未验收。

回归 cluster/buffer/store 共 **20 passed**（89.02 秒，14 条上游警告），覆盖原生 FeatureLoader
的四/八 CPU rank、DP1/DP2 数据顺序、完整更新组容量、所属 TP 组 ACK、释放与不同 launcher
共用 rendezvous。另用真实两机八卡探针验证 global/TP all-reduce、DP all-gather 和 reduce-scatter，
各 rank 值与位置正确，两端日志出现 `NET/Socket`。Ruff 通过。

- [4K 独立核验](../../outputs/dspark_two_node_20260916_4k_dp2_run1/verification-status.json)
- [4K 进程清理](../../outputs/dspark_two_node_20260916_4k_dp2_run1/cleanup-verification.json)
- [128K 结果与完整核验](../../outputs/dspark_two_node_20260916_128k_dp2_run1/verification-status.json)
- [128K 进程清理](../../outputs/dspark_two_node_20260916_128k_dp2_run1/cleanup-verification.json)
- [组件回归](../../outputs/dspark_cross_node_dp_20260916/tests.log)
- [两机 GPU 通信探针](../../outputs/dspark_cross_node_dp_20260916/collectives/result.json)
- [启动命令](../../scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/README.md#两机消费者-dp248-卡)

## 两机 128K TCP20 与 RDMA 检查（2026-09-16）

**20 次更新的延长测试通过，RDMA 跨机检查被第一台容器的网络配置阻塞。**
TCP 运行目录为 [dspark_two_node_20260916_128k_tcp20](../../outputs/dspark_two_node_20260916_128k_tcp20)，
Run ID `dspark-e7988ab71d12`。仍为生产/消费各占一台机器的四张 GPU，TP4+TP4/DP1/GAS4，
128K、消费端唯一 64 GiB CPU 池、完整有限值与 SHA256 校验。

| 检查项 | 结果 |
| --- | --- |
| 输入与退出 | 单个 epoch 的 80 个不同样本，每个 131072 token；启动器退出码 0 |
| 读取与更新 | 320 次 rank 读取校验，各 rank 按 0–79 顺序消费并完成 20 次更新；context 梯度检查通过 |
| 对象生命周期 | 80 批均在四个 rank 读取 ACK 后释放，剩余 0；累计逻辑特征写入 600.156 GiB |
| 池与背压 | 预留峰值 45.012 GiB，1477 次容量背压，无节点内存压力 |
| Checkpoint | 独立读取 DCP 的 fc optimizer step=20；metadata SHA256 与四个分片引用范围通过 |
| 源码与清理 | 两端源码 hash 匹配，运行期间未修改；本轮标识进程及两机 GPU 进程已退出，Ray 保留两节点/16 张空闲 GPU |

Checkpoint metadata SHA256：`ce3d7545e6f9a744322b85a6621b60210b6b4be6d23ad22ce8da5680d93f0a6a`。
20 步日志 loss 与梯度范数均有限；前三步与上轮短测一致，最后一步 loss 为 2.74412。
样本不同，不以首尾 loss 推断收敛。

第 2–20 次更新间隔中位 **93.56 秒**、均值 93.19 秒、范围 88.92–96.52 秒。
按每次更新四个 131072-token 输入计，区间吞吐约 5626 输入 token/秒；这不是 draft 预测 token
吞吐或 GPU 利用率。首次推理至消费者结束为 1958.07 秒（32.63 分钟，含 checkpoint/退出，
不含模型初始化）。本轮仅有独立 CPU 小包 RDMA 诊断，没有并行加载其他 GPU 模型。

| 主机计时阶段 | 全轮中位耗时 |
| --- | --- |
| CPU 转换，含输入读取 | 6.55 秒/批 |
| producer 描述符/SHA256 | 7.65 秒/批 |
| 跨机 Store put | 5.65 秒/批 |
| writer 总耗时 | 20.02 秒/批 |
| 消费 get / SHA256 | 11.17 / 5.02 秒/批/rank |

第 4 次至第 20 次更新的监控记录中，前/后四分位窗口的进程 RSS 中位数分别为：
生产端 63.870→63.869 GiB，消费端 165.123→165.189 GiB（+68 MiB）。全轮 RSS 峰值为
63.872/208.414 GiB，最低准入 headroom 为 3508.79/3382.03 GiB，没有明显持续累积。
RSS 求和可能重复计算共享页，只反映趋势；内存准入使用节点/cgroup headroom。
两机时钟有偏差，趋势窗口以消费端本机时间定位，再按同一轮监控序号对齐生产端。
**这轮约半小时的活动区间不能代替小时/天级测试，也不能排除慢泄漏。**

RDMA 首轮误选 `mlx5_z0`，跨机 Store put 因传输重试耗尽失败。VPD 证实它是 NVLink 管理桥。
改用 `mlx5_10` 后，第二台本机 1 MiB RDMA write/SHA256 成功；跨机 write 返回 -1，
第一台 QP 转 RTR 报 `No such device`。第一台只有 `eth0`，第二台具备 `net1–net8`，
第一台缺少对应 RoCE 网口且到对方 RoCE IP 走默认路由。先修复第一台平台网络分配/命名空间，
再运行标准 Store 探针与同配置 RDMA20；本轮没有 RDMA 训练结果或 TCP/RDMA 加速比。
详细记录与命令见 [RDMA_DIAGNOSIS.md](../../scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/RDMA_DIAGNOSIS.md)。

新监控经过真实运行，相关 `test_pipeline_cluster.py` 为 4 passed（1.74 秒），Ruff 通过。
独立验收脚本初次比较内存中的 tuple 与 JSON list 时失败，规范化 JSON 后重跑通过；
此处仅修正验收工具，没有修改训练源码。日志及脚本保存在诊断目录。

- [结果与 checkpoint 核验、完整计时、内存趋势](../../outputs/dspark_two_node_20260916_128k_tcp20/verification-status.json)
- [进程清理与 Ray 资源检查](../../outputs/dspark_two_node_20260916_128k_tcp20/cleanup-verification.json)
- [独立验收日志](../../outputs/dspark_rdma_stability_20260916/tcp20-verification.log)
- [RDMA 实测结果](../../outputs/dspark_rdma_stability_20260916/roce-probe.json)

## 生产转换优化复测（2026-09-16）

运行 [dspark_two_node_20260916_128k_opt1](../../outputs/dspark_two_node_20260916_128k_opt1)
正常退出，仍为相同两节点、TP4+TP4、128K、64 GiB 池、TCP/CPU、12 批与 3 次更新。
48 次读取校验、四个 rank 的 context 梯度检查、全部特征释放通过；独立读取 DCP 的
fc optimizer step=3，metadata 校验及四个分片引用范围检查通过。两机 GPU 进程清空，Ray 保留。

`convert_hidden_states()` 跳过 CP1 恒等索引复制和空 padding 写入；CPU BF16 使用有界分块的
指数位检查拒绝全部 NaN/Inf，输出仍为独立连续缓冲区。CP head/tail 排列、padding、
CUDA 有限值检查路径、完整 SHA256 和训练语义保留。没有修改 vLLM 核心或放宽内存预算。

| 观测值 | 优化前 | 优化后 |
| --- | --- | --- |
| 特征转换中位耗时/批，含输入读取 | 42.34 秒 | 6.54 秒 |
| writer 总耗时中位数/批 | 56.53 秒 | 21.72 秒 |
| 第 2/3 次更新间隔 | 218.46 / 233.82 秒 | 83.81 / 92.99 秒 |
| 首次推理至消费者退出 | 791.19 秒 | 371.40 秒 |

最后一行包含校验、checkpoint 和退出，不含模型初始化；仅是一对三步短程运行，
不代表长期吞吐。优化后跨机 put 中位数 7.27 秒、消费 get 12.25 秒，生产加快后
重叠程度也发生变化，不将这一差异单独归因于网络。完整比较见
[performance-comparison.json](../../outputs/dspark_two_node_20260916_128k_opt1/performance-comparison.json)。

两次完整 `[131072,6,5120]` 单线程 CPU 对照的六个字段均逐字节一致：
42.50→6.65 秒、42.61→6.58 秒。真实训练三步日志 loss 为 4.50247、3.08793、3.93578，
与基线一致；四个 rank 记录的 fc、首层 K/V 梯度范数也相同。这不宣称整个 checkpoint 逐位相同。

相关回归 `test_qwen38_vllm.py` 与 `test_pipeline_cluster.py` 为 **22 passed**（16.19 秒），
覆盖所有 65536 种 BF16 编码、后续检查块中的 Inf、非连续输入、输出独立性和原有 CP/worker 行为。
Ruff 与 diff 检查通过。CPU 记录见
[profile.json](../../outputs/dspark_feature_conversion_20260916/profile.json)、
[comparison.json](../../outputs/dspark_feature_conversion_20260916/comparison.json)、
[tests.log](../../outputs/dspark_feature_conversion_20260916/tests.log)；整链路证据见
[独立核验](../../outputs/dspark_two_node_20260916_128k_opt1/verification-status.json)。

## 两机 128K 增补（2026-09-16）

**真实 Qwen3.8-27B target 与五层 DSpark draft 的两机 4+4、128K 短程训练已正常退出。**
运行目录：[dspark_two_node_20260916_128k_run1](../../outputs/dspark_two_node_20260916_128k_run1)，
Run ID `dspark-9287098d7f12`，共享项目 `/mnt/afs-agentpro/lezewei/DeepSpec`。

| 检查项 | 实际结果 |
| --- | --- |
| 节点与 GPU | 生产 `172.20.1.195` GPU 0–3；消费 `172.20.5.39` GPU 0–3；两端为八卡 B300 节点 |
| 拓扑 | 一个 vLLM TP4；一个 TorchTitan TP4/DP1/GAS4，每个模型的 rank 均在同一节点 |
| Store | 消费端唯一 64 GiB CPU 池，TCP 配置，端点 `172.20.5.39:44567` |
| 输入 | 12 个长度均为 131072 的微批，每批特征 8,055,160,848 字节 |
| 退出与生命周期 | 启动器退出码 0；12 批生产、12 批释放、剩余 0；全部释放均晚于四个 rank 的读取确认 |
| 读取、梯度及更新 | 48 次 SHA256 读取校验；四个 rank 的 fc、首层 K/V 梯度有限且非零；各完成 3 次更新 |
| Checkpoint | 完整 `step-3` DCP；独立读取 fc optimizer step=3，metadata hash 匹配，四个分片覆盖引用范围 |
| 容量 | 预留峰值 48,330,965,088 字节（45.012 GiB）；记录到 424 次背压等待 |
| 源码与清理 | 两节点源码 hash 在该轮核验时匹配；两机模型 GPU 进程已退出，Ray 集群保留运行 |

Loss 为 4.50247、3.08793、3.93578，来自不同累积组，不作为收敛证据。
Checkpoint metadata SHA256：`76e6aae18a54819cf27b3c2bf68a3d873a36c84e68d6dd3729bbbc0d3e972ad6`。

新增分项计时的全轮中位数如下。各项是主机调用时间；不同批次的中位数不能相加解释为
端到端延迟，也不等同于网络带宽或 GPU trace。

| 阶段 | 中位耗时 |
| --- | --- |
| 生产端特征转换，含输入读取与有限值检查 | 42.34 秒/批 |
| 生产端描述符与 SHA256 | 8.01 秒/批 |
| 生产端 Store 跨机 put | 5.74 秒/批 |
| 消费端 Store get | 10.33 秒/批/rank |
| 消费端 SHA256 | 5.12 秒/批/rank |
| 训练 forward/backward 主机区间 | 0.64 秒/批/rank，首批含编译开销 |

本轮显示生产端转换耗时高于跨机写入，下一步应细分转换中的复制、重排和有限值检查，
随后延长稳定性测试并做 RDMA 对照。尚未验证消费者跨机 DP、16 卡运行、GPU 直传、
长期稳定性或相对原流程的性能收益。

两机接入过程中的 4K 第一次运行因临时其他 GPU 进程中止。第二次原生训练和 checkpoint
完成，但启动器在退出阶段误判自身 rank，退出码为 1；该轮保留独立失败记录。
修复方式为按 PID 与启动时间缓存已确认进程归属、忽略 zombie/dead 状态，防止 PID 复用误判。
修复后的本轮 128K 完整验证了正常退出。

回归先运行 cluster/buffer/store/norm 四个文件：14 passed（62.30 秒）；随后新增退出竞态回归，
定向 cluster 测试为 4 passed（1.81 秒），合计覆盖 15 项不同测试。Ruff、Bash 语法和 diff 检查通过。

- [运行结果](../../outputs/dspark_two_node_20260916_128k_run1/result.json)
- [独立核验与分项计时](../../outputs/dspark_two_node_20260916_128k_run1/verification-status.json)
- [两机版本、GPU UUID、预算和源码 hash](../../outputs/dspark_two_node_20260916_128k_run1/environment.json)
- [退出后 GPU 检查](../../outputs/dspark_two_node_20260916_128k_run1/cleanup-verification.json)
- [4K 原生训练通过但启动器失败记录](../../outputs/dspark_two_node_20260916_4k_run2/verification-status.json)
- [启动命令](../../scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/README.md#两机-44)

## 最初单机 4K 验收（历史记录）

2026-09-16，真实 Qwen3.8-27B target 与五层 DSpark draft 的流水线运行正常退出。
本轮证明单机短程训练链路已打通；跨机接入、128K 长程运行和性能对照仍待后续验证。

## 基线与配置

| 项目 | 实际值 |
| --- | --- |
| 主机 | `app-e110ba357a9f4d969bc3d910774f334e-64776bd647-2n7s6` |
| 项目 | `/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm` |
| 分支 / commit | `dev/vllm_torchtitan` / `ee5d0c82f89358ef03f735af9ac3fd17feb33a39` |
| vLLM commit | `1ee54c40df7ffe2c8934f5bd1c79917f34cb954e` |
| Python 环境 | `/tmp/deepspec_vllm_torchtitan_envs`，未使用 uv |
| 生产者 | 一个 vLLM 实例，TP4，Ray 分配 GPU 0–3 |
| 消费者 | 一个 TorchTitan 实例，DP1 / TP4 / CP1 / PP1，GPU 4–7，GAS4 |
| Target | `/mnt/afs_agents/hongjiawei/share_models/Qwen/Qwen3.8-27B` |
| Draft | 原生五层 DSpark，H=5120，512 anchors，block size 7 |
| 特征层 | `[1,16,31,46,61]`，并保留最终 norm 后隐藏状态用于监督 |
| 样本 | 原生预处理生成的 12 个真实样本，每个 4096 token |
| 传输 | Mooncake Store，TCP 配置，pinned CPU 接收后传入 GPU |
| 内存池 | 单节点共享 4 GiB，特征预留窗口 8 批 |

上述 commit 是未提交改动的基线。本次增加 `deepspec/pipeline/` 与测试，修改原生
数据读取、微批物化接口及归一化初始化。原有文档删除、其他输出和 vLLM 子模块未提交
改动保持原状。实际执行代码的 SHA256 见运行目录的 `environment.json`、
`initialization-fix.json` 和 `verification-status.json`。

## 实际结果

运行目录：[`dspark_pipeline_4plus4_20260916_run4`](../../outputs/dspark_pipeline_4plus4_20260916_run4)。

| 检查项 | 结果 |
| --- | --- |
| 正常退出 | 启动器退出码 0；生产者、消费者均完成 |
| 生产 / 释放 | 12 / 12 批，剩余对象 0 |
| 每 rank 数据顺序 | 四个 rank 均按位置 0–11 消费，无遗漏或重复 |
| 读取校验 | 48 次读取全部通过 SHA256 校验 |
| 源对象释放 | 每批均在四个 rank 全部读取确认后释放 |
| 训练更新 | 四个 rank 均完成 3 次更新，消费位置依次为 4、8、12 |
| Loss | 4.56851、3.03412、4.12592；对应不同更新组，不据此判断收敛 |
| Context 梯度 | 四个 rank 的 fc、第一层 K/V 投影梯度均非零、有限 |
| Checkpoint | 完整 DCP `checkpoints/step-3`，训练步 3，消费位置 12 |
| Checkpoint 独立读取 | metadata SHA256 匹配；从 DCP 实际读取的 fc Adam step 为 3 |
| 背压 | 记录到 9 次等待；最大特征预留 2,013,790,336 字节，约 1.88 GiB |
| 组件与回归 | 10 项通过，47.57 秒 |
| 清理 | 本次模型 GPU 进程、Ray、Mooncake 均退出；临时暂停的 CI 已恢复 |

首个完整累积组的 fc 梯度范数为 116.49，四个 TP rank 一致。第一层 K、V 的本地
分片梯度范数分别位于 2.32–3.39 和 51.95–58.24，符合各 rank 持有不同分片的布局。
隐藏特征经过原生 `fc → hidden_norm → 各层 context K/V` 路径参与了反向传播。

事件记录中，推理区间与训练计算区间的交集约 0.783 秒，后台特征读取与计算区间的
交集约 2.333 秒。这些是主机事件区间，包含调用与同步开销；不能作为 GPU kernel
重叠比例或相对原流程的加速比。

## 跑通前修复的两个问题

1. **TP 归一化层漏初始化。** 指定环境的 Transformers 5.16.1 通用初始化器按
   类名识别 RMSNorm，`DraftNorm` 不符合名称条件。模型从 meta 存储物化后，这些
   权重未被正确设为 1，真实运行出现 fc 零梯度。新增回归测试将未初始化存储填充为
   NaN，稳定复现漏初始化；原生 `init_weights()` 按 `Qwen3RMSNorm` 继承关系显式
   初始化后，测试与真实四卡反向检查均通过。没有改变模型结构、loss 或优化器规则。
2. **检查未识别 SelectiveAC 包装路径。** `named_parameters()` 中出现
   `layers.0._checkpoint_wrapped_module...`，原检查找不到裸路径下的 K/V 参数。
   现通过模块访问取得参数，保留非零、有限梯度门槛。

复现和修复记录：

- [初始化失败测试](../../outputs/dspark_norm_initialization_red.log)
- [初始化修复测试](../../outputs/dspark_norm_initialization_green.log)
- [最终 10 项测试](../../outputs/dspark_pipeline_component_tests_final.log)
- [最终运行结果](../../outputs/dspark_pipeline_4plus4_20260916_run4/result.json)
- [独立验收](../../outputs/dspark_pipeline_4plus4_20260916_run4/verification-status.json)
- [训练日志](../../outputs/dspark_pipeline_4plus4_20260916_run4/consumer.log)
- [时序与生命周期事件](../../outputs/dspark_pipeline_4plus4_20260916_run4/events.jsonl)
- [CI 恢复记录](../../outputs/dspark_pipeline_4plus4_20260916_run4/ci-restoration.json)

## 验证边界与复现

本轮没有运行跨机 RDMA、CPU→远端 GPU 直传、128K 性能对照或长时间稳定性测试。
内存池及队列容量受到现有预算控制，但本轮的池内预留峰值不等于整机 CPU RSS 峰值。
Checkpoint 的持久化和进度已检查；没有执行训练重启恢复实验。按当前约定，特征
消费后释放，不实现特征恢复、重新生成或故障重放。

复现前预留 8 张 GPU。运行命令见 [README](README.md)，输出目录须使用新名称。
原有文件特征读取路径仍保留，选择原生 recipe 即可使用原流程；归一化初始化修复
同时适用于该路径在当前环境中的 TP 运行。
