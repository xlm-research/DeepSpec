# Phase 0 Research：统一 Ray 拓扑与生命周期

日期：2026-09-20。输入：[已澄清规格](spec.md)。本文是设计决策，不是多节点运行结果。前次 [TorchSpec 比较](../../.specify/assessments/torchspec-ray-management/comparison.md)提供组织方式参考；本次以规格 M0–M3 为准。源码摘要见 [source-manifest.json](source-manifest.json)。

## 研究范围与证据

使用已安装 specify-cli 的 plan 流程，分派原生 vLLM 放置与 TorchTitan 多节点训练两项研究；分别保留 [推理证据](research-vllm.md)与[训练证据](research-training.md)。本地 Python 包元数据核验：Python 3.12.14、torch 2.13.0、Ray 2.58.0、vLLM 0.26.1rc1.dev719+g1ee54c40d.d20260912、Mooncake 0.3.13.post1、Transformers 5.16.1、pytest 9.1.1。执行时仍须验证每个节点加载的源码与二进制一致；包版本不能代替工作区摘要。

不存在项目 constitution 或 extension hooks。按规格的后端、数学、预算、资源隔离、有限失败与兼容性约束执行前置检查，无设计冲突。模板脚本成功解析功能目录但未找到模板；已使用 `specify preset resolve plan-template` 确认的 CLI 内置模板补齐。

## R1：一个不可变计划，分层分配资源

**Decision**：配置 v3 → 规范化任务 → `TopologyPlan` → `Allocation`。计划包含逐节点角色额度、每副本 TP 分片、训练 rank、服务位置、读者映射、字节预算及期限。实际 Ray 节点与 GPU 身份必须另行记录并核对。旧 v1/v2 配置与 shell 入口转入同一规划/控制流程。

**Rationale**：当前 `schema.py` 只升级 transport；`cluster.py` 固定两个节点、`consumer_nodes=1`，`run.py` 又有单机启动与资源清理。仅放宽数字会留下角色落点、预算、日志命名和 reader 映射的隐含假设。

**Alternatives considered**：复制一份三节点 launcher 会继续分叉；整个任务一个 PG 会混淆 vLLM worker 与训练 launcher 的 bundle 粒度。采用每推理副本一个 PG、每训练节点一个整卡 bundle，统一登记所有权和失败回滚。单个 actor 需要的资源必须由一个 bundle 容纳，不能把 8 卡 Consumer 放到八个单卡 bundle 上。[Ray placement groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)

## R2：M2 显式 4＋4 放置，保留原生 vLLM Ray V2

**Decision**：每个 TP8 副本建立 8 个单 GPU bundle，前 4 个固定推理节点 A、后 4 个固定 B；另设 A 上的 CPU engine-core bundle。项目唯一拥有并释放这些 PG，vLLM 借用。frontend 单独申请计划内 CPU，不占 engine-core 的 CPU bundle。DP1/DP2 使用同一 `AsyncLLM.from_vllm_config` 接入路径。配置 `VLLM_RAY_BUNDLE_INDICES` 按 TP 生成；训练资源始终独立。

**Rationale**：默认 strict/fill 会把一个副本放在单节点；span 对 TP8 和每节点 8 GPU 的组合也不能表达每节点只取 4 卡。`CoreEngineActorManager` 已能接受 `placement_groups/local_dp_ranks`，但调用链未透传。需要三处窄范围接入：运行时 PG 透传；借用 Ray PG 时 CPU engine core 不再按 `local_dp_rank * world_size` 猜本机 GPU；Ray V2 获得真实设备后、初始化设备/加载权重前执行 allocation gate。后者位于 `ray_executor_v2.py` 的 GPU IDs 收集与 `initialize_worker` 之间，具体位置见推理证据。

**Alternatives considered**：默认 PACK、用占位 actor 抢走额外 GPU、重复外层与内层预留都不能满足隔离与可判定性；改 MP executor 会偏离 FR-019。配置扩展只传运行时 PG 句柄和计划身份，不把 Ray 对象序列化进持久化 JSON。缺少所需接入能力时预检报 `BACKEND_CAPABILITY_MISSING`，不退化为错误拓扑。

**迁移约束**：M0/M1/M3 的 TP4 副本改用相同资源所有权规则。保留旧 DP1 的 batch、window、writer-inflight 行为；不能直接用现有单请求 semaphore 覆盖批量参数。保留 V2 Ray executor 与原 hidden-state runner 的选择。

## R3：TP8 特征格式保持，writer 预算按真实位置累加

**Decision**：保留完整 BF16 hidden-state 特征及现有 identity/shape/token 校验；特征字节数不因 TP4→TP8 自动乘二或除二。两个 CPU core 均固定 A，V2 的本地优先排序使每副本 TP0 writer 位于 A；在 allocation gate 与 connector 事件中再次验证。

**Rationale**：本地 hidden-state extractor 提供完整 `[T,L,H]`，只有 TP0 发布。M2 的 A 有 2 个 writer，B 无 CPU 特征写入者。写入暂存、注册 host buffer 与在途写预算应在 A 累加；B 仍需计入模型执行及 extraction 的 GPU 空间。

**Alternatives considered**：按两个节点平均分写缓存会漏算 A 的峰值；为了 TP8 重定义分片特征会不必要地改变训练数据契约。源码支持是实现依据，仍需真实 TP8 shape、完整校验、数值回归和 writer 落点验证。

## R4：M3 每节点 launcher，原生 TorchTitan 分片 DP

**Decision**：两个 Consumer 各申请本节点 4 GPU，分别以固定 `nnodes=2`、`nproc_per_node=4`、node_rank 0/1 启动 torchrun；静态 rendezvous、同一训练节点 0 地址/端口、`max_restarts=0`。全局 world=8、TP4、`data_parallel_shard_degree=2`、CP=PP=1。

**Rationale**：现有 `consumer_command` 已有多节点分支，TorchTitan mesh 为 DP-major/TP-minor。训练节点 0 的 ranks 为 0–3，节点 1 为 4–7；跨节点 DP 组是 (0,4)、(1,5)、(2,6)、(3,7)。这里保留当前 FSDP/分片 DP，不能误改为 DDP。[torchrun 官方说明](https://docs.pytorch.org/docs/stable/elastic/run)

**Alternatives considered**：每 rank 一个 Ray actor 需要重建训练初始化边界；引入 TorchSpec trainer 会改变既有训练路径。静态 node_rank 可复现映射，不开放 elastic membership。

**Readiness**：把当前 `consumer_ready` 布尔开关升级为全 rank 身份握手；现有全局 barrier 保留，但节点/设备/TP/DP/plan hash 必须在准入前校验。rendezvous 的端口先在训练节点 0 选定，启动时占用冲突有界失败；不默默更换正在运行任务的 endpoint。

## R5：样本游标与 checkpoint 使用原生单位

**Decision**：producer replica=`position % inference.dp`；训练 DP=`position % training.dp`；指定读者为该训练 DP 组内全部 4 个 TP ranks。两种 DP 独立。12 样本、global batch 4、local microbatch 1：DP2 的 GAS=2，native microstep cursor=6，global sample cursor=12，optimizer steps=3；DP1 的 GAS=4、native cursor=12。所有布局都是 48 次指定读者完整校验。

**Rationale**：`consumer_microbatches` 已说明原生游标单位。`PhaseCheckpointer` 保留 DCP、同步、rank0 commit 与所有 rank 的 read_commit 流程。需要共享输出路径和独立通用 verifier，不能复用单机 debug verifier 中 4 ranks/4 shards/cursor=样本数的假设。

**Alternatives considered**：读完 ACK 当 optimizer commit 会错误宣告训练完成；硬编码 DCP shard 数不等价于 rank 数或完整状态。M1 的 producer DP2/consumer DP1 目前被 CLI 拒绝，虽然规格要求覆盖四种组合；计划显式解除该人工耦合并新增回归，不声称四种组合已经跑过。

## R6：单池、多节点预算、完整更新组准入

**Decision**：首版保留一个 TCP/CPU 特征池，放在第一个训练节点；默认同节点运行任务自建 Mooncake master，也允许显式借用可跨节点访问的外部 master。所有节点使用实际可路由地址。元数据走 Ray，大张量走 Mooncake。

逐节点预算由实际 pool、writer 数、local reader 数、GAS、prefetch depth/bytes、writer-inflight、注册缓冲和安全余量推导；M3 只在训练节点 0 计一次 pool，每个训练节点分别计 4 个 reader，不能在每节点重复计 world=8，也不能漏掉远端读者。沿用 `memory.py` 对物理内存/cgroup/headroom 的保守上界；新的完整组件公式见 [数据模型](data-model.md)。

每个 optimizer 更新组第一次准入时收齐相关节点的新鲜预算；随后保证已准入组可以推进。批量 reserve 遍历到下一个更新边界时必须重新检查，不能只检查 batch 首位置。读者尚未 ACK、源删除尚未确认与训练侧仍持有的副本分别计账；源池额度归还不等于节点内存归还。

**Rationale**：现有 ledger 的顺序、bytes/window、all-reader ACK、删除确认已形成有效契约。`reserve_batch` 的后续 prefix 和现有两节点预算映射需要随通用拓扑校正；不以解除背压解决拓扑问题。

**Alternatives considered**：多池会增加对象定位与删除归属状态，首版无必要；按样本数代替 bytes 会破坏 128K 上界；RDMA/GPU 直收优化另行评估。

## R7：统一生命周期与独立于阻塞 RPC 的监督

**Decision**：普通 driver `RunController` 编排三个深接口：InferenceGroup、TrainingGroup、StoreService；少量 NodeAgent 负责节点事实、进程归属和本地 lease watchdog。每次运行的 registry 记录 owner、资源 ID、节点、PID/start-time、创建/回收状态。Ray actors 不自动重启，PG 不设 detached。

heartbeat/stop 不能与长时间的 `run()` RPC 共用单线程队列。Consumer 启动子进程后返回 handle，由独立监督路径观察；现有 `deepspec.orchestration.process` 的 subreaper 与父进程退出保护继续复用。NodeAgent 的本地 watchdog 在父 actor/driver 死亡或 lease 超时时清理登记的本次进程并写节点报告；不能只依赖 driver 的 finally。Ray 默认 PG 生命周期与创建者关联，但资源释放异步，仍须核验。[PG 生命周期](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.placement_group.html)

终止先关准入、通知所有等待方，停止模型组，释放源对象与连接，再回收 PG 和自建服务；外部服务只断开本任务连接。不可达或权限不足无法核实的资源记为 unknown，不把其计为 cleaned。取消优先于后续清理噪声，保留首个致命原因。所有期限有限并写入计划。

**Alternatives considered**：只加 try/finally 无法覆盖 driver SIGKILL；通配 pkill 或集群 ray stop 会影响其他任务；detached actor 加无限重试会形成新恢复系统，超出范围。

## R8：版本化 CLI/配置契约和证据链

**Decision**：新增 `python -m deepspec.pipeline.cli` 的 preview/run/transport-check/status/cancel/verify 子命令，旧入口转接。提供结构 JSON Schema、M2/M3 示例与跨字段语义规则。preview 无 GPU 预留、无模型和大池；run 重验环境与空闲额度，处理预览后的资源变化。

准备配置/输入计划/拓扑 hash 不变；实际分配、事件、节点状态、训练提交与清理报告分开存放。单一写入者更新每份 JSON，按 run/node/rank 命名事件文件，避免两个 consumer 覆盖同一文件。正常、失败、取消、driver 丢失都有可核查记录。

**Alternatives considered**：新增 HTTP 服务或数据库没有必要；直接覆盖旧 pipeline.json 又把运行时字段加进去，会使不可变计划与实际状态混淆。

## Phase 0 结论

技术选择均已确定，无待用户回答的设计问题。剩余工作是按上述接入点实现并取得运行证据：尤其是借 PG 的 vLLM V2 路径、TP8 特征、跨节点 FSDP、逐节点容量与 driver 丢失清理。没有进行 GPU 启动、跨节点训练或吞吐测量，也没有把这些验证风险标成已通过。
