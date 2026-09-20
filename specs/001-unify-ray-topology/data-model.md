# Data Model：运行、拓扑、容量与训练提交

本文件定义待实现的逻辑模型；字段进入版本化 JSON/事件，Ray handle 只在运行时存在。配置格式见 [contracts/task-config.schema.json](contracts/task-config.schema.json)，状态协议见 [contracts/runtime.md](contracts/runtime.md)。

## 身份与关系

| 实体 | 关键字段 | 约束与关系 |
|---|---|---|
| TaskConfig | schema_version=3、layout、nodes、inference、training、store、transport、data、timeouts、output_dir | 用户输入；升级后生成 canonical config；旧配置保留原语义并报冲突 |
| Run | run_id、namespace、config_hash、plan_hash、state、reason、created_at、deadline、output_dir | run_id 全局唯一；一个目录仅一个运行；retry 产生新 run_id，本版不从内存状态恢复 |
| NodeSpec / NodeFacts | alias、selector、Ray node_id、IP/hostname、cpu_limit、GPU inventory/UUID、memory snapshot、env/source/model/input digests | alias 只是用户标识，selector 必须唯一解析；IP/hostname 不代替 node_id/设备身份 |
| TopologyPlan | version、layout、config_hash、input_plan_hash、roles、replicas、training_ranks、services、node_budgets、timeouts | 准备后不可变；不含活 Ray handle；实际分配是另一实体 |
| InferenceReplica | replica_id、tp、worker_slots、core_node、core_cpu_bundle、writer_slot、PG declaration | M2 TP8：A 4 slots、B 4 slots；writer=TP0 在 A；所有副本 TP 相同 |
| TrainingParticipant | global_rank、node_rank、local_rank、local_world_size、node_id、TP/DP coordinate、device_slot | global ranks 连续唯一；DP-major/TP-minor；所有参与者使用相同 plan/model/mesh |
| Allocation | allocation_id、run_id、plan_hash、owner、kind、Ray ID、node_id、bundle_index、GPU UUID、created/observed_at、release_state | 同一物理 GPU 只能分给一个角色；PG owner=DeepSpec，vLLM=borrower；精确 ID 回收 |
| ProcessIdentity | node_id、pid、start_ticks、run_id marker、parent/group identity | PID 单独不能证明归属；回收前重验 start-time；不可核验则 unknown |
| NodeBudget | node_id、pool_bytes、writer/reader bounds、prefetch/client bytes、static cap、approved startup budget、retained charge lower bound、remaining bound、observed headroom、snapshot request/sequence/epoch、age upper bound | 覆盖该节点所有角色；启动总量与运行期剩余分配量分开检查；新更新组准入前收齐新鲜节点事实 |
| InputPlan / SamplePlan | hash、teacher identity、position、sample_id、input_identity、length、fields、nbytes、producer replica、reader ranks、update_index | 顺序固定；字节数从实际 tensor shape/dtype 计算；分组不由完成顺序决定 |
| FeatureRecord | run_id、position、state、reservation_bytes、writer、descriptor、claimed/acked readers、delete result | 同一个样本只有一个指定 writer；只接受指定 reader；终态有审计记录 |
| ReaderCopy | run_id、position、reader rank、node_id、nbytes、prefetch/active state、read/retired times | ACK 后独立副本仍可被反向计算持有；副本释放与源对象删除是不同事件 |
| TrainingProgress / Commit | optimizer_steps、native_microstep_cursor、global_sample_cursor、input_plan_hash、teacher identity、checkpoint_path、commit_identity、verified | READ_ACK 不能改变 optimizer_steps；完整提交由原生 checkpoint 协议决定 |
| RunEvidence | schema_version、run_id、plan/allocation refs、per-node/rank events、metrics、verification、cleanup reports | 区分 declared/observed/verified；时钟仅用于本节点耗时，跨节点关联用事件 ID/因果关系 |

## 配置与计划校验

结构层使用 JSON Schema，语义层在 CPU 准备阶段校验：

1. 按 M0–M3 矩阵检查节点数量、角色、每节点额度、TP/DP；M0 允许推理/训练同节点但设备集合互斥；其余角色节点分离。M2 主布局 inference quotas=8/8、DP2，DP1 对照=4/4；M3 推理4、训练4/4。
2. 同一 alias/selector 不得隐含两台或重复同台机器。指定 GPU UUID 时只能使用声明集合；未指定时由 Ray 分配额度内空闲卡，再冻结实际 UUID 映射。资源不足有界失败，不假设连续 GPU 编号。
3. 推理实际占用 GPU=TP×DP，各副本布局完全一致；M2 DP 容量上限=min(floor(G_A/4),floor(G_B/4))。配置额度与实际占用分开：每副本在两端各用4卡，剩余额度不预留、不扩大TP。DP接受容量可满足的正整数；本次DP1/2之外的规模必须单独记录尚未验收，不能冒充已有运行证据。
4. 训练 world=TP×DP=节点本地 GPU 数之和；首版 local_world_size 均匀，CP=PP=1；data_parallel_replicate_degree=1、data_parallel_shard_degree=DP。
5. samples_per_update=4，local microbatch=1；GAS=4/training.dp 必须为正整数。样本计划必须包含完整更新组，每组与每个 DP microstep 都完整。
6. 所有运行等待期限为正数且有限；lease heartbeat 间隔小于 lease 期限。`timeouts_seconds.budget_snapshot` 是预算快照有效期，省略时规范化为5秒，与lease/heartbeat独立；规范化值进入config/plan hash。处理期限与收尾期限分别计时。
7. 验证模型/输入/源码身份、共享输入计划与输出可读写；跨节点服务 endpoint 不得为只在某节点有效的 loopback。
8. v1/v2→v3 的显式字段冲突即拒绝；不静默覆盖 top-level/transport 的不同值。旧 `store.rdma_devices` 原样映射到 `transport.rdma_devices`，与显式新值不同时报错；缺省为空字符串，保留原后端自动选设备语义。未设置的旧 writer-inflight 使用原有 window 所允许的保守上界，不能把旧 batch 隐式压成单请求。

## M2 / M3 身份映射

M2 的两个推理副本分别占 `A:4 + B:4`，每副本 TP ranks 0–3 固定 A、4–7 固定 B；replica 0/1 的实际设备集合互不重叠。两个 CPU core 和 TP0 writer 在 A。训练节点 C 的 local ranks=global ranks 0–7。

M3 的训练映射（GPU slot 是 Ray 本节点分配后的逻辑位置，不是物理卡号）：

| node_rank | local_rank / GPU slot | global_rank | DP rank | TP rank |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 1 |
| 0 | 2 | 2 | 0 | 2 |
| 0 | 3 | 3 | 0 | 3 |
| 1 | 0 | 4 | 1 | 0 |
| 1 | 1 | 5 | 1 | 1 |
| 1 | 2 | 6 | 1 | 2 |
| 1 | 3 | 7 | 1 | 3 |

TP groups 为 [0,1,2,3] 与 [4,5,6,7]；DP shard groups 为 [0,4]、[1,5]、[2,6]、[3,7]。这保持现有 TorchTitan FSDP，不是新建 DDP 两副本训练器。

对位置 p：`producer=p % inference.dp`，`training_dp=p % training.dp`，reader ranks=`[training_dp*4, ..., training_dp*4+3]`。M1 的两种 DP 可以不同，生产归属不用于推导训练归属。

通用训练与核验按冻结计划计算。令 `U=training.steps`、`B=training.global_batch_size=4`、`D=training.dp`，本次运行的计划样本数必须为 `N=U*B`，每rank的 `GAS=B/D`、最终原生微步游标 `M=N/D`、全局样本游标 `N`、更新数 `U`；每个DP组消费其指定的 `N/D` 个样本，完整指定读者校验总数为 `N*training.tp`。第k次更新的预期游标分别为 `k*B/D` 和 `k*B`。verifier须从不可变计划得出预期值，再比对实际事件和原生commit/DCP；不能从待验证checkpoint自身推导预期值。下面的12样本表仅是 `U=3` 的验收实例，`U=5` 时应为20样本、DP1/DP2原生游标20/10、80次读取。

| 12 样本验收 | training DP1 | training DP2 |
|---|---:|---:|
| 每 rank GAS | 4 | 2 |
| 原生同步微步 cursor | 12 | 6 |
| 全局样本 cursor | 12 | 12 |
| optimizer steps / 每 rank 更新数 | 3 | 3 |
| 完整指定读者读取总数 | 48 | 48 |

## 字节预算与释放规则

`S_max` 是输入计划中最大的完整特征大小，不能从平均序列长度估算。对节点 n 定义 pool `P_n`、TP0 writer 数 `W_n`、每 writer 有限 inflight `I`、本地训练 reader 集合 `R_n`、GAS `A`、每 reader 预取字节上限 `F_r`、实际 Store client 数 `C_n`。第一版采用可核验的保守 host 特征上界：

```text
writer_bound_n = 3 * W_n * I * S_max          # raw/converted/pinned 暂存上界
transport_bound_n = W_n * I * S_max          # 注册的异步写 slot
reader_bound_n = sum(2 * A * S_max + F_r for r in R_n)
client_bound_n = C_n * 16 MiB                # 按实际连接/缓冲实现核算
feature_bound_n = P_n + writer_bound_n + transport_bound_n
                  + reader_bound_n + client_bound_n + 1 GiB
static_cap_n = min(0.8 * physical, 0.8 * cgroup_limit, optional_user_feature_cap)
startup_budget_n = min(static_cap_n, headroom_before_feature_allocation - 64 GiB)
# 启动检查：feature_bound_n <= startup_budget_n

remaining_bound_n = max(0, feature_bound_n - retained_charge_lower_bound_n)
# 新更新组准入：feature_bound_n <= min(startup_budget_n, current_static_cap_n)
#             且 current_headroom_n >= 64 GiB + remaining_bound_n
```

`F_r` 缺省为 `prefetch_depth * S_max`，每次 submit 还必须遵守真实 nbytes 上限。保守重叠项允许高估，不能未经所有权/存活期证明就删掉。client buffer 实际尺寸如不同，按当前配置而非固定 16 MiB 计算。该公式约束 host 特征路径，不宣称限制模型 CUDA allocator；模型、KV 与 extraction 的 GPU 占用独立记录和检查，OOM 仍走全任务失败。

`retained_charge_lower_bound_n` 只包含本运行拥有、属于上述组件上界、已证实驻留且已计入本次headroom约束的互不重叠内存下界，例如已实际提交并锁定驻留的池/注册缓冲。它必须在本轮准入组完成前保持有效；释放或替换这些分配须在下一次采样重新核算。配置的pool容量、尚未触页的虚拟分配、可回收页和聚合RSS均不能直接作为抵扣；无法证明的组件抵扣为0。所有约束域（物理内存与有效cgroup）都须支持该下界，不能跨域抵扣。样本源对象在预分配池中删除，不改变池本身的驻留抵扣；ReaderCopy退休也不自动证明OS已释放内存。剩余上界仍包含所有已准入组尚未物化的承诺，不能仅按下一个样本估算。

边界算例（GiB）：headroom初始300、feature_bound=200、保留64，且static cap不构成限制，启动预算236，通过。池已实际驻留并可证明抵扣64后，headroom=236、remaining_bound=136，运行期检查 `236 >= 64+136` 仍通过；若其他进程再占40，headroom=196，则新组等待或有界失败。池只声明64而未实际驻留时，抵扣仍为0。禁止以 `current_headroom-64` 再比较整个feature_bound，重复扣算已有占用。

- M2：pool 只计 C；writer 数 A=2、B=0；训练 readers C=8，A/B=0；DP1 对照 A writer=1。
- M3：pool 只计训练节点0；两个训练节点各4 readers；唯一 writer 在推理节点。不可把全局8 readers 在每节点重复计费，也不可只核算池所在节点。
- 预览检查每个完整更新组的源字节和 window 可容纳、每个节点组件上界能承受该组；run在分配特征内存前重新检查启动预算。运行中每个新更新边界按上述剩余分配量检查headroom，并遵守下述快照有效期。
- `reserve_batch` 可返回可用前缀；如果前缀跨更新边界，必须重新做节点准入检查。已准入组在其已证明的保留额度内完成，不在半个更新处用新的高水位规则让它自锁。
- 源池 reservation 在所有指定 reader ACK 且删除成功后才归还；失败删除保持 reservation。ACK 后的 ReaderCopy 另行占用 reader bound，到消费/反向所需引用释放后才退休。
- 记录 approved/observed 两类值。来自其他进程的总内存压力影响新准入，但不能伪称 OS RSS 精确等于本任务内存（共享页可能重复计算）。

### 预算快照有效期

每次更新组准入请求携带 `run_id/plan_hash/update_index/request_id`。NodeAgent收到请求后现场采样，返回 `node_id/boot_id/agent_epoch/sample_seq`、本地单调采样时刻和预算事实，不返回缓存值冒充新样本。controller保存发出该请求时的本地单调时刻；在最终全节点准入决定时，用同一controller时钟计算 `age_upper_bound=now-request_sent_at`，作为包含往返延迟的保守年龄上界，禁止跨节点时钟直接相减。

全部节点均须满足 `0 <= age_upper_bound < timeouts_seconds.budget_snapshot`，等于有效期即过期；同时要求响应身份/请求匹配、节点epoch未变、序号未倒退。heartbeat只证明存活，不能刷新预算快照年龄。任一响应缺失、延迟、过期或epoch变化即关闭新组准入并重新采样；刷新/容量等待共用从首次等待开始的 `transfer` 总期限，且不超过剩余run期限，重试不重置期限。已准入组继续遵守原批准上界，不能因下一组快照过期而在组内自锁。

## 状态转换

```text
Run: preparing → allocating → initializing → ready → running → draining → succeeded
     任一非终态 → failed / cancelled
Feature: reserved → writing → ready → deleting → released
ReaderCopy: prefetching → materialized_and_acked → active → retired
Allocation: declared → acquiring → acquired → releasing → released / unknown
```

Feature 的 claimed/acked 是按 reader 的集合，与源状态一起校验；没有所有 reader ACK 不能进入 deleting。重复 claim/ACK/write 的已有拒绝语义保留；重试删除只在 deletion manager 的有限策略内进行。失败 Run 关闭所有新准入，不能由后来的某个 rank 成功事件改回 succeeded。

readiness 分两道：allocation gate 收集所有推理 worker shells 与训练 launchers 的物理身份；initialization gate 收集所有推理/训练 ranks、存储与通信就绪。均比较 plan_hash 和预期集合，重复相同上报可幂等，冲突上报致命失败。

## 持久化与并发

`config.normalized.json`、`plan.json`、原生 input-plan 固定；`allocation.json` 记录运行事实；`status.json` 由 RunController 原子更新；事件按 component/node/rank 分文件追加，每个文件一个 writer。checkpoint 仍由原生保存流程负责；`verification.json` 是独立检查结果，不取代原生 commit。

节点 watchdog 在 driver 丢失后只写自己的 `cleanup/node-<node_id>.json` 与 orphan-failure 记录，避免与 controller 竞争 status 文件。`status` 汇总命令在 lease 过期后报告 failed、cleanup confirmed/unknown；若 driver 恢复联系，已过期的 run 被 fence，不能继续提交或复活。该 fence 用于终止一致性，不提供恢复。
