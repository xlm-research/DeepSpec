# 原生 vLLM Ray 放置研究

日期：2026-09-20。范围：方案 B 的 M2 TP8×DP2（两推理节点各 8 GPU，每副本各节点 4 GPU）、TP8×DP1 对照，以及 M0/M1/M3 的原生后端兼容。本文依据当前工作区源码；已安装 vLLM 元数据版本为 `0.26.1rc1.dev719+g1ee54c40d.d20260912`，运行环境的 PYTHONPATH 优先使用本仓库 vLLM。未启动 GPU、Ray actor 或服务，源码可行性不代表多节点验收通过。

## 决定

采用 **DeepSpec 统一创建并持有精确 Placement Group，原生 AsyncLLM / CoreEngineActorManager / RayExecutorV2 借用这些 PG**。M0/M1/M2/M3 以及 DP1 对照统一资源所有权路径；保留 vLLM 原生 DP 请求路由、TP worker、模型执行、隐藏状态提取。增加三处边界适配：PG 参数透传、CPU EngineCore 跳过本机 GPU 区间推导、V2 在 worker 初始化前报告并校验实际分配。不上自建模型执行器，不使用全局 monkey patch。

项目已显式启用原生 V2 Ray executor：[run.py:36](../../deepspec/pipeline/run.py#L36)。因此新增初始化门禁落在 V2，而不是切换到旧 Ray executor。

## 为什么现有 native DP 策略不满足 M2

- 当前 producer 把 `tensor_parallel_size` 写死为 4；DP>1 才用 AsyncLLM，配置 `data_parallel_size_local=DP`：[actors.py:106](../../deepspec/pipeline/actors.py#L106)、[actors.py:209](../../deepspec/pipeline/actors.py#L209)。启动环境还写死了 `VLLM_RAY_BUNDLE_INDICES=0,1,2,3`；TP8 时会触发 indices 数量校验，必须随计划生成。
- native `strict/fill` 使用单节点 `STRICT_PACK`；`span` 要求 `world_size > max_device_per_node`。M2 的 TP8 与单节点 8 卡相等，直接不满足 span 条件；即使绕过条件，普通 `PACK` 也不保证每副本 4+4：[utils.py:581](../../vllm/vllm/v1/engine/utils.py#L581)。不能依赖节点白名单加默认调度获得规格承诺。
- 外层预留整块 GPU 后再让 native DP 另建 PG 会等待已经被自己占有的资源。现有代码为 DP>1 特意不创建 producer PG：[cluster.py:558](../../deepspec/pipeline/cluster.py#L558)。新实现必须借用同一组 PG，不能保留两套申请。

## PG、控制进程与所有权

M2 的每个 DP 副本建立一个 PG，按下列顺序排列九个 bundle，使用 `PACK` 配合每个 bundle 的硬节点资源约束；这里正确性来自节点约束，不来自 PACK 的偏好：

```text
bundle 0..3: {GPU: 1, node:<A 的 Ray IP>: 0.001}
bundle 4..7: {GPU: 1, node:<B 的 Ray IP>: 0.001}
bundle 8:    {CPU: 1, node:<A 的 Ray IP>: 0.001}
```

DP2 创建两组，A/B 各占 8 GPU；DP1 创建一组，各占 4 GPU。每个 GPU bundle 必须恰好 1 GPU，不能用 `{GPU: 4}` 代替四个 TP worker bundle；原生校验拒绝单 bundle 超过 1 GPU：[ray_utils.py:605](../../vllm/vllm/v1/executor/ray_utils.py#L605)。M0/M1/M3 每副本为四个 1 GPU bundle 加一个 CPU core bundle，按各自计划绑定节点；训练 launcher 的资源由训练侧单独持有。

CPU EngineCore 固定在其副本最后的 bundle，native manager 就是以 `bundle_index=world_size` 启动 core：[utils.py:483](../../vllm/vllm/v1/engine/utils.py#L483)。Producer frontend 自己需要另一个明确计入 CPU 预算的资源槽，固定 A，`num_gpus=0`，不能占用仅有 1 CPU 的 core bundle 后等待 core 启动，也不能捕获子任务到训练 PG。前端可用独立 CPU PG；两份 GPU PG 的 CPU core bundle 只供 native core 使用。

协调器先记录本运行 PG handle/ID/名称、创建责任及计划 hash，再等待所有 PG ready；部分分配、等待超时或初始化失败均按注册表回收。名称带 run_id，清理使用精确 handle/ID，不扫描通用 `dp_rank_*` 名称。native manager 在传入 PG 的分支将 `created_placement_groups=[]`，退出只删除它自己创建的 PG；这是已经存在的借用语义：[utils.py:440](../../vllm/vllm/v1/engine/utils.py#L440)、[utils.py:995](../../vllm/vllm/v1/engine/utils.py#L995)。DeepSpec 负责停止 native engine/worker 图，再回收借出的 PG 并核实释放。无论 M0 还是 M2，都使用这一路径。

## 三处最小 vLLM 接入边界

1. **PG 透传。** 在 `ParallelConfig` 中加入仅运行期使用的 `ray_placement_groups`、`ray_placement_local_dp_ranks` 及 `ray_placement_plan`（含版本、run_id、plan_hash、分配门禁 endpoint、每副本预期 bundle/节点）。前两个是类型明确的 Ray handle 列表和整数列表；计划持久化 JSON 只记录 ID/声明，不序列化活 handle。完成 `AsyncEngineArgs.create_engine_config()` 后由 DeepSpec adapter 绑定这组运行参数，调用已经存在的 `AsyncLLM.from_vllm_config()`：[async_llm.py:206](../../vllm/vllm/v1/engine/async_llm.py#L206)。`launch_core_engines()` 把 PG 列表及 local ranks 传给现成的 `CoreEngineActorManager` 参数；当前该调用没有传这两个参数：[utils.py:1121](../../vllm/vllm/v1/engine/utils.py#L1121)。验证列表数等于 DP、每 PG 的 GPU/CPU bundle 和节点匹配计划、不能与自动放置参数冲突。运行期字段从图编译 hash 排除，沿用现有 `placement_group` 排除方式：[parallel.py:774](../../vllm/vllm/config/parallel.py#L774)。DP2 的 CPU cores 均在 A，`data_parallel_size_local=2`、`local_dp_ranks=[0,1]` 表示 core 本地身份，不能解释为各副本全部 GPU 都在 A。DP1 对照传一组 PG、local ranks `[0]`。`launch_core_engines` 按 `backend=ray` 分支，没有 `DP>1` 限制，因此 DP1 可以用同一入口。读取参数后核对最终 config，拒绝继承环境变量把 DP、地址或 executor 悄悄改掉。
2. **CPU core 不推导 TP8 本机卡区间。** `EngineCoreActorMixin` 当前按 `local_dp_rank * world_size` 推导一整副本的本机卡号：[core.py:2392](../../vllm/vllm/v1/engine/core.py#L2392)、[utils.py:307](../../vllm/vllm/v1/engine/utils.py#L307)。在借入 PG 且使用 native Ray 的明确分支，CPU core 跳过这段 GPU 区间推导，保留 local DP identity/通信信息；实际映射完全交由 Ray worker 资源分配结果。否则 TP8×DP2 的第二个 CPU core 会尝试把 A 的本机卡 8..15 当作自己的范围。原生 V2 已经在发现实际物理 GPU 后按每节点映射设置 worker config：[ray_executor_v2.py:406](../../vllm/vllm/v1/executor/ray_executor_v2.py#L406)、[ray_executor_v2.py:159](../../vllm/vllm/v1/executor/ray_executor_v2.py#L159)。无需给 core 申请 GPU，也不能伪造 CUDA_VISIBLE_DEVICES。
3. **模型初始化前的实际分配门禁。** V2 Step 5 创建的 worker 只是延迟初始化 actor，Step 6 获取真实 `(node_id, physical_gpu_ids)`，Step 7 才执行 `initialize_worker()`：[ray_executor_v2.py:105](../../vllm/vllm/v1/executor/ray_executor_v2.py#L105)、[ray_executor_v2.py:350](../../vllm/vllm/v1/executor/ray_executor_v2.py#L350)。在 Step 6 完成后、Step 7 发出任何初始化请求前，将实际 rank/bundle/node/GPU 报告给运行协调器，做有界校验与放行；协调器先收齐全部推理副本和训练 launcher 的分配事实，检查 GPU 唯一、角色隔离、每副本 4+4、节点数量/配额、run_id/plan_hash 一致，再放行 GPU 初始化。任何一方失败或超时须唤醒其余等待方，所有远程 waits 使用配置中的有限期限。现有 PG 校验只警告跨节点并检查 driver 是否属于 PG，不满足完整门禁：[ray_utils.py:290](../../vllm/vllm/v1/executor/ray_utils.py#L290)。实现门禁回调应封装在 adapter，vLLM 只承载通用运行期计划/报告入口，不能把训练业务嵌进 executor。

分配门禁之后才是初始化门禁：收齐所有 TP worker 的真实分布式身份、所有训练 rank、存储可达性和通信就绪，才进入全运行 ready。当前 connector 的 `producer_worker` 事件发生在 cache 注册时，适合作为初始化后的第二次核验，不能代替 GPU 初始化前检查：[connector.py:58](../../deepspec/pipeline/connector.py#L58)。

## TP rank、writer 节点与字节预算

M2 固定 bundle indices `0..7`（其他布局按 TP4 为 `0..3`），由计划生成并检查外部覆盖。V2 明确支持 bundle indices 并据此分配 rank：[ray_executor_v2.py:304](../../vllm/vllm/v1/executor/ray_executor_v2.py#L304)、[ray_utils.py:364](../../vllm/vllm/v1/executor/ray_utils.py#L364)。默认路径会按 core 所在节点优先排序，所以也不能假设物理 GPU 编号就是 TP rank；显式 indices、实际 node/GPU 报告、初始化后 `producer_rank/tp_rank` 三者一起核验。计划使两个副本的 TP0 writer 都位于 A。保持 DP identity suffix 防止 worker/connector 名称冲突：[utils.py:355](../../vllm/vllm/v1/engine/utils.py#L355)。

TP8 不要求新增隐藏状态 gather 或改训练特征形状。原生提取路径已有完整 `[T, L, H]` 堆叠：[extract_hidden_states.py:124](../../vllm/vllm/v1/spec_decode/extract_hidden_states.py#L124)；cache-only layer 使用模型完整 `hidden_size`：[extract_hidden_states.py:345](../../vllm/vllm/model_executor/models/extract_hidden_states.py#L345)；只有 TP0 发起 D2H/写入：[example_hidden_states_connector.py:351](../../vllm/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py#L351)。项目严格检查完整 BF16 shape、token IDs 和有限值：[qwen3_8_vllm.py:106](../../deepspec/trainer/qwen3_8_vllm.py#L106)。

因此每样本特征字节数由 token 数、层数、hidden size 与 dtype 决定，不因 TP4→TP8 而除二或乘二；特征仍由其 DP 副本唯一 TP0 写一次。**两个 writer 均在 A，A 的生产暂存、D2H pinned copy、转换后独立副本、writer 在途/异步传输额度必须聚合 DP2**，不能按推理 GPU 的 4+4 拆成两节点各一半。B 仍按其实际工作记录基础/模型 CPU 内存；特征池归属和 consumer 读取副本另按真实节点计费。TP8 的模型通信兼容性及数值偏差仍须通过真实 4K/128K 验收；不得把同模型 TP4/TP8 的浮点输出要求为逐 bit 相同。

## 项目修改位置及验收重点

| 文件/模块 | 计划修改 |
|---|---|
| `deepspec/pipeline/topology.py`、`schema.py` | 声明 `producer_tp` 与 per-replica per-node 分片；DP 从所分配额度计算且校验，不硬编码仅 DP1/2；M2 保持 TP8。 |
| `deepspec/pipeline/run.py`、`cluster.py` | 统一 PG 所有权；动态 bundle indices；移除按每副本只在一个节点的推断；计划驱动环境、逐节点 budget 和证据核查。 |
| 新 DeepSpec vLLM adapter | 绑定 runtime PG/计划参数；统一 DP1/DP2 AsyncLLM 构建、分配报告、初始化、失败、关闭协议。 |
| `deepspec/pipeline/actors.py` | TP 从计划读取；DP1/DP2 使用统一 adapter，同时保留既有 DP1 `producer_batch_size`/`reserve_batch` 准入含义，不能直接把它换成每 DP 永远只允许一个请求的当前 async semaphore。 |
| `deepspec/pipeline/connector.py`、buffer/预算模块 | 核验所有 DP/TP 身份及实际 writer 节点；仅 TP0 写入；按 writer 实际节点累加暂存，保留 byte reservation、全读者 ACK、成功删除才退额度。 |
| `vllm/vllm/config/parallel.py`、`vllm/vllm/v1/engine/utils.py` | 运行期字段及 native manager 参数透传；字段校验、hash 排除及借用生命周期。 |
| `vllm/vllm/v1/engine/core.py` | 借 PG/native Ray 分支的 CPU core 跳过 GPU 区间推导。 |
| `vllm/vllm/v1/executor/ray_executor_v2.py` | 延迟 worker 初始化前的实际分配报告与有界 gate；失败时撤销本次 workers。 |

验证顺序：无 GPU 的配置/PG 形状/所有权/错误注入测试 → 轻量 Ray worker 实际分配检查（模型初始化方法不得被提前调用）→ M0/M1 既有完整回归 → M2 TP8×DP1 与 TP8×DP2 的实际三节点 GPU 验收，并由 M3 验证训练跨节点。M2 重点检查每副本 4+4、两个 writer 所在节点及预算、四卡额度不使用额外未分配 GPU、CPU bundle 不互相等待、部分 native core 创建失败时借出 PG 都能回收、128K 特征维度及 12 样本/3 更新/48 次读取证据。报告必须区分源码审查、模拟测试、轻量 Ray 分配和真实模型训练。
