# Phase 0 research: 跨节点 draft 训练

日期：2026-09-20。本文依据当前工作区源码进行只读研究；没有启动 Ray、Mooncake、torchrun 或 GPU，也没有执行下述测试。源码位置对应研究时的工作区，不把历史运行记录当作本次 M3 验收。

## Decision

M3 使用一个三节点 Ray 集群：推理节点分配 4 GPU，保留原生 vLLM TP4×DP1；两个训练节点各分配 4 GPU，每节点一个 Consumer launcher，两个 `torchrun` agent 组成同一个 world size 8 的 TorchTitan TP4×DP2 实例。每个 TP4 组留在本节点，DP collective 跨训练节点。M2 仍是两个推理节点各 8 GPU、推理 TP8×DP2，以及单训练节点 8 GPU、训练 TP4×DP2；M3 不改变 M2 的 TP8 选择。

DP2 保留当前 `data_parallel_shard_degree=2`、`data_parallel_replicate_degree=1`，即原生 FSDP 分片 DP。不能把“DP2”解释成改用两个独立训练器或替换为 DDP。输入计划、DSpark loss、梯度累积、优化器、TorchTitan mesh 和完整 DCP 均继续由现有原生实现负责。

依据：[recipe.py:14](../../deepspec/pipeline/recipe.py#L14) 设置 shard degree 和 TP4；[config_registry.py:156](../../torchtitan/torchtitan/models/dspark_draft/config_registry.py#L156) 定义 TP4 配方；[parallelize.py:72](../../torchtitan/torchtitan/models/dspark_draft/parallelize.py#L72) 使用原生 dp_shard/FSDP mesh 并调用 `fully_shard`。

## 已存在的能力与真正缺口

| 当前事实 | 设计含义 | 源码 |
| --- | --- | --- |
| `consumer_command` 已支持 `nnodes`、`node-rank`、共享 master 地址/端口、run ID 和 `max-restarts=0`，local world 由 global world 除节点数得出 | 扩展现有 launcher；无需把每个训练 rank 改为 Ray actor | [actors.py:51](../../deepspec/pipeline/actors.py#L51) |
| Consumer 验证 Ray 分给自己的设备数，设置 CUDA_VISIBLE_DEVICES，并通过 `run_owned` 启动 TorchTitan | 每个训练节点只申请自己的 4 GPU；保留本地子进程监管 | [actors.py:268](../../deepspec/pipeline/actors.py#L268) |
| `cluster.py` 明确写死 `consumer_indices=[1]` 和 `consumer_nodes=1` | 有命令构造能力不等于当前入口可运行 M3；须由拓扑计划给出两个训练节点 | [cluster.py:437](../../deepspec/pipeline/cluster.py#L437) |
| consumer PG bundle 和 actor 每个都申请 `consumer_world_size` GPU | 改为节点局部 GPU 数，避免在 M3 的两节点各申请 8 卡 | [cluster.py:561](../../deepspec/pipeline/cluster.py#L561)、[cluster.py:629](../../deepspec/pipeline/cluster.py#L629) |
| NodeMonitor 用角色字符串区分预算与文件名，按全局 consumer world 算每节点读者数 | 改用唯一 node/role 实例标识和每节点 local readers；避免两个 consumer 写同一文件、重复计池或全局读者暂存 | [cluster.py:240](../../deepspec/pipeline/cluster.py#L240)、[cluster.py:278](../../deepspec/pipeline/cluster.py#L278) |
| `consumer_nodes` 旧配置默认值为 `consumer_dp`，新入口则显式覆盖为 1 | 兼容适配必须保留旧 saved run 语义；新拓扑统一显式设置节点数，不能靠默认值推断 | [topology.py:23](../../deepspec/pipeline/topology.py#L23) |

历史文档记录过“两物理节点、生产与部分训练混部”的 12 GPU 跨节点 FSDP 运行，说明原生训练及协议有既有基础；该布局不满足 M3 的三节点角色隔离，当前 launcher 也已改成单训练节点，所以仍须按新入口独立验收。参见 [VALIDATION.md:114](../../deepspec/pipeline/VALIDATION.md#L114)。

## TorchTitan mesh 与 exact rank 表

原生 dataloading mesh 的轴顺序为 `(pp, batch, cp, tp)`；storage mesh 为 `(pp, dp_replicate, dp_shard, cp, tp)`。M3 的形状是 `(1, 2, 1, 4)` 与 `(1, 1, 2, 1, 4)`，即 DP-major、TP-minor。由此 `global_rank = node_rank * 4 + local_rank`，`dp_rank = global_rank // 4`，`tp_rank = global_rank % 4`；`LOCAL_RANK` 是 launcher 可见设备列表中的索引，不能直接当成整机物理 GPU 编号。依据 [parallel_dims.py:220](../../torchtitan/torchtitan/distributed/parallel_dims.py#L220)、[trainer.py:361](../../torchtitan/torchtitan/trainer.py#L361)。

| 训练节点 | node rank | local rank | global rank / reader ID | DP shard rank | TP rank | 样本 position |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T0 | 0 | 0 | 0 | 0 | 0 | 0,2,4,6,8,10 |
| T0 | 0 | 1 | 1 | 0 | 1 | 0,2,4,6,8,10 |
| T0 | 0 | 2 | 2 | 0 | 2 | 0,2,4,6,8,10 |
| T0 | 0 | 3 | 3 | 0 | 3 | 0,2,4,6,8,10 |
| T1 | 1 | 0 | 4 | 1 | 0 | 1,3,5,7,9,11 |
| T1 | 1 | 1 | 5 | 1 | 1 | 1,3,5,7,9,11 |
| T1 | 1 | 2 | 6 | 1 | 2 | 1,3,5,7,9,11 |
| T1 | 1 | 3 | 7 | 1 | 3 | 1,3,5,7,9,11 |

TP groups 为 `[0,1,2,3]`、`[4,5,6,7]`；跨节点 DP groups 为 `[0,4]`、`[1,5]`、`[2,6]`、`[3,7]`。M2 同样的训练 global rank 与 DP/TP 坐标放在一个节点上，其 local rank 是 0–7，不能把 DP rank 永久等同于 node rank。

`sample_readers(position)` 返回 `(position % DP)` 所属的四个 TP ranks；loader 按 `entries[dp_rank::dp_world_size]` 消费并验证原生 DP rank 等于 global rank 整除 TP。参见 [topology.py:31](../../deepspec/pipeline/topology.py#L31)、[data.py:28](../../deepspec/pipeline/data.py#L28)、[原生 data.py:62](../../torchtitan/torchtitan/models/dspark_draft/data.py#L62)。

全局每次更新仍为 4 个样本，每 DP rank 的 microbatch 为 1 个样本，因此 GAS=2。12 个不同样本对应 6 个同步 DP microsteps、3 次更新，以及 `12×4=48` 次指定 reader 校验；不是 96 次。checkpoint 的 `next_global_microbatch=6`，换算全局样本位置为 `6×2=12`。依据 [topology.py:38](../../deepspec/pipeline/topology.py#L38)、[trainer.py:171](../../deepspec/pipeline/trainer.py#L171)。

## Launcher、rendezvous 与 readiness 方案

拓扑解析后按稳定 node rank 为两个训练节点各创建一个 launcher，资源配额是 4 GPU 和配置声明的 CPU。launcher 使用相同的共享配置、run ID、global world=8、nnodes=2、nproc-per-node=4、master 地址与端口；仅 node rank、节点设备和本地网络接口不同。保留现有 static torchrun 路径和 `--max-restarts=0`；run ID 不是端口隔离的替代品。由 T0 的管理 actor 在所选可达接口上分配并登记本轮 rendezvous 端口，端口在 torchrun 真正绑定前存在竞争，必须将冲突计为有界启动失败并整体回滚，不能“找到空闲端口”后视为已经就绪。现有 `rendezvous_port` 只是 `free_port()`，参见 [cluster.py:300](../../deepspec/pipeline/cluster.py#L300)。

初始化前核验三节点 Python/源码/依赖、teacher/input plan/config 身份和共享存储访问一致。跨节点训练沿用当前支持的 Socket 路径及节点本地 NCCL/Gloo interface 映射；RDMA 优化不作为本次前提。配置的 rendezvous、distributed initialization、运行和清理阶段都需有限 deadline；当前 recipe 的 600 秒通信超时只是现有默认，不能代替完整生命周期期限。

当前 `StreamingDSparkTrainer` 先提交 rank initialized event，再执行全 world barrier，最后由 global rank 0 调用 `consumer_initialized`。这已避免正常路径中单个 rank 未到 barrier 就开闸，但 buffer 只持有一个布尔值，且 rank/node/TP/DP 对照在整个训练结束后的 `summarize_events` 才执行。参见 [trainer.py:26](../../deepspec/pipeline/trainer.py#L26)、[buffer.py:210](../../deepspec/pipeline/buffer.py#L210)、[run.py:276](../../deepspec/pipeline/run.py#L276)。

设计改为每个训练 rank 提交结构化握手：run/plan 身份、node ID、node rank、global/local rank、world/local world、GPU 映射、TP/DP 坐标、GAS，以及需要对齐的训练身份。协调器收齐计划中的 8 个不同 rank、逐项校验并收到全部必需推理角色就绪后，才允许本轮从 initializing 进入 ready 并开放正常生产 admission。重复、缺失、错节点或不匹配握手均不能使 readiness 成立；有限等待到期或任何必需 rank 失败使整组失败。原生 barrier 继续作为 distributed initialization 的屏障，不承担管理平面的身份校验职责。

## 字节预算、读取完成与训练提交

M3 两个训练节点各按 local readers=4、GAS=2 计算接收/校验/prefetch 暂存；只有实际放置 Mooncake CPU pool 的节点计 pool 容量，其他节点不能重复计为本地已预留，也不能因没有本地池忽略自己的客户端和读缓冲。生产节点独立计 writer staging/in-flight。预算必须来自统一拓扑中的实际 placement；当前 `feature_budget` 已区分 pool/readers/writer，但调用侧仍使用全局读者数，参见 [memory.py:42](../../deepspec/pipeline/memory.py#L42)。

每个样本仍只在四个指定 TP readers 拿到校验过的独立副本后删除源对象，成功删除后才返还 admission credit。M3 中这四个读者在同一个训练节点上是合法结果；其 ACK 只证明本样本的源缓冲可释放，不能证明跨节点优化器更新已完成。完整更新至少需要全体 8 ranks 的对应 update 事件以及一致的 microbatch/sample cursor；成功终态还需要完整 DCP commit 和清理证据。依据 [buffer.py:105](../../deepspec/pipeline/buffer.py#L105)、[trainer.py:41](../../deepspec/pipeline/trainer.py#L41)、[run.py:253](../../deepspec/pipeline/run.py#L253)。

## Checkpoint 与失败语义

当前 `PhaseCheckpointer` 要求同步保存完整状态。它调用原生 DCP save 后 CUDA synchronize 和全 rank barrier，仅 global rank 0 写 commit，再 barrier，由每个 rank 读取同一 commit；commit 含 run ID、training identity、input plan identity、world size、completed updates 和 native cursor。依据 [checkpoint.py:46](../../torchtitan/torchtitan/models/dspark_draft/checkpoint.py#L46)、[checkpoint.py:102](../../torchtitan/torchtitan/models/dspark_draft/checkpoint.py#L102)。每个 rank 的 RNG 和 buffers 也进入完整状态，见 [原生 trainer.py:168](../../torchtitan/torchtitan/models/dspark_draft/trainer.py#L168)。

M3 必须让 T0、T1 和独立验证进程看到同一 checkpoint 命名空间。相同路径字符串但两个节点各自本地存储不满足条件。共享路径的配置、可写性和跨节点可见性应在初始化前核实。仅 node rank 0 读取 `training-result.json`，其他 launcher 返回自己退出状态；总协调器收到两个成功退出、全部 rank update 事件与独立 checkpoint 验证后才能标记训练完成。现有结果写出也仅 global rank 0 负责，见 [train.py:45](../../torchtitan/torchtitan/models/dspark_draft/train.py#L45)、[actors.py:334](../../deepspec/pipeline/actors.py#L334)。

独立验证至少检查：commit 与 `.metadata` SHA256；metadata 引用分片存在且 byte ranges 完整；原生 world=8、completed updates=3、cursor=6、plan/run/training identity；实际读取 DCP optimizer step=3 和数据游标，以及全 rank 状态覆盖。分片数以当前 pinned DCP writer 的预期布局为依据；不要让一个固定文件数代替完整引用校验。可复用现有 DCP CPU 读取方法 [compare_torchtitan_checkpoints.py:15](../../tests/compare_torchtitan_checkpoints.py#L15)。现有单节点验证脚本硬编码 4 ranks、4 shards、cursor=sample count，须抽取拓扑参数化验证后才能用于 M3，见 [debug_single_node.py:388](../../scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.py#L388)、[debug_single_node.py:438](../../scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.py#L438)。

任何 rank、launcher、节点或必要 collective 失败都由管理层终止整个训练实例及本轮生产，停止 admission 并唤醒 buffer 等待者。`torchrun --max-restarts=0` 及单机 subreaper 保留，但不能只靠 torchrun 本机错误传播等待另一节点自行退出。每节点 owner 需要有界关闭本轮子进程，driver 丢失时也应触发相同所有权约束；不可达节点报告 unknown，不能宣称已清理。现有基础见 [train.py:62](../../torchtitan/torchtitan/models/dspark_draft/train.py#L62)、[process.py:45](../../deepspec/orchestration/process.py#L45)、[cluster.py:707](../../deepspec/pipeline/cluster.py#L707)。完整 checkpoint 不自动赋予流式样本重放/自动恢复能力。

## M1 兼容性中的独立 DP 组合

规格 M1 包含生产 DP1/DP2 与训练 DP1/DP2 的四种组合。当前 CLI 显式要求 `producer_dp=2` 时 `consumer_dp=2`，所以不能称当前入口已经完整支持四种组合。见 [run.py:705](../../deepspec/pipeline/run.py#L705)。

计划需解除这项人为耦合，生产路由使用 `position % producer_dp`，读者集合独立使用 `position % consumer_dp`，全局训练 batch 和 GAS 按训练 DP 计算。生产 DP2、训练 DP1 时每个样本仍由全部四个训练 ranks 读取，native cursor=12、GAS=4；生产端可乱序完成，但 admission 和输入计划保持原顺序。四种组合要各自验证，不从 DP2/DP2 的历史证据推断 DP2/DP1 已通过。

## 实施触点

| 文件或模块 | 必要变更 |
| --- | --- |
| `deepspec/pipeline/topology.py`、新统一 topology plan/兼容适配模块 | 显式 node/rank/local world/TP/DP 计划；保留 legacy consumer_nodes 的含义；生产 DP 与训练 DP 独立 |
| `deepspec/pipeline/run.py`、`cluster.py` | CLI 解除 M1 DP 耦合；两个训练节点及每节点资源；共享 rendezvous、来源一致性和存储可见性；按节点唯一记录；结束核验基于拓扑计划 |
| `deepspec/pipeline/actors.py` | 消费计划驱动的本地 launcher 配额、有限阶段 timeout、局部结果和所有权；保留 torchrun 原生训练入口 |
| `deepspec/pipeline/trainer.py`、`data.py`、`buffer.py` | 全 rank readiness 握手和前置校验；统一 node/global/local mapping；明确 read ACK 与 training commit |
| `deepspec/pipeline/memory.py` 及 NodeMonitor 调用侧 | 每节点 local readers、实际 pool host、writer staging 与客户端/prefetch 预算；全局完整 update admission |
| `deepspec/orchestration/process.py` 及统一生命周期模块 | 复用现有本地 subreaper；补跨节点失败/driver 丢失/不可达清理状态和有界退出 |
| 独立验证脚本和 `tests/test_pipeline_cluster.py` 等 | 消除单节点 rank/cursor/shard 常量；增加 M3 前置验证和失败组退出测试 |

TorchTitan mesh、训练数学和 PhaseCheckpointer 本次优先复用；若需要增加管理证据字段，在适配层/训练入口做最小扩展，不另写训练器。

## 可执行 CPU 验证与真实验收分层

以下命令仅列为实施期间的验证建议，本研究没有执行它们。纯逻辑测试不需要启动模型、Ray 服务或 Mooncake master；确保使用项目已验证的 Python 环境：

```bash
cd /mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm
DEEPSPEC_TEST_PYTHON=/mnt/afs_share/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python
CUDA_VISIBLE_DEVICES='' "$DEEPSPEC_TEST_PYTHON" -m pytest -q tests/test_pipeline_cluster.py -k 'dp_launchers or separated_consumer or dp_groups or dp_events or memory_limits'
CUDA_VISIBLE_DEVICES='' "$DEEPSPEC_TEST_PYTHON" -m pytest -q tests/test_pipeline_buffer.py
```

现有命令构造测试核验多节点 shared rendezvous 和不同 node rank，以及单节点 8 rank launcher；事件测试核验 DP/TP 坐标、reader ownership 和 native cursor。依据 [test_pipeline_cluster.py:239](../../tests/test_pipeline_cluster.py#L239)、[test_pipeline_cluster.py:292](../../tests/test_pipeline_cluster.py#L292)。buffer 测试覆盖完整 update 容量、指定 readers 和成功删除后返还 credit，见 [test_pipeline_buffer.py:29](../../tests/test_pipeline_buffer.py#L29)。

实施时补充的 CPU case：M3 exact rank 表与 M2 同节点对照；M1 四种 DP 组合的独立路由；缺失/重复/错节点/错 identity 握手不放行；两节点重复 local GPU ID 仍可区分；每节点 4 GPU 而非 8 GPU 配额；预算按局部读者与 pool owner；单 rank 错 cursor、错 reader、假成功及不完整 checkpoint 被拒绝；模拟一个 launcher/driver 失败后整组关闭且保留无关任务。同步/异步模拟不能替代真实网络 collective。

`tests/pipeline_rank_probe.py` 是 CPU/Gloo 数据协议 probe，但使用真实 Ray 和 Mooncake，需单独的隔离集成环境；不要混进“无服务单元测试”。`tests/pipeline_collective_probe.py` 是实际八 GPU NCCL probe，会初始化 CUDA；在真实 M3 两个训练节点分别 4 GPU 时执行可验证 TP 留在本节点、DP all-gather/reduce-scatter 跨节点。它当前显式构造 process groups，不证明原生 TorchTitan mesh 与计划一致，仍需训练握手证据。源码分别见 [pipeline_rank_probe.py:17](../../tests/pipeline_rank_probe.py#L17)、[pipeline_collective_probe.py:13](../../tests/pipeline_collective_probe.py#L13)。

最终真实 M3 在三节点角色隔离下分别运行 4K 和 128K，每轮 12 样本、48 次 SHA256 reader 校验、8 ranks 各 3 次更新、cursor6/sample12，以及独立完整 DCP、字节预算、全部对象释放和资源清理证据。另做单训练节点失联/一个 rank 失败/取消与 driver 退出场景，检验整组失败和未知清理状态。M2、M3 可顺序复用同三台机器；未实际执行的场景必须标记未执行或资源不可用。

## Alternatives considered

每 rank 一个 Ray actor 会增加框架生命周期与 distributed initialization 的适配范围，且本需求已有可复用的每节点 torchrun 框架，首版不采用。两个独立训练任务既不实现全局 TP4×DP2，也不满足统一优化器更新，拒绝。修改 TorchTitan mesh 轴顺序没有必要且会破坏现有 reader/cursor 契约，拒绝。将 READ_ACK 当 checkpoint 提交、通过缩小训练 world 绕过失败、或自动重启补算样本，都不在首版范围内。
