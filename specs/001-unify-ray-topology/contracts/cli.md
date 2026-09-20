# CLI 与任务配置契约（待实现）

新入口为 `python -m deepspec.pipeline.cli`。本文件中的命令是实施目标，目前不能据此假定新 CLI 已存在。旧 `deepspec.pipeline.run` 和训练 shell 入口继续可用，并适配到同一规划/控制层；保留 `PIPELINE_PYTHON`、模型/输入、batch/window、训练和预算语义。

## 命令

| 命令 | 输入 | 输出与副作用 |
|---|---|---|
| `preview --config FILE` | v3 JSON 或已支持的 legacy 配置 | 先结构/语义校验，再 CPU 准备及只读节点预检；在配置指定的新 output_dir 保存规范化配置、plan、输入计划、环境与预算；不预留 GPU、不加载模型、不建大池 |
| `run --plan FILE` | preview 产生的 plan.json | 重验身份与资源，分配、运行、保存、核验、清理；只能运行未执行过的计划，已有终态拒绝复用 |
| `transport-check --plan FILE` | 完整合法计划 | 在所有实际 writer/reader 节点进行小对象 TCP/CPU 写/读/校验/删探针；自建小型 probe pool，默认64 MiB，单独 probe namespace；无模型/GPU预留，不代表真实特征容量验收 |
| `status --run-dir DIR [--json]` | 运行目录 | 只读合并 status、节点事件、lease 与 cleanup；展示 read 与 committed 进度，不把 lease 过期显示成健康 |
| `cancel --run-dir DIR` | 运行目录 | 原子写 run_id/plan_hash 对应取消请求并通知协调器；等待声明的处理/清理期限；重复请求幂等 |
| `verify --run-dir DIR` | 完成后的证据和 checkpoint | 独立 CPU 核验原生完整 DCP/commit、输入身份、游标、样本/读者、实际布局和清理；写独立 verification 记录，失败不能把原 status 改成成功 |

exit code：0=命令成功（status 成功读取不表示任务成功）；2=配置/身份/使用错误；3=运行或核验失败；4=资源不足/节点不可达导致无法执行验收；130=主动取消。错误 JSON 至少含 `code, message, run_id, phase, node_id, field_path, retryable`，不适用字段可为 null。输出文件身份与 stdout 的 run_id 必须一致。

preview 的 output_dir 必须不存在，避免覆盖已有证据；run 从其已冻结 plan 启动，不重新生成不同输入。重新验收更换目录/run_id。preview 后资源被占用，run 在 allocation 期限内失败并回滚，而不是修改已确认卡数。

## 配置规范

结构文件：[task-config.schema.json](task-config.schema.json)。示例：[M2](m2.example.json)、[M3](m3.example.json)。`${...}` 仅用于示例模板，需在生成实际 JSON 时显式替换；CLI 不隐式执行 shell 或扩展表达式，发现未替换变量即拒绝。

| 字段 | 语义 |
|---|---|
| schema_version | 新格式为3；legacy 1/2 有独立升级路径，不能直接套 v3 Schema 报未知字段 |
| layout | M0/M1/M2/M3；所有跨字段矩阵规则均须检查，JSON Schema 通过不等于布局合法 |
| ray_address | 已有集群地址；旧单机入口的本地 Ray 自建行为仍由适配层记录 owner |
| nodes | alias→唯一 node_id 或 IP selector，cpu_limit 是本任务可用 CPU 上限；可指定更低 feature_memory_cap_bytes |
| inference | tp/dp、每节点 GPU 额度、batch_size 与有限 writer_inflight；不能把 DP 自动扩展到训练 |
| training | TP4、DP1/2 shard 模式、CP=PP=1、逐节点 GPU 额度、global_batch_size=4、steps |
| data | source_path、context_length、epochs；准备后持久化原生输入计划和各样本实际字节 |
| store | pool 所在节点、pool_bytes、容量利用率；master owned/external；external 必须有可达 endpoint |
| transport | TCP/CPU/full 是必验基线；window、prefetch_depth/bytes 均有界；rdma_devices 保留旧设备选择字符串，缺省为空；M2/M3 限定 TCP/CPU |
| timeouts_seconds | allocation/initialization/run/transfer/collective/cleanup/lease/heartbeat，正数有限；heartbeat<lease；budget_snapshot 是独立的预算快照有效期，缺省5秒 |

旧版支持的 RDMA/CUDA 分支保留其原支持边界与独立校验，schema 可表达不代表新增矩阵已经支持；本功能不新增传输优化。M2/M3 明确要求 TCP/CPU；M0/M1 的首版必验矩阵同样使用 TCP/CPU。

规范化显式补入 `transport.rdma_devices=""` 和 `timeouts_seconds.budget_snapshot=5` 的缺省值，并将实际取值纳入config/plan hash；JSON Schema的default只是注释，不能依赖validator自动填值。显式值不被默认覆盖。预算快照在最终准入时年龄上界必须严格小于有效期，边界与刷新期限见 [数据模型](../data-model.md#预算快照有效期)。

语义校验必须在 GPU 副作用前完成：

- alias 唯一、selector 唯一、额度非重叠、完整 TP/DP 组；M0 一台物理节点合计8卡；M1两个角色节点；M2/M3三个角色节点。
- M2 必验 DP2 额度为8/8/8，DP1对照为4/4/8；M3为4/4/4。M2 的推理 DP 可以按两端额度声明更大的正整数，每端实际使用4×DP卡；超过任一端额度则拒绝，余卡不自动使用，更大规模单独验收。pool 默认第一个训练节点，若指定其他位置必须明确受支持；本版 v3 仅接受第一个训练节点作为pool/master服务位置。
- inference replica 的 GPU quota 与 tp/dp 一致；training world/local_world、GAS、输入计划组和 reader 映射一致；M1 pDP2/cDP1 是合法组合。
- CPU 必须足够容纳每节点 launcher（默认2×local GPU CPUs）、NodeAgent、frontend、每个native core、FeatureBuffer/master 和 gate；组件 CPU 声明之和不可超过 cpu_limit。
- 模型/输入/源码/依赖身份、共享目录、服务可路由性、预算与后端 capability 可验证。不支持的布局报 `UNSUPPORTED_TOPOLOGY`；缺少 vLLM 借 PG 接入报 `BACKEND_CAPABILITY_MISSING`。
- 所有超时写入最终计划；样例值是可调整的运行政策，不是模型必须在这些时长内完成的性能承诺。

## 旧配置兼容

旧字段映射必须可审阅：`producer_dp/consumer_dp/consumer_world_size` 对应 inference/training 并行配置；单个 producer_node/consumer_node 转成节点列表；`pool_bytes/window/prefetch_*` 进入相应区块。保留样本计划、预算和 batch 值，不以新的默认值覆盖显式旧值。

如果旧顶层与新字段同时存在且不同，报告具体冲突；未知新字段拒绝。历史 `consumer_nodes` 缺省解释必须按原 schema/布局推导，不能把 `consumer_dp=2` 自动变成两训练节点。旧 CLI 对 pDP2→cDP2 的人工限制解除后单独测试 pDP2/cDP1，不把新增合法组合说成已有实测。

`--rdma-devices` 在旧入口落入 `store.rdma_devices`；v1/v2升级时将其原样映射到v3的 `transport.rdma_devices`，兼容快照及Store客户端再从该规范字段回填旧后端参数。若新旧位置均显式提供，字符串相同则接受，不同则报包含两个字段路径的冲突；两者均缺失时才填空字符串。空字符串保留旧后端的自动选择语义，非空值在RDMA模式下须逐参与节点核查设备存在及可用，不能静默清空或换设备。TCP模式保留但不使用该字符串，不因字段非空启用RDMA；M2/M3的协议限制不变。T041/T044/T057须覆盖显式多设备值、缺省、同值并存、冲突、后端透传及TCP不切协议；这项配置兼容检查不扩大真实RDMA验收范围。

## 运行目录

```text
RUN_DIR/
├── config.normalized.json
├── plan.json
├── environment.json
├── inputs/input-plan.json
├── allocation.json
├── status.json
├── events/<component>-<node_id>-<rank_or_replica>.jsonl
├── control/cancel.json
├── transport-probe.json
├── checkpoints/                 # 原生 TorchTitan 保存格式
├── verification.json
└── cleanup/node-<node_id>.json
```

保留旧消费者需要的 `pipeline.json` 兼容快照；它指向同一计划、配置 hash 与原生输入路径，不能成为另一个可变真相来源。JSON 原子替换，事件按独立 writer 分文件。stderr 保留详细诊断，status/error 提供可机器判断的原因与清理状态。
