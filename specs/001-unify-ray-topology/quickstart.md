# Quickstart：统一拓扑入口与验证路径

统一 `deepspec.pipeline.cli` 六个命令、原生推理接缝和 M3 双 launcher 已有实现及 CPU 契约测试。真实训练通过范围须查阅 [验收报告](acceptance-report.md)，不能由实现存在推断；当前集群仅有一个八卡节点，多节点矩阵仍缺运行资源。 配置与状态详见 [CLI 契约](contracts/cli.md)、[运行契约](contracts/runtime.md)和[数据模型](data-model.md)。

## 1. 环境与资源前提

```bash
cd /mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm
source ./h800conda.sh
export DEEPSPEC_ROOT="$PWD"
export DEEPSPEC_SPEC="$DEEPSPEC_ROOT/specs/001-unify-ray-topology"
"$PIPELINE_PYTHON" -c 'import sys, importlib.metadata as m; print(sys.executable); print({p:m.version(p) for p in ("torch","ray","vllm","mooncake-transfer-engine")})'
```

在实际参与节点确认同一 Python、源码/二进制、模型与输入身份，模型路径及共享输出均可用。M2 需要三个不同物理节点各8卡；M3只使用同三台节点各4卡并调整角色。节点间需支持 Ray、Store TCP、训练 rendezvous 与 collective 通信。填写真实 Ray 地址、node ID 和数据文件；不要把当前本机的服务地址推断成三节点已就绪。

需要提供的环境变量：`RAY_ADDRESS`、`TARGET_MODEL_PATH`、`DATA_SOURCE`、`INFER_NODE_A`、`INFER_NODE_B`、`TRAIN_NODE_B`、`TRAIN_NODE_C`。M2使用 A/B 推理和C训练，M3使用A推理和B/C训练；顺序复用时 TRAIN_NODE_B 可对应原推理B，但节点ID必须真实、唯一。

可在执行验收前只读查询已有集群：

```bash
: "${RAY_ADDRESS:?设置已有 Ray 集群的地址}"
"$PIPELINE_PYTHON" - <<'PY'
import json, os, ray
ray.init(address=os.environ['RAY_ADDRESS'], namespace='deepspec-plan-inspect')
print(json.dumps([{'id':n['NodeID'],'ip':n['NodeManagerAddress'],'alive':n['Alive'],'resources':n['Resources']} for n in ray.nodes()], indent=2))
ray.shutdown()
PY
```

不通过此命令启动或停止集群。运行前应保留资源清单，确认所需节点实际存活。GPU与内存额度只在实施后的preflight/实际分配中确认。例子的64 GiB pool、window8是已知基线量级；不同输入若不能容纳完整更新组，应被拒绝并给出所需预算，不能盲目照搬或静默放宽。

## 2. 当前可用的文档和纯逻辑检查

这些检查不加载模型，不启动 Ray/Mooncake 服务：

```bash
"$PIPELINE_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from jsonschema import Draft202012Validator
p=Path(os.environ['DEEPSPEC_SPEC'])/'contracts'
s=json.loads((p/'task-config.schema.json').read_text())
Draft202012Validator.check_schema(s)
v=Draft202012Validator(s)
for name in ('m2.example.json','m3.example.json'):
    v.validate(json.loads((p/name).read_text()))
    print(name, 'structure valid; environment placeholders still need expansion')
PY
CUDA_VISIBLE_DEVICES='' "$PIPELINE_PYTHON" -m pytest -q tests/test_pipeline_buffer.py
CUDA_VISIBLE_DEVICES='' "$PIPELINE_PYTHON" -m pytest -q tests/test_pipeline_cluster.py -k 'dp_launchers or separated_consumer or dp_groups or dp_events or memory_limits'
```

JSON Schema只证明结构；布局算术、节点事实、预算与后端能力须由新planner验证。现有单元测试通过也不代表M2/M3实现完成。

示例显式填写 `transport.rdma_devices=""` 与 `timeouts_seconds.budget_snapshot=5`；省略时由规范化实现填入相同默认值，Schema validator本身不会填值。旧M0/M1的显式RDMA设备值按CLI契约原样迁移，M2/M3继续要求TCP/CPU。预算快照年龄达到5秒即需重新采样；修改有效期会改变冻结配置/计划身份。

## 3. 生成实际 M2 / M3 配置

先设置上面要求的真实环境变量。本示例生成M2 4K输入文件；M3改 `EXAMPLE=m3`、`CASE_NAME=m3-4k`；128K改context为131072并使用新目录与满足长度要求的数据。

```bash
export EXAMPLE=m2
export CASE_NAME=m2-dp2-4k
export CONTEXT_LENGTH=4096
export RUN_OUTPUT="$DEEPSPEC_ROOT/outputs/ray-topology-acceptance/$CASE_NAME/$(date +%Y%m%d_%H%M%S)_$$"
export CONFIG_FILE="/tmp/deepspec-$CASE_NAME-$$.json"
"$PIPELINE_PYTHON" - <<'PY'
import json, os, re
from pathlib import Path
from jsonschema import Draft202012Validator
p=Path(os.environ['DEEPSPEC_SPEC'])/'contracts'
value=json.loads((p/(os.environ['EXAMPLE']+'.example.json')).read_text())
def expand(v):
    if isinstance(v,dict): return {k:expand(x) for k,x in v.items()}
    if isinstance(v,list): return [expand(x) for x in v]
    if isinstance(v,str):
        return re.sub(r'\$\{([A-Z_][A-Z_0-9]*)\}',lambda m:os.environ[m[1]],v)
    return v
value=expand(value)
value['data']['context_length']=int(os.environ['CONTEXT_LENGTH'])
Draft202012Validator(json.loads((p/'task-config.schema.json').read_text())).validate(value)
assert Path(value['data']['source_path']).is_file(), 'DATA_SOURCE must exist'
assert Path(value['model_path']).is_dir(), 'TARGET_MODEL_PATH must exist'
assert not Path(value['output_dir']).exists(), 'Use a new run output directory'
with Path(os.environ['CONFIG_FILE']).open('x') as f:
    f.write(json.dumps(value,indent=2)+'\n')
print(os.environ['CONFIG_FILE'])
PY
```

重新运行请换 `CASE_NAME` 与 CONFIG_FILE；不删除已有运行证据。128K不能仅改context参数就算验收，应核对准备后的实际序列长度与输入身份，并在报告中记录真实长度。

## 4. preview → transport → run → verify

以下命令已接通；每次使用新的输出目录和冻结计划：

首次真实训练验收还须先完成tasks中的T079/T080：实际运行事件与独立节点/进程/资源采样均已接通，正常/背压/失败的轻量检查通过。后续status展示可以晚于这些采集能力完成，但每个真实case必须从启动到清理保留FR-018证据，不能事后补采时长或峰值。

```bash
"$PIPELINE_PYTHON" -m deepspec.pipeline.cli preview --config "$CONFIG_FILE"
"$PIPELINE_PYTHON" -m deepspec.pipeline.cli transport-check --plan "$RUN_OUTPUT/plan.json"
"$PIPELINE_PYTHON" -m deepspec.pipeline.cli run --plan "$RUN_OUTPUT/plan.json"
"$PIPELINE_PYTHON" -m deepspec.pipeline.cli status --run-dir "$RUN_OUTPUT" --json
"$PIPELINE_PYTHON" -m deepspec.pipeline.cli verify --run-dir "$RUN_OUTPUT"
```

阶段默认期限见生成的 `plan.json.timeouts_seconds`；所有等待均为正数有限期限。修改期限必须重新 preview。模型和 GPU PG 释放后，独立 CPU verifier 才能加载 DCP，验证工作集也须通过节点预算预检。源码改变后不能复用旧计划。

已有集群优先填写明确的 GCS 地址。使用 `ray_address="auto"` 时，首次检查会解析并冻结实际地址，后续检查和运行继续使用该地址，避免本机其他测试启动临时 Ray 后改变连接目标。当前只有一个物理节点，可执行 M0；M1–M3 继续记录为 blocked，不能用同节点多个进程代替多个物理节点。

preview期望：角色/配额/TP/DP、rank/reader表、writer位置、单池位置、逐节点budget与阶段期限完整；GPU预留/模型进程/大池数为0。检查前后Ray资源与GPU进程差集，不能仅相信返回字段。transport-check的临时64 MiB探针不计为真实特征池验收，退出需清理其对象/服务/CPU actors。

M2期望：两个TP8副本各有A/B 4＋4的实际执行证据，A上两个TP0 writer；第三节点8个训练ranks。M3期望：A推理4卡，B/C各4个训练ranks，TP留在节点内、DP shard跨节点。两者均完成12个计划样本、48次完整reader校验、8 ranks各3更新、native cursor6/sample cursor12、可独立加载完整checkpoint；最终源对象释放和所有本次资源确认回收。

## 5. 必须完成的真实训练矩阵

每行除注明外均为12样本、3更新、48次读取。每个case使用独立目录；4K先通过再进行对应128K，以尽早暴露分组与生命周期错误。

| 场景 | 推理 | 训练 | 序列长度 | 目的 |
|---|---|---|---|---|
| M0 | 单节点TP4DP1 | 同节点另4卡TP4DP1 | 4K、128K | 旧单机入口与新入口等价回归 |
| M1-11 | 一节点TP4DP1 | 另一节点TP4DP1 | 4K、128K | 旧角色分离基线 |
| M1-12 | 一节点TP4DP1 | 另一节点TP4DP2 | 4K | 生产/训练DP独立 |
| M1-21 | 一节点TP4DP2 | 另一节点TP4DP1 | 4K | 新解除的CLI耦合必须验证 |
| M1-22 | 一节点TP4DP2 | 另一节点TP4DP2 | 4K、128K | 现有最大单角色节点组合 |
| M2-DP1 | 两节点各4卡TP8DP1 | 第三节点8卡TP4DP2 | 4K | 扩DP对照，TP固定8，未分配推理卡不使用 |
| M2-DP2 | 两节点各8卡TP8DP2 | 第三节点8卡TP4DP2 | 4K、128K | 主布局 |
| M3 | 一节点4卡TP4DP1 | 两节点各4卡TP4DP2 | 4K、128K | 跨节点draft独立验收 |

M2-DP1由M2示例改 inference.dp=1、两推理节点gpus各4，训练维持8卡/DP2；写入新配置并重新preview。M0/M1使用同一schema按矩阵生成配置，亦保留原有启动方式的对照。M0/M1的DP1训练原生cursor12/GAS4，DP2训练cursor6/GAS2，不能统一按样本数断言cursor。

以上计数只适用于三次更新的矩阵实例。通用CLI/verifier按冻结计划计算：样本数=steps×global_batch_size，原生cursor=样本数/training.dp，完整读取数=样本数×training.tp；例如steps=5时样本20、DP1/DP2原生cursor20/10、完整读取80。T043的CPU核验fixture须覆盖1/3/5次更新，真实13项矩阵规模保持不变。

## 6. 故障与容量验证

以下 CPU 契约测试已实现，可单独执行：

```bash
CUDA_VISIBLE_DEVICES='' "$PIPELINE_PYTHON" -m pytest -q \
  tests/test_pipeline_plan.py tests/test_pipeline_lifecycle.py \
  tests/test_pipeline_vllm_placement.py tests/test_pipeline_multinode_training.py \
  tests/test_pipeline_verification.py
```

测试应覆盖有实际意义的契约：错误node/重复GPU/不足CPU/过大DP在GPU初始化前拒绝；native core部分创建失败；全rank握手缺失/冲突；旧配置冲突；每节点reader/pool/writer计费；更新边界跨batch；慢reader、失败删除与副本存活；错误cursor、缺DCP范围/身份、假成功；有界cleanup与PID重用。实际Ray资源探针和CPU/Gloo传输探针单独运行，不能把它们归入无服务单元测试。

单节点可运行八个 CPU rank 的真实 Gloo 通信及故障传播探针，独立 supervisor 记录进程回收，结果写入验收索引：

```bash
CUDA_VISIBLE_DEVICES='' "$PIPELINE_PYTHON" -m tests.run_pipeline_acceptance --cpu-collective-probe
CUDA_VISIBLE_DEVICES='' "$PIPELINE_PYTHON" -m tests.run_pipeline_acceptance --cpu-collective-probe --collective-fail-rank 7
```

上述探针不加载模型、不使用 GPU，仅证明单节点 CPU 进程间通信和受控故障清理。跨节点探针须使用冻结计划并在计划中的真实节点启动；单节点结果不能计入 T067 的跨节点验收。

真实集成故障在隔离的验收run上执行：M3停止一个**本次登记的**训练rank或隔离一个训练节点；验证另一训练节点与生产侧停止，终态failed。执行主动取消：

```bash
"$PIPELINE_PYTHON" -m deepspec.pipeline.cli cancel --run-dir "$RUN_OUTPUT"
"$PIPELINE_PYTHON" -m deepspec.pipeline.cli status --run-dir "$RUN_OUTPUT" --json
```

另用登记的driver PID/start-time注入driver SIGKILL，确认节点watchdog/subreaper与PG回收；故障工具只接受该run的registry身份，不能以名称或整机进程列表盲杀。验证旁路任务和外部Ray/master健康，可达节点残留为0；不可达资源应是unknown，不能写cleanup_complete=true。

预算压力测试让真实128K reader延迟、删除重试耗尽、下一个更新组节点headroom不足，检查准入阻塞/有限失败、源对象不提前删除、容量不提前归还。被阻塞的已准入更新组应仍能完成，随后更新组不能绕过新鲜预算检查。

## 7. 证据与完成条件

每case记录 planned / not_run / blocked / failed / passed、资源布局、环境与源码摘要、模型/数据/实际长度、开始结束时间、全部rank与读取/更新证据、内存上界/观测、checkpoint verifier和cleanup。对未实现的新测试记录not_run，不用空文件或模拟passed替代。

只有[规格 SC-001–010](spec.md)对应证据完整且通过，才可声称功能交付。既有 `debug_logs/h800_setup_20260920/setup-result.json` 是历史单机基线；本次生成的设计文档与schema校验不能替代M2/M3真实验收，也不支持吞吐改善结论。
