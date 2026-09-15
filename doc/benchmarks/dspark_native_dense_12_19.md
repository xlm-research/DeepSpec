# TorchTitan DSpark：任务 12–19

状态：12–19 的实现已写入；第 19 项在原有严格容差下尚未完全对齐，不能标为完成。
先连续实现 12–19，再按用户最新要求停止逐项测试，直接验证
`TP2 × CP2 × PP2` 的结果对齐与同拓扑恢复。

## 实现

配方位于 `torchtitan.models.dspark_draft.config_registry`。

| 任务 | 配方函数 | 八卡布局 | 实现要点 |
| --- | --- | --- | --- |
| 12 | `qwen38_27b_hsdp` | replicate2 × shard4 | 原生二维 FSDP mesh、FP32 归约、完整阶段状态 |
| 13 | `qwen38_27b_sp` | shard2 × TP4，SP 开启 | context/query 分别切分；支持 query token 数不能整除 TP；完整参数梯度收缩 |
| 14 | `qwen38_27b_vocab_parallel` | shard2 × TP4 | LM/Markov 输出按词表分片；CE、L1、acceptance/confidence 使用全词表归一化 |
| 15 | `qwen38_27b_sp_vocab_parallel` | shard2 × TP4，SP/loss 同开 | SP 输出接词表分片 head，保持冻结 head 的 hidden 梯度 |
| 16 | `qwen38_27b_cp` | shard4 × CP2 | 重组 producer features 后建立 head/tail 视图；接入现有 DSpark ring attention |
| 17 | `qwen38_27b_tp_cp` | TP4 × CP2 | TP peers 共享 anchors；CP 独立监督参与分母一次 |
| 18 | `qwen38_27b_pp` | shard4 × PP2 | 原生 `Schedule1F1B`；五层按 3/2 划分；query 和可微 context 跨 stage 传递 |
| 19 | `qwen38_27b_tp_cp_pp` | TP2 × CP2 × PP2 | 联合布局、全部 stage 的状态提交和同拓扑恢复 |

SP 和 loss parallel 分别使用原生 `parallelism.enable_sequence_parallel` 与 DSpark
`loss.enable_vocab_parallel` 开关。词表 loss 的通信只归约归一化量、目标位置值及
词表分片贡献；训练不收集完整词表 logits。测试在进程退出后于 CPU 拼接保存的
logits 分片，与完整词表参考比较。

PP 的逻辑 GAS 等于原生外层累计次数乘以 pipeline microbatch 数。每份逻辑
microbatch 先按自己的全局有效监督分母归一化，再参与等权平均。分区停点、
checkpoint 游标及恢复检查均使用逻辑 microbatch 数。

## 验收方法

入口为 DeepSpec 阶段启动器 → 八个真实 TorchTitan worker。参考由保留的真实
Qwen DSpark 训练循环和独立 loss 表达式生成。模型为五层、24 Q heads / 4 KV
heads、hidden64、FFN128、vocab128、16-token 输入、2 anchors × block3。
这是实际模型类的小规模数值验收；完整尺寸/128K 的结论见任务 10 的独立报告。

- FP32/BF16、两个完整 updates、逻辑 GAS2，覆盖不同有效分母、空监督和边界监督。
- 比较输出、loss、全部可训练梯度、clip norm、模型参数、FP32 master/Adam 和 scheduler。
- 数值容差沿用参考基线：`rtol=1e-4, atol=1e-6`。
- 第 19 项执行连续训练、第一阶段及全新 worker 恢复三个流程。
- 连续与恢复的完整 DCP、下一 update、RNG 和数据游标要求精确一致。
- 验证所有本任务 worker 已退出；记录外部作业占用，受资源争用影响的耗时不作速度收益结论。

运行器：`tests/run_torchtitan_parallel_acceptance.py`。
当前产物：`outputs/dspark_dense_12_19_20260915_v2/`。
原始失败尝试保留在该目录和 `outputs/dspark_dense_12_19_20260915/` 中。

```bash
source env.sh
for dtype in float32 bfloat16
do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  python -m tests.run_torchtitan_parallel_acceptance \
    outputs/dspark_dense_joint_replay \
    --topologies tp_cp_pp --dtypes "$dtype" --phases continuous
done

# 连续训练的数值断言会报告失败；仍可用其真实 checkpoint 检查恢复。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
python -m tests.run_torchtitan_parallel_acceptance \
  outputs/dspark_dense_joint_replay \
  --topologies tp_cp_pp --phases first resumed

# 只读 CPU 产物，汇总所有比较项，不改变容差或重新启动 GPU worker。
python -m tests.report_torchtitan_joint_alignment \
  outputs/dspark_dense_joint_replay
```

只在检查过未完成尝试且其 worker 已退出后使用 `--retry-incomplete`；它会保留并
重命名未完成的阶段目录。共享 GPU 的小规模数值调试使用
`DEEPSPEC_PARALLEL_SHARED_GPUS=1`，报告会标注这一条件。

## 已确认的证据

- 四进程 Gloo 梯度检查：每个进程两项通过，覆盖不均匀 SP、词表 loss、冻结 head
  的 hidden 梯度及零监督分母。
- Producer 特征重组及错误输入检查：五项通过。
- 用户收窄验证范围前，HSDP FP32/BF16 均已完成两步参考对齐、恢复及提交失败
  流程；跨进程恢复后 754 个 checkpoint 字段、344 个张量及下一步更新精确一致。

第 13–18 项不再单独运行验收；第 19 项配方未开启
SP/词表并行，因此不能将联合布局证据视为这些独立开关的完整验收。

## 第 19 项：实际结果

用户最新指示：暂不修改 CP。新增的 all-to-all 实验已停止并撤回，保留原有 ring
CP；本节保留其真实数值差异，不以实验结果替代已运行的验收。

详见 `outputs/dspark_dense_12_19_20260915_v2/tp_cp_pp/alignment-report.json`。
比较覆盖全部 stage，先恢复各 rank 的完整参数，再核对 stage 所有权及 TP/CP
副本一致性。所有数值比较仍使用 `rtol=1e-4, atol=1e-6`。

| 比较项 | FP32 | BF16 |
| --- | --- | --- |
| teacher logits、标签、mask、三个 loss 分母 | 精确一致 | 精确一致 |
| draft logits | 通过，最大绝对差 `5.66e-7` | 未通过，最大绝对差 `0.00879` |
| CE/L1/confidence、总 loss | 全部通过 | CE 通过，部分 L1/confidence/总 loss 未通过 |
| 全部梯度 | 通过，两步最大绝对差 `1.12e-8`、`6.76e-8` | 未通过，两步整体相对 L2 差 `1.08%`、`2.44%` |
| clip norm | 通过 | 首步 `7.09375`，参考 `7.125`；第二步一致 |
| 模型参数 / FP32 master | 首步 9 个、次步 8 个元素超阈值，最大差 `8.61e-6` | 未通过，参数最大差 `0.00390625` |
| Adam moments / scheduler | 通过 | scheduler 一致，部分一阶 moments 未通过 |

FP32 参数差的具体诊断：`layers.0.mlp.gate_proj.weight[39,63]` 的梯度为
`-1.81248e-9`，参考为 `-1.69697e-9`。将两者分别代入相同的首步 Adam 公式
`-lr*g/(abs(g)+eps)`，预测参数差 `8.3604e-6`，实测 `8.3596e-6`。
接近零的梯度经 Adam 的归一化放大了浮点归约差异。

另以独立 CPU 公式重放两步 Adam：两种精度的 FP32 master 最大绝对误差均为
`7.45e-9`。该结果用于诊断，不替代完整参考对齐判定。BF16 剩余差异已经出现在
训练输出与梯度中，尚未消除；不能仅凭更新公式正确就宣称联合数值验收通过。

本轮修复了两处 PP 接入问题：

- 当前 PyTorch 的公开 `step()` 仍接收整批输入，TorchTitan 已传入 microbatch
  列表；薄适配层直接交给原生 `Schedule1F1B._step_microbatches()`。
- PP tensor tuple 的 confidence 输出及其静态 metadata 沿用基线 FSDP 的
  输出 dtype 转换，避免将 BF16 基线改成 FP32 输出。

FP32/BF16 均完成保存退出与同拓扑新进程恢复：各 754 个 checkpoint 字段、
344 个张量、全部 rank 的下一步更新/输出/RNG/游标精确一致。所有本任务 worker
已退出。该恢复结论与跨拓扑数值比较的失败结论分别记录。

后续只读张量诊断将 BF16 的首次前向差异定位到第一层 attention：embedding、
teacher 投影、Q/K/V 及 q/k norm 均精确一致，第一层 `o_proj` 输入的相对 L2
差为 `0.003003`。参考重放的首个 forward 与存档逐位一致。诊断张量保留在
`/tmp/deepspec-dense-12-19/trace-reference/` 和 `trace-joint/`；暂停的诊断工具
移至 `.scratch/dspark-torchtitan-orchestration/debug_joint_trace.py`，训练 fixture
中的临时 hook 已移除。

## 关闭 CP 的对照（用户追加要求）

用户已接受本节实测误差范围。保留原始严格比较结果，验收脚本的容差不作静默
放宽；该接受针对本次关闭 CP 的对照，CP 数值修复仍按用户要求暂停。

使用四卡 `DP1 × TP2 × CP1 × PP2`，保留同一份 DP1 参考、初始权重、固定
features、五层模型、两个 updates 和逻辑 GAS2。没有更换 CP 算法；只将 CP
关闭。产物为 `outputs/dspark_dense_19_no_cp_20260915/tp_pp/alignment-report.json`。

| 比较项 | FP32 | BF16 |
| --- | --- | --- |
| 四个 microbatch 的输出及三个 loss / 总 loss | 原有容差内，logits 最大差 `3.50e-7` | 全部逐位一致 |
| 两步全部梯度 | 原有容差内，最大差 `3.73e-9`、`3.65e-8` | 全部逐位一致 |
| 两步模型参数、FP32 master、Adam moments、scheduler | moments/scheduler 通过；参数/master 首步 4 个、次步 3 个元素超阈值，最大差 `5.35e-6` | 全部逐位一致 |
| 首步 clip norm | 一致 | `7.09375`，参考 `7.125` |
| 第二步 clip norm | 差 `9.54e-7`，通过 | 一致 |

因此无 CP 时，BF16 的梯度和训练结果已逐位对齐；整套严格检查仍会因首步
clip norm 的差异返回失败。原生 PP 会先计算 stage norm 再做跨 stage 归约，
其 BF16 归约顺序与整模型参考不同。本次两步的实际裁剪后梯度和更新均未改变；
这里不将 norm 差异隐藏或改写成通过。FP32 仍有接近零的梯度经 Adam 放大的
少量参数差异。

本组仅运行用户要求的连续两步对照，四个 worker 均已退出；未追加逐票测试或
重复恢复测试。

```bash
source env.sh
for dtype in float32 bfloat16
do
  DEEPSPEC_PARALLEL_SHARED_GPUS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  python -m tests.run_torchtitan_parallel_acceptance \
    outputs/dspark_dense_no_cp_replay \
    --topologies tp_pp --dtypes "$dtype" --phases continuous
done
python -m tests.report_torchtitan_joint_alignment \
  outputs/dspark_dense_no_cp_replay --topology tp_pp
```

## 当前支持边界

使用 `spmd_types`、完整 update 后同步 DCP、同拓扑恢复。外层 model compile
继续关闭。PP 首版支持 PP2/1F1B、至少两个 pipeline microbatches、固定长度
packed 输入和零 attention dropout；CP 继续使用已有 ring attention，local batch
为 1。GLM/MoE EP 属于后续任务。
