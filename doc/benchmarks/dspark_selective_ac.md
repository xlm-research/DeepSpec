# Qwen DSpark SelectiveAC 验证

2026-09-14；对应本地票 04。沿用实施前实际工作树基线、原训练循环、
loss、GAS、梯度裁剪与 FP32 master Adam，不更换 target 或 vLLM。

在 draft 的 `train.parallel` 中设置：

```python
use_activation_checkpoint=True,
activation_checkpoint_policy="torchtitan_selective",
use_compile=False,
```

默认策略仍为 `full`。TorchTitan 策略先关闭 HF 内置 checkpoint，再包装每层，
随后应用现有 FSDP2；保持 `preserve_rng_state=True`。当前不接受同时启用外层
model compile。target 的独立配置不需要增加或跟随 draft 的 AC 设置。
生产布局比较只检查 DP/CP/TP/PP 维度，忽略 AC、compile 等变换字段。

继续使用 `env.sh` 的解释器和本地 TorchTitan commit
`f6b9152e9bedcc18f5dc339b9f88265e5a07e988`。运行时将该源码目录加入 draft
的 `PYTHONPATH`；没有安装或重新编译 vLLM。

```bash
DRAFT_PYTHON=/mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python
PYTHONPATH="$PWD/torchtitan:$PWD/vllm:$PWD" OMP_NUM_THREADS=1 \
DEEPSPEC_BASELINE_REFERENCE="$PWD/output/dspark_torchtitan_baseline_20260914/numerics-final" \
"$DRAFT_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m unittest tests.test_dspark_selective_ac.DSparkSelectiveACTest.test_selective_ac_preserves_two_fsdp_updates

PYTHONPATH="$PWD/torchtitan:$PWD/vllm:$PWD" OMP_NUM_THREADS=1 \
"$DRAFT_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m unittest tests.test_dspark_selective_ac.DSparkSelectiveACTest.test_selective_ac_preserves_eight_rank_tp4_updates
```

两卡 DP shard2、TP1、CP1 的 FP32/BF16 SAC 结果与实施前固定归档对照通过。
两次 optimizer updates、GAS2，四个 microbatch 的不等分母、稀疏 mask、
零分母和尾部边界沿用原基线。输出、CE/L1/confidence、全部可训练梯度、clip norm、
模型参数、FP32 master weights、Adam moments/计数、scheduler 和 RNG 均比较，
容差保持 rtol=1e-4、atol=1e-6，未为 SAC 放宽。

八卡候选 D（DP shard2 × TP4、CP1）也通过 FP32/BF16 两次更新对照；
每个 TP group 使用同一 DP 样本与训练随机种子。模型为真实 Qwen3.8 DSpark 类，
测试尺寸为 2 层、hidden64、8 Q heads / 4 KV heads、128 词表和 16 tokens。
此处参考为同一现有 TP4 路径的无 AC 更新，不代表后续新 TP 后端验收。

八个 rank 均实际执行。以下为各项的跨 rank 最大值，包含首次使用时的 FlexAttention
编译和测试观测开销；AC 与参考先后运行，编译缓存状态不同，不能据此计算加速比。

| 精度 | SelectiveAC | 构建秒数 | 两次更新测试秒数 | 峰值 allocated 字节 |
| --- | --- | ---: | ---: | ---: |
| FP32 | 否 | 1.607 | 89.162 | 200472064 |
| FP32 | 是 | 2.584 | 3.254 | 68402688 |
| BF16 | 否 | 0.153 | 32.896 | 200263168 |
| BF16 | 是 | 0.078 | 0.783 | 68175360 |

日志位于 `output/dspark_torchtitan_implementation/`：
`selective-ac-red.log` 为策略尚未实现时的失败；`selective-ac-first.log`
记录包装后观测参数命名不一致的问题；改为通过原模型参数路径观测后，
`selective-ac-second.log` 两卡通过，`selective-ac-eight.log` 八卡通过。
`selective-ac-eight-metrics.json` 保存全部 32 条逐 rank/精度/策略测量。

独立 Standards review 无发现。Spec review 发现在线 target 的显式关闭 checkpoint
配置会继承 draft SAC 策略而被误拒绝；已在 target 默认配置与拓扑视图中隔离该策略。
通过真实 Qwen trainer 初始化、在外部模型加载边界停止的两卡回归复现了原错误，
修复后相同 target 配置在 full/SAC 两种 draft 策略下保持一致。
对应日志为 `selective-ac-target-red.log` 与 `selective-ac-target-green.log`；
Spec 复审无剩余发现。修改的配置、适配器及数值测试通过 mypy 检查；
现有 parallel config 和 activation checkpoint 的 9 个测试也全部通过。

完整阶段 DCP、卸载恢复、真实 5 层 128K 和 draft 生命周期性能由后续票验收。
