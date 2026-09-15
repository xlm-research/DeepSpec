# 03: 将 draft 阶段对齐完整 optimizer update

**What to build:** 每个 draft training phase 都在完整 optimizer update 后结束，使后续持久化与卸载不需要半步梯度快照，同时继续执行原有 DSpark 样本和更新序列。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 02：固定 target 生产配置，支持独立的 draft 消费布局。

**Status:** completed

- [x] 将缓存分区边界调整到完整 optimizer update 及对应 scheduler/进度更新之后，允许分区大小和数量改变，且阶段内可以包含多个完整 updates。
- [x] 在旧分区边界落在 GAS 窗口中间的案例中，对照调整前后的实际训练样本顺序、逻辑 microbatch 归属及每次 update 分组，结果一致。
- [x] 保持原有 GAS 与 microbatch 等权平均，不提前 optimizer.step，不缩短累积窗口，不补样本，也不为阶段卸载额外丢弃样本。
- [x] 保留每 epoch 洗牌后按完整 global batch 截断的规则，以及完整 update 的停止边界；覆盖 epoch 尾部、多个分区和 max-train-steps 停止场景。
- [x] 阶段索引及下一 microbatch/update 游标与实际消费一致，可以供阶段 checkpoint 恢复使用；正常阶段边界不存在待继续累积的梯度。
- [x] 通过真实 Qwen 训练入口，使用不等有效分母且 GAS 不小于 2 的固定 features，比较边界调整前后至少两次连续更新及 optimizer/scheduler 状态。


Implementation: `5fd7557` — evidence and commands in `doc/benchmarks/dspark_feature_phases.md`.
