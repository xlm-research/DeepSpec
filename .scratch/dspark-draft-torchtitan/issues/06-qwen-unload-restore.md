# 06: Qwen 阶段间完整卸载并恢复 GPU 训练状态

**What to build:** Qwen 在相邻 draft training phases 之间释放完整 GPU 训练状态，让下一 target 阶段使用释放的资源；随后从已提交 DCP 重建，继续原有训练轨迹。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 04：Qwen 接入 TorchTitan SelectiveAC；05：阶段 checkpoint 支持 DCP 独立提交与重启恢复。

**Status:** ready-for-agent

- [ ] 通过保留的 DeepSpec 阶段入口执行完整 update、scheduler/进度更新、结束在途计算通信、同步提交完整 DCP、释放 draft GPU 状态，再允许进入下一 target 阶段；保存失败沿用停止交接语义。
- [ ] 释放模型参数和必要 buffers、梯度、FP32 master weights、Adam 状态、FSDP 临时状态、预取 batch 及其他 draft GPU 引用；不能用 CPU 快照或仍常驻的 GPU 状态替代已提交 DCP。
- [ ] 下一阶段按同一拓扑及一致的 SAC/FSDP2 适配流程重建模型与 optimizer，完整恢复权重、master weights、Adam、scheduler、样本索引及进度。
- [ ] 训练 RNG 在重建与状态加载完成、下一次 draft forward 前恢复；改变初始化或 target 阶段的随机消耗，不会改变恢复后的 anchors 和 draft 随机序列。
- [ ] 用固定 features 对照连续训练与跨阶段卸载恢复，覆盖不等有效分母、GAS 不小于 2 及至少两次 updates，比较下一样本、各 loss、梯度、clip norm 和完整 optimizer 更新。
- [ ] 覆盖新进程恢复以及单机八卡 DP shard2 × TP4、CP1 的短序列阶段循环，不能依赖上个阶段存活的 CPU/GPU 模型对象。
- [ ] 连续多个阶段记录峰值、卸载后的稳定显存及资源归属，证明 draft 状态释放且无随阶段增长的残留；将 CUDA context 与通信运行时基础占用单独识别，不要求显存归零。
- [ ] 使用既有 vLLM 交接确认 target 阶段可以继续，保持 target 解释器、源码、推理设置和监督契约不变。

