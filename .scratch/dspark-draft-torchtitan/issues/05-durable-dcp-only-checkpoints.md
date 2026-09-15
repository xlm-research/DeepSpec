# 05: 阶段 checkpoint 支持 DCP 独立提交与重启恢复

**What to build:** 每个对齐后的 draft 阶段能够同步提交一份独立可恢复的完整 DCP，并在新进程中接续下一次更新。阶段保存不再强制重复导出 HF 权重，保存失败不会产生虚假的成功交接。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 03：将 draft 阶段对齐完整 optimizer update。

**Status:** completed

- [x] 完整阶段 DCP 包含模型全部必要参数与 buffers（含 frozen 权重）、FP32 master weights、Adam moments/计数、scheduler、各 rank RNG、训练配置、模型/target 身份和样本/阶段进度，不依赖原进程内的状态对象。
- [x] 沿用现有 checkpoint 存储配置；阶段保存无需 HF/safetensors，完整性验证适配 DCP 与必要元数据，能识别缺失或不完整的训练状态。
- [x] 同步完成保存与验证后才提交 checkpoint、发布最新恢复入口并允许推进阶段交接；未提交目录不能被自动恢复选中。
- [x] 通过真实 Qwen 训练入口执行完整 update、保存并退出，在新进程以相同拓扑恢复至少下一次 update；与连续训练对照样本、RNG、loss、梯度和 optimizer/scheduler 状态。
- [x] 恢复时校验模型/训练身份、样本索引、进度和固定拓扑的一致性；配置不匹配不能被当作完整恢复成功。
- [x] 在真实保存/提交边界注入首份 checkpoint 前失败、已有 checkpoint 后失败及不完整写入，验证各 rank 协调停止，不进入下一 target 阶段，且上一成功提交仍可恢复。
- [x] 重启从最近成功提交且验证通过的完整 checkpoint 恢复；首次提交前失败从初始状态重跑。本票不要求保存失败后的原地自动重试。


## Implementation evidence

Implemented with full-state DCP commits, standalone restart validation, invalid-attempt quarantine and aligned suspension. Real two-rank Qwen/Adam/DCP tests cover fresh-process continuous-vs-resumed full trajectories (483.82s), isolated checkpoint storage, five configuration mismatches, four actual filesystem failures (44.68s), fresh replay and replacement of a damaged newer checkpoint (32.08s, full-state/RNG match), and deferred suspension (32.95s). Six artifact validation regressions and the legacy CPU reference passed. Standards and Spec reviews have no remaining findings. See `doc/benchmarks/dspark_phase_checkpoints.md` and `output/dspark_torchtitan_implementation/`. GPU unloading and retention/export continue in 06 and 07.
