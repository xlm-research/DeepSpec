# TorchTitan 阶段编排实施基线

本轮按用户批准的 30 票清单实施。票 01 已完成；此记录不表示新 TorchTitan 训练入口、阶段恢复或 128K 已验收。

当前 DeepSpec HEAD 为 `c75ea27`，TorchTitan 为 `f6b9152e9bedcc18f5dc339b9f88265e5a07e988`。实施前实际工作树已保存到 `output/dspark_torchtitan_orchestration_20260914/`，含源码归档、文件散列、Git diff/status 和环境清单。相较最初基线归档，有 19 个文件变化、14 个新增文件，详见 `baseline-file-delta.json`；现有用户修改包含在本轮快照中。

复用最初 `numerics-final` 的 8 份固定产物，全部通过已有 SHA256 清单验证。使用原训练解释器、PyTorch 2.13.0+cu130 和两个实际 B300 GPU，重放真实 Qwen DSpark 小规模模型。FP32/BF16、GAS2、不等分母、零分母与两个完整 updates 的全部既有对照通过；每 rank 测试约 43.5 秒，退出码 0，无 skip。日志为本轮输出目录的 `baseline-replay.log`。

重放使用已归档的权重、features、RNG 和预期结果，对照全部可训练梯度、clip norm、参数、FP32 master/Adam、scheduler、数据进度及日志分项。沿用原 rtol=1e-4、atol=1e-6；此容差不自动推广到新增并行计算。现有固定特征入口已足够复用，因此票 01 没有新增训练循环或进行无必要的预重构。

训练解释器仍为 `env.sh` 选择的现有环境。本机八张 B300 在启动前空闲。已安装的 spmd_types 0.2.5、torch_remat 0.2.0 和 tyro 1.0.16 可继续使用；完整 TorchTitan Trainer 的导入发现缺少 grain。票 02 将先解析所需依赖，约束所有现有包版本，并记录安装结果，不能把旧 AC 组件导入成功误报为完整 Trainer 可用。

可复现命令沿用原基线记录，将 `DEEPSPEC_BASELINE_REFERENCE` 指向原 `numerics-final`，使用现有训练解释器执行两 rank 的 `tests.test_dspark_training_baseline`。本轮不改写原数值参考。
