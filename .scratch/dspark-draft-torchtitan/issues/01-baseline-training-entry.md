# 01: 固化 DSpark 数值基线与最小训练验证入口

**What to build:** 让后续适配能够通过现有 DSpark 训练入口复现当前工作树的更新结果，并有足够小的入口驱动完整 draft 阶段。先完成必要的局部预重构，保留 DeepSpec 对数据消费、训练循环和生命周期的控制。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** None (can start immediately)。

**Status:** completed

- [x] 记录实施前实际工作树的可复现版本，包括相关未提交修改、模型与训练配置、初始权重、固定 target features、样本顺序、anchors/RNG、冻结集合、参数与通信精度、GAS、拓扑及软件构建；历史 HEAD 或上游版本不能替代该基线。
- [x] 核对现有实际解释器及源码编译 vLLM 的环境，以 TorchTitan commit f6b9152e9bedcc18f5dc339b9f88265e5a07e988 为参考记录组件兼容性；不默认新建环境、重装 vLLM 或更换 PyTorch/CUDA，必须改变基线的具体冲突单独报告。
- [x] 优先复用既有训练入口；只有阶段确实无法被驱动时才补充最小接口。预重构前后输入、各项 loss、梯度和连续更新在预先确定的容差内一致，不创建另一套训练循环。
- [x] 使用真实 Qwen3.8 DSpark 模型类的小规模配置，先 FP32、再 BF16，固定已就绪 features，至少完成两个 optimizer updates；FSDP 对照至少包含两个实际 rank 且 GAS 不小于 2。
- [x] 覆盖同一累积窗口中各 microbatch 有效权重和不同、mask、位置衰减、边界和零分母案例，验证每 microbatch 全局加权均值再按 GAS 等权平均，保留 epsilon 与 confidence target detach。
- [x] 比较模型输出、CE/分布 L1/confidence BCE、全部可训练参数梯度、clip norm、FP32 master weights、Adam moments/计数及 scheduler；冻结 LM head 保持冻结且能够向 draft hidden 传回梯度。
- [x] 根据固定环境中的基线误差预先确定并说明容差，记录实际 GPU/rank 数和可复现运行方式；不以 toy、mock 数学路径、梯度存在性或被 skip 的多卡测试作为真实模型通过证据。


实施提交：`7e67c3e`。验收记录：`doc/benchmarks/dspark_torchtitan_baseline.md`。
