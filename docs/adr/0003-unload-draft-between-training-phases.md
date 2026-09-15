---
status: accepted
date: 2026-09-14
---

# Draft 训练阶段之间卸载完整 GPU 训练状态

用户要求在一个 draft 训练阶段结束后从 GPU 卸载完整 draft 训练状态，在下一 draft 阶段开始时重新加载并继续训练。卸载包括参数与必要 buffers、梯度、FP32 master weights、Adam moments 和其他 optimizer 状态，以及 draft 持有的临时计算、通信和预取 batch 缓冲。当前 Qwen 路径在 target 特征生成期间保留 draft 常驻；新设计采用阶段间卸载，接受由此增加的状态保存与恢复职责。用户同时要求尽量提高性能，优化须保持完整卸载和恢复契约。

训练连续性仍遵循 [ADR-0001](0001-preserve-dspark-training-semantics.md)。用户已同意将缓存分区对齐完整 optimizer step，保持样本顺序、GAS 和每次 update 的分组，允许分区大小和数量变化。这样可以避免为阶段切换引入 FSDP 内部未同步累计梯度的快照协议；正常阶段切换时没有待继续累积的半步梯度。

资源卸载控制归 DeepSpec，见 [ADR-0004](0004-separate-feature-delivery-and-draft-training.md)：DeepSpec 决定两端是否卸载，vLLM 和 TorchTitan 各自执行本端释放并报告完成。Draft 的完整 update、checkpoint 提交以及下一阶段恢复由 TorchTitan 执行；DeepSpec 收到资源释放完成后调度下一阶段。

用户在 ADR-0004 的 Q9 中选择保存后退出本轮训练进程，下阶段由 DeepSpec 重新启动 TorchTitan 并恢复。第一版以所有训练 worker 退出并释放其 GPU 资源实现阶段卸载，进程启动和通信组重建计入 draft 耗时。

阶段交接按以下顺序设计：完成本次 optimizer update 与对应 scheduler/进度更新，结束在途计算和通信（PP 路径须排空流水线），同步将完整 checkpoint 落盘并确认提交成功，再释放 draft 持有的 GPU 状态。下一 draft 阶段重建运行对象并恢复模型、FP32 master weights、Adam moments/计数、scheduler、RNG 和数据进度。模型重建与 target 阶段不能扰动恢复后的 draft RNG 序列。

用户已确认同一训练任务中各 draft 阶段使用相同并行拓扑；不同训练任务可以选择不同的合法组合，target 沿用其独立配置。跨拓扑恢复另行设计。每阶段落盘使恢复不依赖原训练进程中的 CPU 快照，具体存储位置依环境细化。

本 ADR 的完整恢复要求适用于新训练流程自身的阶段续训和从已提交 checkpoint 重启。[ADR-0004 的 Q6](0004-separate-feature-delivery-and-draft-training.md) 已确认本次保证新流程完整恢复，旧 DeepSpec 任务的完整训练状态转换不纳入本次范围。

每阶段只提交完整 draft DCP 与必要 config/identity/progress 元数据，保留模型参数（包括 frozen 参数）、optimizer 和全部恢复状态；HF/safetensors 在评估需要或最终交付时导出。新 checkpoint 提交成功后滚动保留最近两份完整阶段 checkpoint，显式里程碑与最终产物单独保留。

阶段保存失败时协调停止当前训练任务，不进入下一 target 阶段。重启后从最近成功提交并验证的完整 checkpoint 恢复；未提交目录不作为恢复入口，上一成功提交不因新保存失败而被删除。首次提交前失败则从初始状态重跑，首版不要求原地自动重试。

“完整卸载”以释放 draft 持有的 GPU 状态为验收目标，第一版检查所有训练 worker 退出和相关资源释放，不把其他进程的 GPU 占用计为 draft 泄漏。当前每 epoch 洗牌后只训练完整 global batch，正常 epoch 结束不存在半步梯度；新的分区方案保持这一取样规则。

当前仍为设计讨论，未授权实现或执行训练。
