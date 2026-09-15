---
status: superseded in part by ADR-0004
date: 2026-09-14
---

# 保留 DSpark 训练循环，先以 Qwen3.8 验证 TorchTitan 组件接入

2026-09-14 后续决定：用户已确认完整 draft 训练职责和配置归属 TorchTitan，尽可能复用其已有配置及组件，本次先 Qwen3.8 DSpark，见 [ADR-0004](0004-separate-feature-delivery-and-draft-training.md)。下文中 DeepSpec 持有 draft 训练循环及训练状态管理的约定已被替代；DeepSpec 继续调度两端、决定训练多少批数据并控制两端资源卸载，同一批 GPU 分阶段交替。其余未被修改的训练语义和验证约束继续沿用。以下保留原决定及其取舍依据。

本轮采用组件级接入：DeepSpec 继续拥有 DSpark 的训练循环和 target/draft 生命周期，在现有接口处适配选定的 TorchTitan 模块。用户已确认先用 Qwen3.8 DSpark dense draft 验证；训练语义遵循 [ADR-0001](0001-preserve-dspark-training-semantics.md)。

本轮优化对象限定为 draft model。TorchTitan 组件、并行适配、训练状态管理及性能调优均作用于 draft；target model 与 vLLM 特征生产作为既有监督来源，其模型实现、推理内核、并行配置和生成策略不在优化范围。已经确认的缓存分区对齐属于训练阶段交接协议，保持 target 监督内容与顺序。

用户明确 vLLM 为源码编译安装，继续使用现有运行环境。TorchTitan 接入以现有环境为兼容基线，保留 vLLM 的解释器、源码、编译产物及依赖组合；不预设新建 draft 环境，也不把接入 TorchTitan 视为更换 vLLM 或其 PyTorch/CUDA 栈的授权。

完整 TorchTitan Trainer 接管曾作为备选，但需要同时迁移模型协议、batch/loss 契约和阶段生命周期。保留当前循环能在复用基础设施时缩小变更范围，并以现有 DSpark 实现作为对照。

## 职责约束

用户在实施过程中补充明确：draft model 的实际定义应放在
`torchtitan/torchtitan/models/` 下的独立 draft 模型目录。该目录承载模型主体、
draft attention/decoder block、配置及模型专属并行适配；DeepSpec 的训练入口调用
这里定义的模型。保留 DeepSpec 训练循环并不表示模型定义继续归属 DeepSpec。
模型归属调整仍须通过既有固定特征数值对照，保持 DSpark 数学与 checkpoint 状态连续性。
用户进一步要求优先复用 TorchTitan 已有模型组件和实现包。新 draft 按 TorchTitan
原生模型接口组织，通用层、attention 内核、FFN、归一化、并行和状态适配优先使用
现有实现。自有代码限定在 DSpark 所需的输入/监督契约及经数值验证确有必要的适配，
不能仅迁移旧模型文件并继续手写 TorchTitan 已具备的通用能力。

- DeepSpec 训练循环继续协调 batch 消费、forward/loss、backward、梯度累积、optimizer update 和保存恢复时机。所选组件在对应接口内承担其职责。
- target 特征生产与 draft 训练的交接仍由 DeepSpec 管理。用户已同意将缓存分区对齐完整 optimizer step，保持样本顺序、GAS 和 update 分组，并在阶段间卸载完整 GPU 训练状态。
- 用户已确认解耦 target 特征生产布局与 draft 消费布局：编排层固定既有 producer 配置及样本规划，draft 专用索引记录样本身份、顺序、原始文件与分片归属，reader 根据 draft 拓扑完成读取和分发。沿用 target worker、推理配置与原始 feature 内容；消费侧适配开销计入 draft。
- 每个 draft 的参数、mesh、梯度同步和训练状态都有明确的管理方；组件接入通过现有入口委派，避免两个 runtime 重复初始化或同步同一模型。
- 首阶段以 Qwen3.8 DSpark dense draft 建立数值与恢复对照。MoE 接入时另行验证专家参数布局、通信和梯度归一化。

首阶段模块已确定为 TorchTitan SelectiveAC，初始验证环境为单机八卡 Qwen3.8 DSpark，最终包含真实 draft 尺寸与 128K 输入。用户随后将总体范围扩展为适配 TorchTitan 的所有并行方式，并确认阶段间完整 GPU 状态卸载、同步 checkpoint 落盘与固定 draft 拓扑恢复，见 [ADR-0003](0003-unload-draft-between-training-phases.md)。交付顺序已确认：先完成 Qwen、SelectiveAC 与阶段卸载恢复，再验证 dense 并行组合，最后以 GLM-5.3-Flash DSpark 验证 MoE EP。

全并行适配覆盖各能力及明确的合法组合，后续接入以 `spmd_types` 为主要后端，PP 首先采用 1F1B，允许 CP 保留 DSpark 模型专用通信实现。用户要求在训练语义和状态连续性约束下尽量提高性能；具体支持矩阵、性能方案和剩余决策见 [设计草案](../../doc/qwen38_torchtitan_component_design.md)。旧完整 Trainer 迁移文档保留为历史分析。

性能统计限定为 draft 训练、保存、卸载和恢复的累计耗时，排除 vLLM 推理生成 target features 的耗时。共享基础设施的调整需通过 draft 专用入口或显式配置生效，保证 target 路径的行为保持原样。

当前处于设计讨论阶段。上述选择的确认不代表授权实施；代码改动和训练执行等待用户明确要求开始实现。
