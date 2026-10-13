# TorchSpec 草稿模型训练

本术语表描述草稿模型训练中目标模型、草稿模型和监督样本的关系。

## Language

**目标模型（target model）**：
为草稿模型提供参考预测和中间表征的完整模型。
_Avoid_: 草稿模型、待微调模型

**DSpark 草稿模型（DSpark draft model）**：
学习目标模型的预测及中间表征、用于提出候选 token 的 DSpark 模型。
_Avoid_: 目标模型、视觉编码器

**图文预填充验证**：
对含图片和文本的已有会话进行目标模型前向计算，并以生成一个后续 token 验证请求完成的过程。
_Avoid_: 完整回答生成评测

**assistant 监督段**：
已有 assistant 轮次中的回答内容，包括该轮已有的推理内容与工具调用；是草稿模型回答预测的学习目标。
_Avoid_: 仅最后一轮回答、新生成的单 token

**图文上下文**：
为 assistant 监督段提供条件的 user/tool 消息及视觉输入；不属于回答预测目标。
_Avoid_: assistant 监督段
