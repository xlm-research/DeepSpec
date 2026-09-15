# DeepSpec

本项目训练和评估用于推测解码的 draft model，使候选 token 与 target model 的输出分布相匹配。

## Language

**Draft model（草稿模型）**：
在推测解码中提出候选 token、交由 target model 验证的模型，也是本项目训练的对象。

**Target model（目标模型）**：
在推测解码中提供验证分布，并在 draft 训练中提供监督信号的参考模型。训练语境中的 teacher 指这一角色。

**Target features（目标模型特征）**：
Target model 处理给定 token 序列时产生、供 draft model 使用的隐藏层表示。

**Feature partition（特征分区）**：
一次目标特征生产后供 draft 训练消费的一份有序样本及其特征，是阶段调度的数据单位。一个特征分区可以包含多次完整的训练更新。

**Target feature production phase（目标特征生产阶段）**：
为后续 draft 训练准备一批目标模型特征的阶段，与 draft 训练阶段交替。

**Draft training phase（draft 训练阶段）**：
消费已准备好的特征分区进行 draft 训练的阶段，与目标特征生产阶段交替。

**Phase orchestration（阶段编排）**：
对目标特征生产与 draft 训练的阶段顺序、工作量和资源交接的安排。
