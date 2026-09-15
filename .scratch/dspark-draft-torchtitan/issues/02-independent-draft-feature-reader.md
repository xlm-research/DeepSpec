# 02: 固定 target 生产配置，支持独立的 draft 消费布局

**What to build:** 训练操作者能够保持既有 target 特征生产方式，同时为 draft 选择独立的已支持消费布局。通过稳定样本索引消费既有完整或分片 features，得到原有监督内容和更新输入。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 01：固化 DSpark 数值基线与最小训练验证入口。

**Status:** completed

- [x] 显式固定现有 target 模型身份、特征层、推理参数、owner/device mapping、输出 CP 布局、样本规划、解释器和源码解析位置；draft 的 DP/TP/CP 配置不再决定这些生产设置，保持 vLLM worker 和原始 feature 内容不变。
- [x] draft 输入索引包含稳定样本身份、全局顺序、epoch/分区、原始 feature 及 shard 归属、逻辑 microbatch/update 归属和恢复游标关联；能够定位下一逻辑 microbatch。
- [x] 通过真实 Qwen 训练入口消费 producer CP1 的完整 features，以及同一样本既有 CP 分片重组后的 features；tokens、loss mask、hidden features、有效长度与 token 顺序同参考一致。
- [x] 至少用两种现有合法 draft 消费布局读取相同生产缓存，在匹配逻辑 microbatch 语义的参考下完成真实 loss 和更新对照；新的 SP/CP/PP 执行能力分别由后续票交付。
- [x] 仅修改 draft AC/compile 或消费拓扑时，固定 target 配置和监督契约保持一致；生产布局校验不会因无关 draft 变换字段不同而误拒绝。
- [x] 恢复游标与样本索引一致；缺失、不完整或身份不匹配的 feature shards 在参与训练更新前被识别，不能静默重复、丢失或错配样本。
- [x] 缓存清理依据实际 producer 文件所有权，并等待全部相关 draft 消费者完成；覆盖消费者进度不同的情况。索引、读取、重组和重分发归入 draft 性能统计范围。


Implementation: `5fd7557` — evidence and commands in `doc/benchmarks/dspark_feature_phases.md`.
