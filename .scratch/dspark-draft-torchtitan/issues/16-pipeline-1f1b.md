# 16: 在现有训练循环中支持两阶段 PP 1F1B

**What to build:** 训练操作者能够在保留的 DeepSpec 训练循环内使用两阶段 PP 1F1B。Pipeline schedule 执行内部 forward/backward，DeepSpec 继续控制逻辑 microbatch、optimizer update 和阶段交接。

验收以母规格《保持 DSpark 训练语义的 TorchTitan draft 并行适配与阶段卸载》及 ADR-0001/0002/0003 为准，保留现有 DSpark 训练语义、draft/target 边界和运行环境。

**Blocked by:** 09：通过 TorchTitan 组件支持复制 DP 与纯 FSDP2。

**Status:** ready-for-agent

- [ ] 完成八卡候选 G：DP shard4 × PP2、TP1、CP1；以真实 Qwen DSpark 模型类形成可运行的两阶段划分，5 个真实 draft layers 不要求等分，也不默认承诺 PP8。
- [ ] stage 间正确传递 draft query、可微 teacher-derived context、对应梯度及必要监督输入；不能仅按 decoder 层自动切分而遗漏 context 的完整反向路径。
- [ ] 流水线物理 microbatch 切分保留原逻辑 microbatch 的样本归属、分母、GAS 等权平均和 RNG；PP stages 不增加监督分母，不提前更新或改变优化配方。
- [ ] DeepSpec 保持 GAS、跨参数梯度裁剪、optimizer/scheduler、进度及 checkpoint 控制；各 stage 状态有明确管理方，schedule 不重复归约或缩放梯度。
- [ ] 真实 Qwen 短序列对照非 PP 的匹配参考，覆盖不等有效分母、多 microbatch 和至少两次 updates，比较可微 context 路径、全部可训练梯度、clip norm、master weights、Adam/scheduler。
- [ ] 每个阶段在完整 update 后排空 pipeline 及在途通信，再同步提交所有 stage 的完整 DCP 后卸载；相同 PP 拓扑下恢复全部模型/optimizer/RNG/样本状态并继续下一次 update。
- [ ] 覆盖新进程恢复和多次阶段循环，验证全部 stage GPU 状态释放、保存失败不推进 target 交接，记录实际八卡参与及分项耗时；target 保持既有生产配置。

