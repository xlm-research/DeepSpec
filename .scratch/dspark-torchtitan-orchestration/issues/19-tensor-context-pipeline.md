# 19: 联合启用 TP、CP 与 PP

**What to build:** 操作者能在一个合法的 TP/CP/PP 八卡布局中完成真实 DSpark 更新、所有 stage 的保存退出和新进程恢复。

**Blocked by:** 17：联合启用 TP 与 CP；18：支持 TorchTitan 两阶段 PP 1F1B。

**Status:** in-progress

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [x] 交付候选 H：TP2 × CP2 × PP2，DP replicate/shard 均为 1；联合配置明确双输入的 stage/layout、监督所有权及可微 context 的反向路径。
- [ ] 固定 producer features、逻辑 microbatch 和训练配方，对照匹配参考，验证分母、GAS 与 TP/CP/PP 归约没有重复。
- [ ] 通过真实 Qwen 短序列至少两个 updates，对照各 loss、全部 stage 的梯度、clip norm、master/Adam/scheduler，覆盖不等有效分母。
- [x] 排空 pipeline 和通信后同步提交所有状态，全体 worker 退出并释放资源，同拓扑新进程恢复下一 update 且 RNG/样本连续。
- [ ] 覆盖联合保存失败与多阶段资源释放，记录实际八卡和运行限制；单能力证据不代替本联合布局验收。

覆盖母规格 User Stories：35、36、40。

## 2026-09-15 实现与联合数值检查

按用户最新要求停止逐项测试，直接运行本票八卡布局。五层真实 Qwen、固定
producer features、两个 updates、逻辑 GAS2；原生 1F1B 的层划分为 3/2，
query 与可微 context 同时跨 stage。运行产物位于
`outputs/dspark_dense_12_19_20260915_v2/tp_cp_pp/`。

严格数值对齐尚未通过，保持 in-progress：FP32 输出、loss、全部梯度、clip
norm 和 Adam moments 通过，但模型/master 少量元素最大差 `8.61e-6`；
BF16 logits 最大差 `0.00879`，两步整体梯度相对 L2 差为 `1.08%`、`2.44%`。
标签、teacher logits、mask 和三个 loss 分母一致。容差未放宽，完整报告见
`doc/benchmarks/dspark_native_dense_12_19.md` 与产物中的 `alignment-report.json`。

本轮按用户收窄后的范围未单独执行联合提交失败注入。

FP32/BF16 均已通过完整恢复：754 个字段、344 个张量和下一步更新/输出/RNG/
游标精确一致。用户随后明确“先不用改 cp”，因此暂停 CP 数值修复并撤回刚加入
的 all-to-all 实验，继续保留原有 ring CP；不得将本票改成 completed。

用户追加要求关闭 CP 做对照：四卡 `DP1 × TP2 × CP1 × PP2` 已跑完两个
updates。BF16 的输出/loss/全部梯度/参数/master/Adam/scheduler 均逐位一致，
仅首步 clip norm 为 `7.09375` 而参考为 `7.125`；FP32 输出和梯度在原有
容差内，参数最大差 `5.35e-6`。产物在
`outputs/dspark_dense_19_no_cp_20260915/tp_pp/alignment-report.json`。
该追加对照不替代本票开启 CP 的联合验收。
