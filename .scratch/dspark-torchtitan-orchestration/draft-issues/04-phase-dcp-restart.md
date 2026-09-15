# 04: 提交完整 DCP，退出并跨进程续训

**What to build:** DeepSpec 能连续调度两个固定特征分区：TorchTitan 在完整 update 后提交 DCP 并退出，下一进程恢复后继续原有训练轨迹。

**Blocked by:** 03：通过 TorchTitan 完成两卡 FSDP2 更新。

**Status:** draft

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [ ] 在既有阶段入口建立统一请求与结果：请求包含已解析配方、全程计划、特征分区、完整 update 停点及恢复入口；结果包含有效提交、已消费范围、完成 updates 和下一消费位置。
- [ ] 每阶段同步提交模型及 frozen 参数/buffers、FP32 master 与 Adam 状态/计数、scheduler、各 rank RNG、训练身份/配置、样本和分区进度；使用 TorchTitan DCP 状态扩展，不以默认 step/token 状态代替完整恢复。
- [ ] 首阶段从初始状态训练；完整 update、scheduler/进度推进、在途计算通信完成及 DCP 提交后，所有训练 worker 退出。DeepSpec 确认退出与 draft GPU 资源释放后才调度下一阶段。
- [ ] 下一独立进程保持同拓扑恢复，在构建与加载完成后、下一训练计算前恢复 RNG；改变初始化随机消耗不改变恢复后的 anchors、样本与更新。
- [ ] 连续训练与跨进程恢复比较后续 loss、梯度、master/Adam/scheduler、RNG 和消费进度；至少两个 updates、多阶段执行，无随阶段增长的 draft 资源残留。
- [ ] 全程 scheduler 长度与当前阶段停点分开，跨阶段不重新 warmup；只允许完整 update 分区，保持既定 GAS、样本顺序与 epoch 完整 global-batch 截断。
- [ ] 阶段保存沿用既有存储配置并支持 DCP-only，无须 HF 文件证明完整性；保存失败不产生成功结果或推进阶段，未提交目录不作为恢复入口。

覆盖母规格 User Stories：13、14、17、18、19、20、26、49、57、58、59、60。

