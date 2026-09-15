# 07: 中断后核对提交与编排进度

**What to build:** 训练或进程失败后，DeepSpec 能从有效提交恢复，识别已经完成的更新，避免消费不完整特征或错误推进阶段。

**Blocked by:** 06：打通真实 vLLM 特征生产与两阶段训练。

**Status:** completed

验收依据：母规格《DeepSpec 编排下的 TorchTitan DSpark draft 训练与阶段恢复》与 ADR-0001、ADR-0003、ADR-0004。DeepSpec 编排两端；TorchTitan 拥有 draft 训练及配置。保留 DSpark 数学、精度与完整 update 语义，只优化 draft，沿用现有环境和源码编译 vLLM。

验证优先使用 DeepSpec 阶段入口 → 真实 TorchTitan 训练进程，固定 target features 隔离数值对照，真实 vLLM 交接另有证据。票 01 的旧入口只用于兼容基线；不以内部 wrapper 调用、toy、mock 数学或 skip 代替真实训练证据。

- [x] 通过真实阶段入口覆盖训练异常、worker 非正常退出、DCP 写入/提交失败和遗留 worker；故障通过实际子进程或文件提交边界注入，不以内部 helper 调用断言代替可观察结果。
- [x] 覆盖首份 checkpoint 前失败、已有提交后失败和提交成功但 DeepSpec 尚未记录完成就中断；重启选择最近有效提交，已提交 updates 不重复，未提交工作按恢复游标重跑。
- [x] 恢复前核对训练配方、target/特征身份、计划、分区、拓扑与 checkpoint；不兼容或不完整状态给出具体错误，不静默选择无关恢复点。
- [x] 缺 shard、错误 tokens/样本、层序/dtype/shape/最终层语义不符及部分 feature 写入不能标记为就绪或被训练消费；失败不得删除仍需恢复的分区和上一有效提交。
- [x] 验证重启后样本/RNG、loss、完整优化状态与连续训练参考一致，scheduler 不重新 warmup；保存失败和未完成资源释放都阻止下一 target 阶段。
- [x] 协调停止当前任务后可由新进程重新启动；不要求原地自动重试、不保存半步梯度、不引入跨拓扑恢复。

覆盖母规格 User Stories：24、25、30、50、60、62、63、64。


## 验收证据

2026-09-14，`output/dspark_torchtitan_orchestration_20260914/phase-interrupt-v2-test.log`：真实两卡 native 阶段入口四种失败与恢复全部通过（1102.611 秒，无 skip）。覆盖首次提交前输入损坏、提交一次后输入损坏、提交后父进程 SIGKILL、提交后 GPU worker SIGKILL；DeepSpec 尚无阶段完成记录时由 durable commit 恢复。每次确认 worker 释放后启动新进程；全部恢复轨迹与连续两步参考逐项 bitwise 相同，包含样本、RNG、全部参数、FP32 master/Adam、scheduler 和梯度。

`retention-save-failure-test.log`：真实 DCP payload 写入后 commit rename 失败，旧提交仍在，无阶段成功记录，GPU worker 全部退出（127.784 秒，通过）。`process-supervisor-v2/result.txt` 覆盖父进程被杀后独立 session 后代的终止和回收。异常路径避免在 finally 中等待仍停留于 collective 的其他 rank，由 Elastic 与 owned-process supervisor 完成退出。

`native-inputs-v3-test.log`：实际 tokenizer/parser、既有完整/两片/三片特征重组和无效输入拒绝共 3 项通过，无 skip；缺失 shard、部分写入、错 tokens/mask、层序/dtype/shape/最终层语义均拒绝。恢复入口核对 run/plan/topology/完整配方身份，使用 native commit 核对进度；持有目录锁，保留未提交特征。

`pending-hf-export-recovery.json`：已提交第 3 步的可选 HF 导出补齐，包含 CPU 进程启动共 69.180 秒；完成 updates 前后均为 3，未重复训练。
