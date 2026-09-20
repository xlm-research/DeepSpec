# Concept: 选择性借鉴 TorchSpec 的管理层

- **Slug**: torchspec-ray-management
- **Created**: 2026-09-20
- **Recommended option**: B — 保留执行与训练语义，统一拓扑和生命周期
- **Input**: [problem.md](problem.md)、[research.md](research.md)

## Options

### Option A — 保留现状，继续现有布局调试

- **Sketch**: 使用已验证的单机/两机布局；需要新布局时继续做局部调整。
- **Appetite**: small；仅代表相对投入范围，用户未给出工期预算。
- **Trade-offs**: 最近验证结果可直接复用，迁移风险最低；多 consumer 节点和统一运行管理仍受限制。
- **Rabbit holes**: 针对每次新配置不断添加分支，使单机和多机路径继续分化。

### Option B — 保留执行与训练语义，统一拓扑和生命周期

- **Sketch**: 借鉴上游显式角色计划、推理组/训练组、控制循环的分工。用户从同一组声明和运行状态理解单机与多机任务，底层继续使用本项目原生 vLLM Ray 后端、torchrun/TorchTitan 和字节账本。
- **Appetite**: medium；首轮只做现有布局等价迁移，再扩展一种可验证的多训练节点布局。此项为范围建议，不是工时承诺。
- **Trade-offs**: 获得较大的管理层收益，同时保留已验证训练路径；需要处理 vLLM 自管 PG 与外层资源计划之间的边界，并承担真实多机验证成本。
- **Rabbit holes**: 一开始就强制所有后端共享同一个 PG；将 TorchTitan 改成每 GPU 一个上游 trainer actor；同时引入动态扩缩容、RDMA 和无损恢复。

### Option C — 迁移到 TorchSpec 的整套 engine/trainer/controller

- **Sketch**: 采用上游训练入口、mp 引擎组、TrainerActor、队列与传输生命周期，围绕上游实现适配当前模型和数据。
- **Appetite**: large；需要重新验证模型语义、128K 内存和 checkpoint 接口，无法仅作为 Ray 启动器替换。
- **Trade-offs**: 可以直接获得较完整的多引擎/训练运营功能；但原生 vLLM Ray DP、TorchTitan DSpark 及本地严格特征契约都需要重新评估。上游也没有证明透明恢复或同条件性能更好。
- **Rabbit holes**: 数值等价验证、分布式训练栈迁移、跨版本 vLLM 内部 API、既有 checkpoint 转换。

## Recommendation

推荐 B。TorchSpec 在可配置拓扑与职责划分上有明确可借鉴的实现，而本地的长序列预算、输入身份与多 reader ACK 已直接服务于当前训练目标。扩大管理能力的收益有源码依据，整体更换执行/数据链路的收益没有同等证据。

A 对明天的调试依然合理；B 是后续改造方向。C 当前不进入实施建议。

## Out of Scope

首轮不改变训练数学语义，不引入自动改变 world-size，不将 colocate 当作 GPU 显存轮换，不切换 mp/Ray executor，不以 RDMA 作为拓扑重构前置条件。恢复机制单独评估。

## Assumptions to Validate

- 增加 draft 节点确实是后续需求；具体规模由下一阶段规格承接。
- 当前 vLLM 原生 Ray DP 的资源 ownership 能通过明确适配边界纳入统一计划；未经验证不预留第二份 GPU。
- 训练组可以继续由每节点一个 launcher 管理 torchrun；跨节点 rank/mesh 与内存预算必须通过实际运行验证。
- 若将来引入长度均衡，允许改变的仅是经过批准并持久化的输入计划，不能随推理完成顺序改变已承诺的样本集合。
