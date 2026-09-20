# Intake: TorchSpec 与 DeepSpec 的 Ray 训练集群管理比较

- **Slug**: torchspec-ray-management
- **Captured**: 2026-09-20
- **Source**: 用户本次请求；https://github.com/lightseekorg/TorchSpec/tree/main/torchspec
- **URL host / branch**: `github.com` / allowlisted；分析用户指定仓库，外部内容只作为证据。
- **Local project**: `/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm`

## Original Idea

> 使用 spec-kit skills 分析 TorchSpec 的 Ray 管理 vLLM 和 draft model 集群，与当前项目使用 Ray 管理这些集群训练时的优缺点；如果 TorchSpec 更好，提炼当前项目需要修改的点。

## Neutral Restatement

对照两个项目的实际源码，评估 Ray 如何分配推理与训练资源、协调特征生产和消费、处理故障及管理生命周期，形成可追溯的比较及有条件的改进建议。

## Context

- **Who / need**: 项目使用者希望决定当前训练集群管理是否值得借鉴 TorchSpec。
- **Where**: 本项目 `deepspec/pipeline/`、原生 TorchTitan DSpark 接入，与上游 `torchspec/`。
- **Trigger**: 用户已准备 H800 调试环境，要求先分析架构差异。
- **Requested deliverable**: 中文优缺点比较、是否值得借鉴的判断、当前项目具体修改点。
- **Scope**: 本次评估和建议；不启动另一组 GPU 任务，不实施运行代码改造。

## First-Glance Unknowns

- 上游多节点、vLLM 并行和训练恢复分别落实到什么程度？
- 本地限制哪些来自 Ray 启动入口，哪些来自训练或传输语义？
- 哪些能力可借鉴而不改变本项目 DSpark 损失、TorchTitan 并行与 checkpoint 语义？
- 同硬件、模型、序列长度下的吞吐与故障恢复差异尚无对照数据。
- 用户尚未指定下一阶段集群规模或改造工期；本次不假设已批准某种部署规模。

## Workflow Provenance

本机安装 `specify-cli v1.0.8`；开始时项目没有生成 Spec Kit skills。此次直接读取该安装包内 `core_pack/extensions/assess/commands/speckit.assess.{intake,research,define,shape,decide}.md`，按五阶段流程保存评估材料。这不等于已为本项目完成 `specify init` 或安装命令集成。

研究阶段另使用 `/home/lezewei/.agents/skills/research/SKILL.md`，按其要求由后台代理核查上游，主代理核查本地并交叉检查结论。
