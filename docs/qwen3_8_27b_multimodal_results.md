# Sensenova Flash Lite v42 图文 DSpark 验收记录

日期：2026-10-13。环境：单机 8 × H800 80GB，目标推理 TP=4，草稿训练四卡 FSDP。
目标模型：`/mnt/afs_agents/hongjiawei/delivery/sensenova-flash-lite-v42-20260830-dpo-mtp`。
使用现有 `h800conda.sh` 环境和当前项目的本地 vLLM。
图文/文本推理与训练目标特征提取的 backend 均为 vLLM。
该环境实际解析的模块入口为
`/mnt/afs_share/lezewei/TorchSpec/vllm/vllm/__init__.py`。

## 图文 prefill

从 `mydata/image-text-327.jsonl` 选取零起始行号 114、217、231。
本地图片相对 `/mnt/afs_toolcall/juxiaolong1/Data/draftData` 解析；完整保留已有会话。
实际 GPU 运行通过，每个请求仅生成 1 token；捕获的训练输入不含这个新增 token。

| 源行号 | 图片数 | processor 展开后 token 数 | assistant 监督 token 数 | 可用 DSpark anchor 数 |
| --- | ---: | ---: | ---: | ---: |
| 114 | 3 | 11296 | 3044 | 3040 |
| 217 | 1 | 18792 | 4658 | 4648 |
| 231（含 tool 图片） | 1 | 77523 | 36384 | 36364 |

五个辅助槽为 `[4, 16, 32, 48, 60]`，最终 pre-norm 槽为 64。
实际从 Mooncake 生产消费者读取 per-layer manifest，辅助张量为
`[prompt_tokens, 25600]`，最终状态为 `[prompt_tokens, 5120]`；所有值有限，
token IDs、mask、hidden states 长度一致，视觉 token 监督为零。
报告：`outputs/qwen3.8-27b-image-text-1node-prefill/summary.json`。
运行日志：`/tmp/torchspec_image_text_prefill_retry.log`。

## 训练与恢复

三条原样本派生为 160 条独立 ID 的重复记录，global batch 为 16。
首次运行完成 10 个真实 optimizer steps，各步 loss 有限、草稿 FP32 参数更新非零。
目标模型及视觉编码器不参与优化，训练侧共享 embedding/head/norm 冻结。
固定 `lr_total_steps=11`，第 10 步实际使用学习率 `7.93732335844e-6`。

第 10 步 checkpoint：`outputs/image-text-smoke-10/checkpoints/iter_0000010/`，
含 model、optimizer（包括 FP32 master 参数）、lr_scheduler、rng 与迭代元数据。
`global_step=iteration=10`，`next_step=10`，`world_size=4`。
rank 0 连续性校验片段的更新后 FP64 求和值为 `-0.15509673717497208`。
训练日志：`/tmp/torchspec_image_text_train_10.log`。

恢复运行累计至第 11 步并正常退出，checkpoint 保存于
`outputs/image-text-smoke-11/checkpoints/iter_0000011/`。
日志确认加载 model、optimizer、FP32 master 参数及 scheduler；rank 0 更新前
校验值与第 10 步更新后完全一致。第 11 步使用学习率 `2.02535131928e-6`，
全部参与优化参数的跨 rank 最大变化为 `2.08360143006e-6`；rank 0 校验片段的
更新后求和值为 `-0.15509541964274831`。此次确实继续了草稿参数优化。
恢复日志：`/tmp/torchspec_image_text_train_11.log`。

第 11 步本地 TensorBoard 指标如下，全部为有限正值，三个启用的损失项均有贡献。
`train/avg_loss` 是现有日志口径；各分量单独列出。

| 指标 | 值 |
| --- | ---: |
| `train/avg_loss` | 8.8183012008667 |
| `train/ce_loss` | 8.818300247192383 |
| `train/l1_loss` | 1.9613405466079712 |
| `train/confidence_loss` | 0.09369590133428574 |
| `train/grad_norm`（裁剪前） | 0.774150550365448 |

元数据、参数连续性及指标核对结果保存在
`outputs/image-text-smoke-11/acceptance.json`。第 11 步完成后 scheduler 降至零，
因此判定恢复步使用正学习率的依据是 optimizer 执行时的 `update_probe` 日志。

## 原文本路径

使用原 `examples/qwen3.8-27b-1node/config.yaml`、原会话 loader 和默认
thinking 仅最后一轮监督规则，真实 GPU 回归正常退出。
输入含两轮 assistant 的纯文本会话，共 100 tokens；只生成 1 token，
有效监督 39 tokens。取回的 mask 与原 loader 的 packed mask 逐 token 完全一致，
辅助/最终状态分别为 `[100, 25600]`、`[100, 5120]`，所有值有限。
报告：`outputs/qwen3.8-27b-image-text-1node-text-prefill/summary.json`。
运行日志：`/tmp/torchspec_legacy_text_prefill.log`。

## 监督回归与代码复查

真实 checkpoint 的 tokenizer/processor 回归已通过。测试以独立字符跨度和
真实 tokenizer offsets 核对完整展开 mask，覆盖三轮 assistant 正文、reasoning、
两组完整 tool 参数、user/tool 上下文以及视觉位置。
DSpark 的 CE、L1、confidence 共用该监督权重。

规范复查和需求复查均已完成，修复后无遗留代码问题。
修复包括上述真实模板测试、原文本配置/loader 回归入口，以及把可选参数更新
验证封装到 optimizer 公共接口。参数更新验证检查全部参与优化的本地 FP32 参数，
跨 rank 取最大变化；校验片段仅用于恢复连续性证据。

## 测试

- 最终 vLLM 引擎及 Loop 捕获合同组合回归：82 passed。
- optimizer 回归（包括校验片段之外发生参数更新）：6 passed。
- 真实 checkpoint 的完整逐消息监督回归：1 passed；全量测试中该项亦通过。
- 批量分词回归：26 passed。
- Ruff 检查、格式检查及提交时的相关 hooks 通过。

完整仓库测试运行结果为 1077 passed、25 skipped、75 failed、125 subtests passed。
其中新增的 1 项失败来自直接 prefill 捕获对象没有 `outputs` 字段；已修复为仅对
该诊断字段返回空列表，并通过上述 82 项最终回归。真实 vLLM 请求仍返回生成 token，
验收脚本仍严格要求 1 token。

其余 74 个失败用例与改动前 `f0cce315cf8d2aa9f72dc0173d1c6ce3383a8da6`
逐项相同；基线聚焦运行结果为 74 failed、78 passed、1019 deselected。
其中 58 项由 `tests/test_placement_group.py` 在收集阶段全局替换 Ray 模块引起，
另外 16 项是当前环境未安装 SGLang 的可选 backend 用例。SGLang 未安装或启动，
不在本轮 vLLM 验收范围内。相关测试文件未改动。最终未再次运行完整仓库 suite，
因此不把完整 suite 记作通过。

全量日志：`/tmp/torchspec_multimodal_full_tests.log`；基线对照日志：
`/tmp/torchspec_multimodal_baseline_failures.log`；最终兼容回归日志：
`/tmp/torchspec_prefill_compatibility_regression.log`。
提交时跳过无关的 actionlint hook：首次初始化其远程环境失败，且本次没有修改
GitHub Actions 工作流；其余适用 hooks 正常执行。

## 复现

命令及配置见 [单机图文示例](../examples/qwen3.8-27b-image-text-1node/README.md)。
第 11 步恢复运行额外设置 `logging.report_to=tensorboard`，仅写本地指标，
便于核对各损失项和梯度范数。

本轮只验收三条原样本及其重复训练记录的链路；未验收全部 327 条会话、
HTTPS/GIF 图片、完整训练 epoch 或草稿质量。
