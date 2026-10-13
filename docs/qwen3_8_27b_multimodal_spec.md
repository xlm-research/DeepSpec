# Sensenova Flash Lite v42 图文推理与单机 DSpark 训练

## 已明确的目标

- 目标模型：`/mnt/afs_agents/hongjiawei/delivery/sensenova-flash-lite-v42-20260830-dpo-mtp`。
- 第一阶段：从 `mydata/image-text-327.jsonl` 选择少量图文样本，使用 vLLM 完成 prefill，再生成 1 个 token；不验收完整回答生成。
- 第二阶段：参考 `examples/qwen3.8-27b-1node`，在单机上跑通图文 DSpark 训练。
- 保留原有纯文本推理能力，在此基础上扩展图文训练。
- 只训练 DSpark 草稿模型；目标模型保持冻结。
- 保留完整已有会话，包括最后 assistant 回答，作为 prefill 输入；新生成的 1 个 token 不加入训练监督。
- 监督所有 assistant 轮次，保留并学习已有 `reasoning_content` 和 `tool_calls`；user/tool 消息保留为上下文，不计算回答 loss；图片 token 是视觉输入，不作为回答预测目标。
- 训练验收为 10 个真实 optimizer steps，验证 loss/梯度有限、草稿参数更新，保存 checkpoint，并恢复后再完成 1 步。
- 用户提供的数据目录为 `/mnt/afs_agents/wangshiyu/dspark-training-evaluation`。沿该目录的来源清单及原始数据的 `meta.root` 已定位图片根目录为 `/mnt/afs_toolcall/juxiaolong1/Data/draftData`；`media/...` 应相对此根解析。

## 环境核查（2026-10-13）

- checkpoint 架构为 `Qwen3_5ForConditionalGeneration`；本地 vLLM 已有对应多模态模型实现。
- 单机示例已指向该 checkpoint，但显式设置 `language_model_only: true`。
- 单机示例冻结目标模型，使用已有 assistant 回答提取 hidden states，仅优化 DSpark 草稿模型；默认四卡训练、四卡目标模型推理。
- 数据共 327 条，全部含 `image_url` 内容块；共 1452 次图片引用，涉及 761 个不同图片地址。其中 746 个不同地址为相对 `media/...` 路径，15 个为 HTTPS URL。
- 每条包含 1–48 次图片引用、12–1406 条消息；所有记录最后一条消息均为 assistant。图片也存在于 tool 消息中，不能只处理最后一条 user 消息。
- 全部 746 个不同本地图片路径已逐个打开并读取验证，可读率 746/746。323 条会话完全使用本地图片；另外 4 条涉及 HTTPS 图片，远程可访问性尚未验证。
- 首轮采用零起始索引 114、217、231。231 替换原候选 246，补充 tool 消息图片，且含已有 reasoning/tool_calls；图片可读，无 GIF 或 HTTPS 图片。真实 processor 展开长度分别为 11296、18792、77523，全部完整保留。
- TorchSpec 已有多模态数据传输路径。图片占位符展开会改变 token 数，训练需启用延迟分词或合适的 renderer，并验证展开后 hidden states 与监督 mask 对齐。
- 本机可见八张 H800；核查时显存占用为零。这不构成后续运行时资源可用性的保证。

## 方案复查（2026-10-13）

### 已证实的方案缺口

- 三条记录不足以启动参考四卡布局：一次 dispatch 需四条，每个 optimizer step 需十六条。当前 controller 在首个 dispatch 前耗尽三条数据时不能靠 epoch 重载恢复，因此“必要时重复采样”必须变成明确的训练数据准备步骤，不能仅设置 `num_train_steps=10`。
- 仅开启延迟分词不足以正确接入图片。现有 `_iter_conversation_samples` 会按 qwen 模板先把内容块摊平成 `<image>`；原生 checkpoint 模板需要结构化图片内容，才能输出 `<|vision_start|><|image_pad|><|vision_end|>`。
- 延迟分词的 `_format_single` 丢弃顶层 `tools` 和 `generation_config`。新增图文路径必须保留并传递这些信息，不能以改 JSONL 文件路径代替适配。
- 原候选样本只覆盖 user 图片，遗漏实际数据中的 tool 图片。
- 恢复参数是累计训练终点，恢复第十步后再跑一步应设 `num_train_steps=11`。默认十步 cosine schedule 会在十步后降到零；首次运行及恢复应固定同一 `lr_total_steps=11`，使恢复的一步仍有正学习率。
- 现有动态 mask 按 assistant 边界扫描，没有显式排除视觉位置。实际数据图片都在 user/tool 中，但接口合同仍要求对 processor 展开后的视觉位置显式置零；不能依赖图片恰好位于不监督的角色。

### 已核实可复用的机制

- Qwen3.5 图文 wrapper 已支持辅助层捕获，视觉融合 embeddings 进入语言模型，再采集各层 residual stream。目前没有证据需要新增模型架构。
- 五个训练层 `[3, 15, 31, 47, 59]` 对应捕获槽 `[4, 16, 32, 48, 60]`；独立最终槽为 64，保持 pre-norm 语义。Loop/mHC ADR 的 post-norm 合同不适用于此 checkpoint。
- DSpark 的 CE、L1、confidence 三项都使用同一监督权重；正确的角色与视觉 mask 应同时约束三项，不必构造三套不同 mask。
- 现有推理 adapter 关闭 prefix caching，重复记录仍须获得完整 prefill 特征。端到端 Mooncake 完整性仍需实际运行验证。

## 修订方案

1. 首轮保留三个不同的本地图文样本；建议采用 114、217、231，覆盖单图、多图及 tool 图片。使用原始 JSONL 派生小子集与路径映射，不改写原数据。先检查原生模板和 processor 实际展开长度；超限时报告并换样本，不静默截断。远程图片及全部 327 条会话的验收留作后续。
2. 补齐独立的图文原生格式化路径，复用现有数据/推理接口，保留结构化内容块、图片顺序、tool_calls、顶层 tools 与 generation_config，显式设置 `preserve_thinking=true`、`add_generation_prompt=false`。不得把图片简单摊平成 `<image>`。显式设置 `last_turn_loss_only=false`，不沿用 thinking 自动选择最后一轮的规则。格式化方式及视觉 mask 合同变化需更新缓存版本，避免读取旧缓存。
3. 第一阶段先完成三条样本的真实图文 prefill，每条 `max_tokens=1`，再验证同一路径的 hidden states 提取。启用 `language_model_only=false`，保留 `VLLM_USE_V2_MODEL_RUNNER=0` 使用现有 `extract_hidden_states` 路径；显式 image limit 按样本最大图片数设定，用于限制初始化 profiling（本地源码的默认图片上限为 999）。验证图片顺序和视觉执行、展开后的 token IDs 与 hidden states 长度一致、辅助层与最终层捕获完整且有限，新增 token 不进入训练数据。
4. 图片展开后构造全部 assistant 监督 mask，并把视觉位置显式置零。以原始消息及已渲染片段核对各轮 reasoning/tool_calls 有监督、user/tool 无监督。每个原样本有效监督 token 和 DSpark 可用训练块须大于零，不能用全零 loss 验收。新增针对真实原生模板、多轮工具调用、展开视觉位置的组合回归，并回归原文本路径。
5. 新增独立单机图文示例，沿用四卡 FSDP、四卡 TP=4、micro batch 1、梯度累积 4，即 global batch 16。用户确认将三个原样本派生为 160 条独立 `data_id` 的重复记录（记录源行号与重复序号），对应十个完整 optimizer steps。重复数据只用于链路验收，不作为质量评估。上下文上限沿用 131072，并为内部生成预留 1 token。
6. 验证只有草稿参数参与优化，loss/梯度有限、启用的损失项有有效贡献、草稿参数有非零更新；保存模型、optimizer（含 FP32 master 参数）、scheduler 与迭代元数据。固定 `lr_total_steps=11`，先运行至 `num_train_steps=10` 并保存第十步，再用 `LOAD_PATH`/`CKPT_STEP=10` 恢复，设置累计终点 11。核对恢复前后状态和学习率连续，并验证恢复的一步仍更新草稿；仅出现加载成功日志不算恢复验收。
7. 保留现有纯文本示例与默认调用方式，回归纯文本 prefill/单 token 推理及原数据/监督默认行为。检查公共接口对纯文本的影响；不要求逐 token 与改动前完全一致。本轮不包含完整训练 epoch、远程图片覆盖或 DSpark 草稿部署评估。

## 复查后已确认

- Q9：用少量原样本的 160 条独立 ID 重复记录完成训练验收。
- Q10：将原样本 246 替换为含 tool 图片的 231。

## 当前状态

用户已确认修订方案，Q9 采用 160 条独立 ID 的重复记录，Q10 使用 114、217、231。实现与运行已启动；采用注册的原生 renderer 直接构造未展开 token IDs，保留结构化内容和 tools，再由 vLLM processor 展开图片并重算 packed mask，无需启用原有延迟分词路径。真实 processor 预检长度分别为 11296、18792、77523，全部保留完整会话。
