# Qwen3.5 Loop/mHC 目标权重与 DSpark 草稿配置核验

后续变更说明：用户已要求将草稿 Q heads / KV heads / head_dim 从 32/8/128 改为 16/4/256，与目标 full attention 一致。以下核验正文保留变更前的配置、hash 和验收快照，其中“当前 JSON”指原核验时的版本；原 GPU 验收与旧草稿 checkpoint 不适用于新结构。重新读取配置时，`max_position_embeddings` 已为 262144，本次仅修改上述三个 attention 字段。

变更后验证：三个字段与实际目标 `text_config` 逐项一致；完整尺寸五层草稿在 meta device 上构造通过，每层 Q/K norm 均为 `[256]`，可训练参数为 924,113,665。另以真实 4096 隐藏宽度、16/4/256 attention 在 CPU 上完成短序列 SDPA 前向和反向，输出与全部参数梯度均有限；该检查仅缩短 rotary cache，不缩减注意力维度。未重跑 GPU FlexAttention 或端到端训练。

核验日期：2026-10-10。结论：**当前 JSON 作为该目标 checkpoint 的配套 DSpark 训练配置，结构与采集接口兼容，未发现需要修改的硬性不一致。** 这不是目标权重本身的模型配置，也不证明草稿效果或推测解码加速已经达到可用水平。

核验对象：

- 草稿配置：[qwen3_5_loop_mhc_dspark.json](../configs/draft_models/qwen3_5_loop_mhc_dspark.json)。
- 目标权重：[ckpt-225-bf16](/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16)。
- 配套训练入口：[config.yaml](../examples/qwen3.5-loop-mhc-dspark/config.yaml) 与当前仓库的 Loop/mHC、采集、DSpark 训练实现。

本次检查文件、权重头部、模型构造及已有运行记录，重跑 18 项轻量测试；未启动新的 GPU 训练，未修改草稿 JSON、训练 YAML 或目标权重。领域术语沿用 [CONTEXT.md](../CONTEXT.md)；本次未作新的架构决策，无需新增 ADR。

## 1. 配置与权重的关系

目标 checkpoint 声明 `Qwen3_5LoopForConditionalGeneration`；草稿配置声明 `Qwen3DSparkModel`，经本地解析得到 `DSparkConfig` 和 `DSparkDraftModel`。草稿主干、Markov head 和 confidence head 从头训练；目标 embedding 与 LM head 从指定目标 checkpoint 加载并冻结。因此，应检查二者的输入输出、词表和采集语义，不要求 decoder 权重逐项同构。

依据：[草稿类型解析](../torchspec/models/draft/auto.py)、[草稿构造](../torchspec/models/draft/dflash.py)、[训练初始化](../torchspec/training/dflash_trainer.py)、[目标 LM head](../torchspec/models/target/target_utils.py)。

| 检查项 | 草稿配置 | 目标权重 / 实现 | 判断 |
| --- | --- | --- | --- |
| 模型类型 | DSpark，5 个草稿层 | Qwen3.5 Loop/mHC，32 个物理层 | 两个模型职责不同，允许不同结构 |
| 隐藏维度 | `hidden_size=target_hidden_size=4096` | embedding、LM head 每行 4096 维 | 一致 |
| 词表 | 248320 | embedding、LM head 均为 `[248320,4096]` | 一致 |
| 目标执行深度 | `target_num_hidden_layers=80` | `32+(28-4)×(3-1)=80` | 一致；不是 80 份独立层权重 |
| 条件特征 | 5 个采集点，每个 4096 维 | 五个自然二维边界输出 | 一致，拼接维度 20480 |
| 草稿注意力 | Q/KV heads 为 32/8，head_dim 128 | 目标 full attention 为 16/4，head_dim 256，带 output gate；另有 linear attention | 草稿从头训练，允许不同；32×128=4096、32/8=4，自身维度成立 |
| 草稿 MLP | intermediate_size 8192 | 目标 12288 | 独立草稿容量选择，允许不同 |
| RMSNorm epsilon | `1e-6` | `1e-6` | 一致；归一化位置另外核对，见下文 |
| RoPE | theta `1e7`，草稿自身全维 rotary | theta `1e7`，目标 partial rotary factor 0.25 / mRoPE | theta 一致，位置编码结构不完全相同；独立草稿不要求照搬目标 attention |
| 上下文 | 131072 | 目标 text_config 为 262144 | 当前方案限定并验收 128K；未覆盖目标的全部 256K 范围 |
| mask token | 248044 | tokenizer 中为特殊 token `<\|endoftext\|>`，也是 pad token | ID 有效；DSpark 用其 embedding 填充待预测槽，不要求目标有独立 MASK token |
| Markov / confidence | rank 256，vanilla，两项 confidence 开关为 true | 草稿独有模块；confidence 输入 4096+256=4352 | 实现支持、维度一致，无需目标存在对应权重 |
| embedding 绑定 | `tie_word_embeddings=false` | embedding 与 LM head 独立存储 | 一致 |

配套 YAML 使用正确的 `model.language_model.embed_tokens.weight`、`lm_head.weight` 和 `model.language_model.norm.weight` key。训练器明确转为 BF16；目标 mHC router 保留 FP32，不由草稿 JSON 决定。

## 2. 五个采集点核验

执行顺序为 `[0..3] + [4..27] × 3 + [28..31]`，层号从零开始。实现依据：[Loop 展开、mHC 收敛与采集](../vllm/vllm/model_executor/models/qwen3_5_loop.py)。

| JSON 执行层号 | 物理层号 | 语义 | vLLM 辅助采集槽 |
| ---: | ---: | --- | ---: |
| 3 | 3 | 循环前最后一层输出 | 4 |
| 27 | 27 | 第一轮循环完成并收敛四流后的输出 | 28 |
| 51 | 27 | 第二轮循环完成并收敛四流后的输出 | 52 |
| 75 | 27 | 第三轮循环完成并收敛四流后的输出 | 76 |
| 78 | 30 | 尾段倒数第二层输出 | 79 |

每次进入物理层 4 展开为四条流，每次离开物理层 27 对流维取 mean，回到 4096 维。因此，`target_hidden_size=4096` 正确，无需因 `mhc.streams=4` 改为 16384。五份表示通过草稿 `context_proj.weight=[4096,20480]` 投影。

最终执行层 79（物理层 31）之后的真实目标 norm 输出单独传输，逻辑槽为 80，不混入五个辅助点。YAML 的 `store_last_hidden_states=true`、`last_hidden_states_prenorm=false` 正确：训练端直接用该最终状态乘目标 LM head，不重复应用目标 norm。

代码链路：[后端层号加一与执行层数](../torchspec/inference/engine/vllm_engine.py)、[V2 runner](../vllm/vllm/v1/worker/gpu/model_runner.py)、[connector](../torchspec/inference/engine/mooncake_hidden_states_connector.py)、[训练端 norm 条件](../torchspec/training/dflash_trainer.py)、[DSpark 目标分布](../torchspec/models/dspark.py)。

## 3. 权重文件与实际草稿构造

读取所有 safetensors 头部及 index，并按 checkpoint config 构造 text backbone、vision、mHC、embedding、LM head 与 loop gate 的预期 key/shape/dtype，得到：

- 11 个分片、857 个张量；index 与实际 shard 的 key 及归属完全对应，无 missing、extra、duplicate 或 shape/dtype mismatch。
- 每个张量的 shape、dtype 字节数与 offsets 一致，offsets 连续，分片末端与文件长度一致。
- BF16：761 个张量，9,409,817,840 参数；FP32：96 个张量，18,875,520 参数，均为 mHC router weight/bias。
- 总参数 9,428,693,360；tensor 数据 18,895,137,760 bytes，与 index 和 `export_complete.json` 一致。
- 32 个物理 text 层中有 24 个 linear attention、8 个 full attention。mHC 正好覆盖物理层 4–27；router weight 为 `[24,16384]`，bias 为 `[24]`，其中 `24=4²+2×4`、`16384=4×4096`。
- `use_gate=false` 时仍保存 `loop_gate.logits` 是可接受的：当前 Loop loader 显式跳过该参数。配置中的 `mtp_num_hidden_layers=1` 不代表此目录有 MTP 权重；本路径未实例化 MTP，也不以 MTP 初始化 DSpark。

通过当前真实草稿解析器在 `meta` device 构建完整尺寸模型，形状断言通过，无需分配完整参数数据：

| 参数 | 实际形状 / 数量 |
| --- | --- |
| `context_proj.weight` | `[4096,20480]` |
| `embed_tokens.weight` | `[248320,4096]`，冻结 |
| 单层 Q / K / V / O | `[4096,4096]` / `[1024,4096]` / `[1024,4096]` / `[4096,4096]` |
| 单层 MLP gate / up / down | `[8192,4096]` / `[8192,4096]` / `[4096,8192]` |
| Markov 两个矩阵 | 各 `[248320,256]` |
| confidence projection | `[1,4352]`，另有 `[1]` bias |
| 可训练参数 | 924,112,385 |
| 草稿内冻结 embedding 参数 | 1,017,118,720；外部目标 LM head 不计入此项 |

本轮运行 `tests/test_loop_capture.py`、`tests/test_dspark.py::TestDSparkConfig`、`tests/test_dspark.py::TestDSparkExport`：**18 passed**。仅有在同一 Python 进程内调用 pytest 导致的 anyio assertion-rewrite warning。该结果验证构造、接口、采集边界、恢复契约及导出，不替代真实目标的数值对照。

## 4. 已有运行证据的适用范围

[验收汇总](../examples/qwen3.5-loop-mhc-dspark/verification.json) 的 3 个输入文件 hash 与 14 个代码 hash 均与当前文件匹配。[目标完整性记录](../outputs/loop_dspark_verification_inputs/target_integrity.json) 指向本次确切目标路径：20 个文件的 size/mtime 均匹配，小于 30MB 的文件 hash 本轮复核也匹配。本轮未重新哈希约 18.9GB 大权重 payload，也未扫描全部目标张量数值。

已有 GPU 运行记录支持：

- [20 条样本、五次更新](../outputs/loop_dspark_acceptance_fixed_20261010/run_summary.json)，以及 [参数变化、冻结 embedding 与 optimizer 检查](../outputs/loop_dspark_acceptance_fixed_20261010/parameter_verification.json)。
- [第 4→5 步恢复并继续更新](../outputs/loop_dspark_resume_fixed_20261010/run_summary.json)。
- [四条各 131071-token 输入的满长更新](../outputs/loop_dspark_128k_fixed_20261010/run_summary.json)：每条 16 块、512 个监督 token，完成一次 optimizer 更新，无 OOM。这是长度压力验收，不是 128K 任务质量评测。

当前方案为内部采集请求预留一个生成 token，所以总长 131072 对应训练输入最多 131071。20 条真实样本及合成长序列测试证明训练链路可运行；没有与原训练实现版本绑定的真实 logits 数值对照。已有收尾异常及清理结果仍以原验收记录为准。

本轮记录的 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| draft JSON | `0a94f95739e9031d78cb2ee2e3d93b661619bf2a19714f26052f79c24351b9df` |
| 训练 YAML | `d730fb521d82137bb7f883636c42a636ae3e850f717790fea103a06144b34760` |
| target `config.json` | `49f91c9f1b4959e536b19b94b81c6848699bc2d9f0fc3d28ea8c706fda75139d` |
| target `loop_config.json` | `6121efa8d32380560e75c72f91495ae1783e97077ab4c0b65cf80166013ec16c` |
| target index | `a636c2d8b91d9123d915c2c7cfb28297e9bceb2ab82daee6bbfb640afb58cdb2` |

## 5. 合理性判断与使用边界

建议保留当前 JSON，继续将其作为 **128K DSpark 配套训练基线**。草稿层数、注意力结构和 MLP 宽度是独立容量选择；为了表面上匹配目标而修改为 32 层、16/4 heads 或 12288 MLP 没有本次核验依据。Markov rank、五层容量及五个采集点的效果是否最优，仍需固定数据与预算的评测或消融实验。

当前有三个明确边界：

1. **草稿效果尚未验证。** 从头训练约 9.24 亿可训练参数，已有五次更新只是功能验收。第 5 步训练 `avg_acc≈2.73%`、CE≈17.52、L1≈1.915，不能据此称为已训练好的加速草稿；这些也不是部署接受率。
2. **当前 Loop vLLM 路径拒绝 speculative decoding。** [配置校验](../vllm/vllm/model_executor/models/config.py) 显式拒绝非空 `speculative_config`。此配置可以用于目标采集训练，直接上线 Loop+DSpark 推测解码还需要另外接入和验收。
3. **恢复时的目标身份检查范围有限。** [training_contract](../torchspec/training/checkpoint.py) 绑定目标路径和 `config.json` hash，未绑定权重及 tokenizer hash。同目录替换权重、保留配置的情况不会被这项检查识别；本次身份关联另有验收完整性记录支撑。该限制不构成本次当前配置与当前权重的不一致。
