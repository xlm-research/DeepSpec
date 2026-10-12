# Qwen3.5 Loop + mHC：DSpark 训练接入设计问答

状态：Q1–Q6 已确认，[规格说明](qwen3_5_loop_mhc_dspark_spec.md)中的实现与 AC-01 至 AC-08 验收已完成，见[验收汇总](../examples/qwen3.5-loop-mhc-dspark/verification.json)。记录日期：2026-10-10。

下文保留设计阶段的调查、候选方案与问答；当前交付结果以规格状态和验收汇总为准。

后续用户决定：草稿 Q heads、KV heads、head dimension 改为与目标 full attention 一致的 16/4/256；下文 32/8/128 为历史设计。当前结构及旧 checkpoint 的恢复限制见[训练规格](qwen3_5_loop_mhc_dspark_spec.md)。

用户需求：参考 `examples/qwen3.8-27b-1node` 的实现接入 DSpark，支持 `qwen3.5-loop-mhc` 的训练。本轮按 `grill-with-docs` 核实已有实现、收敛设计决策，并记录已确认术语。用户已确认冻结目标模型、只训练 DSpark 草稿，单机八卡，128K 长度，使用 `mydata/test_20.jsonl`；vLLM 接入属于训练链路，不要求本次交付草稿推测解码部署或加速效果验收。

关联：[领域词汇表](../CONTEXT.md)、[参考示例](../examples/qwen3.8-27b-1node/README.md)、[Loop/mHC 推理规格](qwen3_5_loop_mhc_vllm_spec.md)、[256K 推理验收记录](qwen3_5_loop_mhc_256k_plan.md)、[参数与状态归属 ADR](adr/0001-loop-physical-weights-execution-state.md)、[训练采集契约 ADR](adr/0002-loop-mhc-dspark-capture-contract.md)。

本文件保留源码调查、当时的候选和问答过程。实施及后续拆分任务以[训练规格](qwen3_5_loop_mhc_dspark_spec.md)中的合同和验收条件为准；历史段落里的“尚待选择”描述不再代表当前决策状态。

## 已核实的基线

- 参考示例采用已有回答监督训练，只优化 DSpark 草稿；目标模型及训练侧的目标 embedding、LM head、final norm 冻结。训练期间目标模型提取输入样本的隐状态；示例关闭在线生成监督与草稿权重同步。证据：[示例说明](../examples/qwen3.8-27b-1node/README.md:3)、[启动参数](../examples/qwen3.8-27b-1node/run.sh:80)。
- 参考资源配置为单机八卡，四卡 FSDP 草稿训练、四卡 TP=4 目标推理；长度上限 131072。其五个辅助采集层为 `[3, 15, 31, 47, 59]`。这些属于参考模型的配置，尚未确定作为 Loop/mHC 的新配置。证据：[训练配置](../examples/qwen3.8-27b-1node/config.yaml:20)。
- 参考示例强制使用 vLLM V1 runner 的 `extract_hidden_states` 路径。Loop/mHC 既有推理规格和运行记录使用 V2 runner、目标 TP=8；二者之间需要实际适配。证据：[runner 设置](../examples/qwen3.8-27b-1node/run.sh:97)、[Loop/mHC 配置](../vllm/examples/qwen3.5-loop-mhc/config.yaml)。
- Loop/mHC 当前目标有 32 个物理层、80 个执行层，物理层 4–27 执行三轮；每轮 mHC 残差使用四条流，轮次结束后收敛为宽度 4096 的表示。参考模型的层号和表示宽度不能直接成为此目标的采集契约。证据：[既有规格的执行计划与 mHC 边界](qwen3_5_loop_mhc_vllm_spec.md)。
- Loop/mHC 已有 8K 与 256K 目标推理运行记录；这些记录的验收范围为目标模型功能和结构，不包含 TorchSpec 特征采集与 DSpark 训练。训练上下文长度也尚未因此确定。证据：[256K 验收记录](qwen3_5_loop_mhc_256k_plan.md)。
- 本次只读 GPU 快照显示本机八张 NVIDIA H800，每卡总显存 81559 MiB、已用 4 MiB；这是检查时的资源状态，不代表某种新训练拓扑已经验证可行。

## 兼容性调查：源码事实与待验证项

1. **采集路径有启动冲突。** TorchSpec 的 prefill 分支仍通过 `speculative_config.method=extract_hidden_states` 请求采集；这里借用草稿接口缓存表示，不代表真正加载 DSpark 做推测解码。Loop 配置目前拒绝任何 `speculative_config`，同时要求 TP=8 和 V2；V2 的方法白名单又不包含 `extract_hidden_states`。证据：[prefill 采集配置](../torchspec/inference/engine/vllm_engine.py:258)、[Loop 支持范围校验](../vllm/vllm/model_executor/models/config.py:821)、[V2 方法校验](../vllm/vllm/config/vllm.py:2383)。因此修改目标路径与 JSON 维度不足以接通训练。
2. **V2 存在另一条可调查路线。** V2 runner 已能从 connector 配置读取 `target_hidden_state_layer_ids`，在没有 `speculative_config` 且 PP=1 时请求辅助表示。它是否完整满足本轮已有回答监督、分块采集和传输要求，还需检查端到端行为。另一种候选是为 Loop 补齐 V1 采集支持。尚未选择路线。证据：[V2 直接采集入口](../vllm/vllm/v1/worker/gpu/model_runner.py:234)。
3. **必须区分物理层和执行层。** TorchSpec 目前用 checkpoint 的 `num_hidden_layers` 过滤采集层、执行 `+1` 转换并追加 final pre-norm 槽位；此 checkpoint 值为 32，而模型运行时展开为 80。connector 的直接采集 manifest 也使用配置总层数。当前 Loop 前向只返回最终归一化结果，没有返回辅助采集表示。证据：[engine 层号转换](../torchspec/inference/engine/vllm_engine.py:195)、[connector manifest](../torchspec/inference/engine/mooncake_hidden_states_connector.py:225)、[运行时展开](../vllm/vllm/model_executor/models/qwen3_5_loop.py:174)、[Loop 前向](../vllm/vllm/model_executor/models/qwen3_5_loop.py:208)。
4. **mHC 内段采集需要定义表示。** 循环内部表示具有四条流，每轮最后才取均值恢复宽度 4096；直接采集内段不能当作普通二维 hidden state 传给草稿。可讨论轮次边界采集或明确的多流归约策略，但这些属于尚待用户决定的训练特征契约。证据：[mHC 展开和收敛](../vllm/vllm/model_executor/models/qwen3_5_loop.py:146)。
5. **最终归一化需要核实正确语义。** Loop 使用 Qwen3.5 的 Gemma 形式 RMSNorm，缩放为 `1 + weight`。训练侧若无法从目标架构构造 norm，会回退到仅乘 `weight` 的通用 RMSNorm。因此沿用 pre-norm 特征时必须保证正确的 norm，而不能把架构构造失败视作无害回退；是否触发回退目前尚未实测。证据：[Loop final norm](../vllm/vllm/model_executor/models/qwen3_5_loop.py:200)、[训练侧回退](../torchspec/models/target/target_utils.py:37)、[回退选择](../torchspec/models/target/target_utils.py:192)。
6. **TP=8 限制不等于 TP=4 数学上不可行。** 现有保护对应先前只验收 TP=8 的范围。已检查的 attention/GDN 头数及 MLP 宽度满足 TP=4 的静态整除条件，但尚未运行 TP=4，也没有草稿训练并存时的显存证据。Q2 已选择单机，后续应据此研究 GPU 划分，不能仅删除校验便声称支持。

## 第二轮期间的静态调查（历史）

以下均为候选方案或源码推导，尚未作为用户选择，也未进行 GPU 构建、训练或容量测试。

### 已有回答监督的 V2 路线候选

若 Q3 选择已有回答，可保留 `train_with_decode=false`、原样本 loss mask 和内部 `max_tokens=1`，单独选择 V2 直接采集：不构造 `speculative_config`，经 connector 配置指定辅助执行层。不能以切换到在线 decode 分支替代；该分支会丢弃原 mask，并且只生成一个 token 的样本可能被过滤。依据：[已有回答路径](../torchspec/inference/engine/vllm_engine.py:645)、[在线生成路径](../torchspec/inference/engine/vllm_engine.py:773)。

推荐供后续选择的最终状态契约是：Loop 的辅助输出只包含五个草稿条件特征，最终参数单独传实际 post-norm 表示，训练配置显式设置 `last_hidden_states_prenorm=false`。这样直接使用目标模型真实 LM-head 输入，避免重建 Gemma norm 的回退风险；不改变五个辅助点的归一化语义，也不改变草稿自己的 final norm 和 Markov head。L1 比较草稿与目标分布，confidence 的监督来自该分布差异，二者没有额外要求必须在传输端保存 pre-norm。另一种可行选择是显式采集 pre-norm 并正确构造 Gemma norm；关键是最终状态只能归一化一次。依据：[V2 传最终输出](../vllm/vllm/v1/worker/gpu/model_runner.py:1858)、[训练侧 norm 应用](../torchspec/training/dflash_trainer.py:292)、[L1/confidence](../torchspec/models/dspark.py:184)。

直接采集在请求结束时发布 `all_token_ids[:-1]`；输入 L、恰好生成一个内部 token 时应得到 L 行。后续验收须核对输入 IDs、原 loss mask 和六份表示的长度及位置对齐。该最后状态的归一化语义应随缓存或训练产物明确记录，避免后续误用。依据：[完成请求发布](../torchspec/inference/engine/mooncake_hidden_states_connector.py:908)。

### 草稿条件特征的候选采集点

| 零基执行层输出 | 语义位置 | 自然输出宽度 | 对应采集槽 |
| --- | --- | --- | --- |
| 3 | 物理层 3，循环前 | 4096 | 4 |
| 27 | 第一轮物理层 27，mHC 收敛后 | 4096 | 28 |
| 51 | 第二轮物理层 27，mHC 收敛后 | 4096 | 52 |
| 75 | 第三轮物理层 27，mHC 收敛后 | 4096 | 76 |
| 78 | 物理层 30，倒数第二执行层 | 4096 | 79 |

这些点可组成五个辅助条件特征，另以槽位 80 表示最终目标状态。依据：[执行展开](../vllm/vllm/model_executor/models/qwen3_5_loop.py:164)、[mHC 轮次结束的收敛](../vllm/vllm/model_executor/models/qwen3_5_loop.py:146)。是否选择这组边界，需在本轮范围回答后确认。

如果把执行层 79 同时作为第五个辅助点，其槽位也是 80；现有 engine 会去重，connector 又把它标为最终目标状态，五个辅助点实际只剩四个。直接重复槽位还会违反严格升序校验并生成相同传输 key。因此独立 final 槽、辅助层数量及角色一致属于接口正确性要求，不能靠配置重复层号解决。依据：[去重](../torchspec/inference/engine/vllm_engine.py:217)、[布局校验](../torchspec/inference/engine/mooncake_hidden_states_connector.py:264)、[角色分配](../torchspec/inference/engine/mooncake_hidden_states_connector.py:887)。

### 草稿尺寸和初始化候选

沿用参考的五层 DSpark、MLP 宽度 8192、Markov rank 256；将草稿和目标特征宽度改为 4096，使用 32 个 query heads、8 个 KV heads、head dimension 128，词表 248320。`32 × 128 = 4096`、`32 / 8 = 4`，现有 attention、context projection、embedding 和 LM head 接口的维度静态自洽。草稿是独立 backbone，不要求复制目标的 GDN 或 mHC 架构。依据：[attention 构造](../torchspec/models/draft/dflash.py:173)、[特征投影和 embedding](../torchspec/models/draft/dflash.py:372)、[DSpark heads](../torchspec/models/draft/dspark.py:145)。

现有 Loop 示例指向 `ckpt-225-bf16`；该 checkpoint 索引和相关配置中未找到配套 DSpark 初始权重。局部环境存在的 MTP 草稿具有不同架构，不能作为上述五层 DSpark 直接加载。当前可推荐从头初始化草稿，沿用目标 embedding 和 LM head；这不代表已确认初始化方案，也不代表全盘不存在其他可用权重。

### 长样本的数据量

仅按五个条件特征加一个最终目标状态、每个宽度 4096、BF16 两字节估算：

| 训练序列长度 | 六份隐状态的有效数据 |
| --- | --- |
| 131072 | 6 GiB / 样本 |
| 262144 | 12 GiB / 样本 |

以上不含 token、mask、复制、拼接及传输 staging。现有 V2 direct capture 在多个 TP rank 收集 CPU 行，发布时再限制由 TP rank 0 写入，因而不能只按单份有效数据预算主存。参考的并发、预取和缓冲参数必须随所选长度重新核查。此估算不等于 GPU 训练峰值或实际吞吐。

资源路线可选目标 TP=4 与四卡 FSDP 同时工作，或单机 TP=8 分阶段采集落盘、释放目标后训练；前者需要新的 TP=4 验证，后者需要按样本规模核算磁盘成本。参考的同时采集训练路线最接近原需求，但尚未选定。若采用该路线，候选初始设置为单个推理请求、microbatch=1、较低预取；sample pool 至少容纳一次训练派发所需的 `训练 DP 数 × microbatch`，以免在积满一批之前停止生产。依据：[训练派发门槛](../torchspec/controller/training_controller.py:458)、[离线采集入口](../torchspec/offline/generate.py:61)。

现有 CI smoke 的日志检查要求实际 optimizer step 和有限正 loss，可复用其验收思路；它本身没有覆盖本轮 Loop 采集语义、最长样本和 checkpoint 恢复。依据：[已有 smoke 检查](../tools/ci/run_2gpu_training_smoke.sh:183)。

## 指定数据的只读核查

使用现有 Loop checkpoint 的本地 tokenizer 和本仓库 `qwen` parser，在 CPU 上按 prefill 输入上限 131071 分词；没有运行目标模型或训练，没有修改原数据。

| 检查项 | 结果 |
| --- | --- |
| 文件 | `mydata/test_20.jsonl`，2,348,954 bytes |
| JSON 与 schema | 20 条有效记录，无坏行；顶层 `conversations`，消息字段 `role` / `content` |
| 消息 | 20 个 system、175 个 user、175 个非空 assistant；各条以 assistant 结束 |
| 实际 token 长度 | 最短 7177，最长 105647，中位数 29477.5，总和 729182 |
| 截断及过滤 | 20 条均未截断，按 `min_loss_tokens=14` 全部保留 |
| 默认监督范围 | 20 条均触发 thinking 检测，`last_turn_loss_only=auto` 实际只监督最后一个 assistant 轮次 |
| 默认监督 token 数 | 每条 196–676，总和 6853 |

格式依据：[qwen 对话解析](../torchspec/data/parse.py:174)。监督策略依据：[默认值](../torchspec/config/train_config.py:42)、[auto 判定](../torchspec/data/dataset.py:202)。`min_loss_tokens=14` 对应两个七 token 草稿块的最低要求，当前样本均满足。

“已有回答监督”本身没有指定是否所有 assistant 轮次都计 loss。候选方案明确沿用本文件在参考默认下的最后轮次监督，并将配置写成 `last_turn_loss_only=true`，避免数据内容检测带来隐式变化；如果用户选择所有 assistant 轮次，应显式改成 false。两者都使用原有回答，早先轮次在最后轮次方案中仍作为上下文。

最长真实样本 105647 tokens，不能代替满 128K 验收。另一个实际差异是：参考全局 batch 为 16，当前按向下取整计算每轮步数，只产生一次更新、使用 16 条；候选方案将梯度累积设为 1，全局 batch 为 4，形成五次更新。依据：[步数计算](../torchspec/controller/setup.py:90)。

## 已确认范围

- **训练对象**：冻结 `qwen3.5-loop-mhc`，只训练配套 DSpark 草稿，不更新目标模型本体参数。
- **部署边界**：必须在单台八卡机器上完成目标模型推理和草稿训练；选定四卡 TP=4 目标推理加四卡 FSDP 训练。拓扑已确认，实际容量及训练仍待实施阶段验证。
- **监督数据**：使用用户指定的 `/mnt/afs_share/lezewei/TorchSpec/mydata/test_20.jsonl`，按参考流程使用已有回答监督；20 条全部通过格式、分词及最少监督长度核查。
- **长度**：支持 128K，即 131072 tokens 的模型总长度上限；具体训练有效行数和内部一个生成 token 的边界在完整方案中说明。
- **交付范围**：接通 vLLM 目标模型采集和 DSpark 训练，完成实际训练并产出 checkpoint。不要求本次部署训练后草稿做 DSpark 推测解码，也不设接受率、加速比指标。
- **技术方案**：Q6 已确认最后轮次监督、V2 直接采集、五个边界执行层特征、独立 post-norm 最终状态、五层 4096 宽度草稿从头训练，以及一轮五次更新、保存恢复和独立 128K 验收。

## 交付契约

完整方案已转为[规格说明](qwen3_5_loop_mhc_dspark_spec.md)，涵盖输入、运行与采集合同、草稿结构、训练规模、checkpoint 及八项验收。后续变更以规格为准，本记录保留决策来源和历史调查。

## 设计树

```text
参考单机示例接入 Loop/mHC 与 DSpark
├─ D1 [Q1 已确认] 冻结目标，只训练 DSpark 草稿
│  ├─ D3 [Q3 已确认] 使用 mydata/test_20.jsonl 的已有回答
│  │  ├─ [已核查] 20 条全部可用，默认只监督最后轮次，最长 105647
│  │  └─ [Q6 已确认] V2 直接采集、保留原 loss mask，显式最后轮次监督
│  ├─ D4 [Q4 已确认] 128K 模型总长度上限
│  └─ [Q6 已确认] 边界五层特征、post-norm 最终状态、五层草稿从头训练
├─ D2 [Q2 已确认] 必须单机八卡完成
│  └─ [Q6 已确认] TP=4 目标 + 四卡 FSDP；容量待实现阶段验证
├─ D5 [Q5 已确认] vLLM 接入训练，交付实际训练与 checkpoint
│  └─ [Q6 已确认] 一轮五步、保存恢复及独立 128K 长度验收
└─ Q6 [已确认] 用户请求将“刚才确认的方案”整理为规格，设计问答完成
```

源码兼容性由调查和后续实现验证，不要求用户代查。第二轮发问时采集和草稿形状仍在做只读核查；随后结合 Q3–Q5 回答完成候选核查，在 Q6 汇总并确认了完整方案。

## 第一轮：已回答

**Q1：训练对象。** 推荐冻结 `qwen3.5-loop-mhc` 作为目标模型，只优化 DSpark 草稿。

**Q2：单机约束。** 推荐将单机完成目标模型推理与草稿训练作为必须满足的交付条件；GPU 划分仍需兼容性和容量证据。

用户回答：“可以”。按本轮两个推荐项记录为确认：只训练草稿，单机八卡。此回答不包含尚未提出的采集策略、训练长度或完整方案确认。

## 第二轮：已回答

**Q3：监督数据。** 推荐沿用已有回答监督训练及参考数据集 `mydata/sensenova-flash-lite-v42-text-1000000.jsonl`。可选更换已有回答数据集，或让 Loop 目标模型在线生成回答；若选择后者，下一轮再确定生成、过滤及草稿参与方式。

**Q4：训练长度。** 推荐先支持参考示例的 128K，即 131072 tokens。另可选择 256K（262144）或首期 8K；这些数值指训练序列的总长度上限。预处理及目标引擎的内部生成 token 预算应在确定采集路径后明确，不能由既有 256K 目标推理验收代替训练长度决定。

**Q5：交付范围。** 推荐本次完成 DSpark 训练接入、实际训练验收以及 checkpoint 保存和恢复。另一选择是同时交付训练产物在 vLLM 中的 DSpark 推测解码，并验收接受率或加速效果；选定范围后再制定具体通过条件。

用户回答：“Q3:/mnt/afs_share/lezewei/TorchSpec/mydata/test_20.jsonl,Q4:128k,Q5:要接入 vllm 完成训练即可”。据此替换原推荐的数据集，确认 128K 和 vLLM 参与的训练范围；不再展开在线生成回答或草稿推测解码部署的分支。

## 第三轮：完整方案已确认

**Q6：是否按上述完整方案进入实现与单机验收？** 推荐选择最后轮次监督、TP=4 加四卡 FSDP、V2 直接采集、自然边界五层条件特征和真实 post-norm 最终状态，五层 DSpark 从头训练；用指定 20 条完成五次更新、保存恢复，并独立验证 128K 全训练链路。如果希望改变监督轮次或任一技术取舍，可直接指出，其余已确认的 Q1–Q5 范围不重新询问。

用户随后请求：“$to-spec 把刚才确认的方案整理成规格说明。”据此将前述完整方案记为确认，并完成规格整理。测试边界沿用该方案的既有训练入口及端到端采集、更新和恢复合同，不新增需要另行选择的测试接口。

## 记录边界

设计阶段的只读检查和文档记录未改动训练实现或启动训练；当时已创建训练规格及采集契约 ADR。2026-10-10 的实施阶段已完成 V2 直接采集、五步训练、完整恢复、满 128K 更新和 TP=8 回归，详见[验收汇总](../examples/qwen3.5-loop-mhc-dspark/verification.json)及[启动说明](../examples/qwen3.5-loop-mhc-dspark/README.md)。工单系统和 triage 标签配置仍缺失，规格未发布到 tracker；本地实现与运行验收已经完成。
