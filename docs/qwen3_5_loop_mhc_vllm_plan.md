# Qwen3.5 Loop + mHC 接入 vLLM：实施方案与设计问答

状态：8K 与 256K 扩展均已完成功能与结构验收；256K 结果见[运行报告](../vllm/examples/qwen3.5-loop-mhc/verification_256k.json)。更新日期：2026-10-10。

用户确认首阶段为单机 8 卡、TP=8、PP=1、8K（8192 tokens）目标模型推理，不需要做对照。保留权重、结构和运行检查，取消外部实现及整段重算的数值对照、误差阈值和参考模型输出采集；TorchSpec 训练与 MTP 不进入首阶段。

本文件保留 8K 设计阶段的源码分析和问答，下面的拟实施描述及当时的证据边界属于历史记录。当前已有模型实现及[八卡运行报告](../vllm/examples/qwen3.5-loop-mhc/verification.json)，包括 8176 输入加 16 输出的生成；用户也已确认 8K 通过。运行检查不证明数值对齐。新需求另见 [256K 扩展设计问答](qwen3_5_loop_mhc_256k_plan.md)。

关联记录：[术语表](../CONTEXT.md)、[参数与状态归属 ADR](adr/0001-loop-physical-weights-execution-state.md)。

正式交付范围和验收契约已整理为[规格说明](qwen3_5_loop_mhc_vllm_spec.md)；本文件继续保留源码证据与设计问答过程。

## 1. 输入与证据边界

- 权重目录：`/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16`。
- 环境激活：`source /mnt/afs_share/lezewei/TorchSpec/h800conda.sh`；加载的是本仓库 `vllm/` 源码，环境设置了 `VLLM_USE_V2_MODEL_RUNNER=1`。
- checkpoint 的 architecture 为 `Qwen3_5LoopForConditionalGeneration`，model_type 为 `qwen3_5`。此前实际配置构建检查显示，本地 vLLM 尚未注册此架构；Transformers 回退也没有可用的模型类或 checkpoint 自带 `auto_map`。
- 可读推理参考：`/mnt/afs_agents/hongjiawei/code/loop_transformers/infer/LightLLM_mhc`。用户提供的 overlay 权重指向原 checkpoint，配置另用了 `qwen3_5_loop`；这不要求 vLLM 同样更改原配置的 model_type。
- 可读训练参考：`/mnt/afs_agents/hongjiawei/code/loop_transformers/loop_mhc/mhc_experiment/model.py` 及相邻 Loop 训练代码。它们支持当前数学语义，但尚未证明与导出权重时的训练 checkout 完全同版。
- 用户提供的 LightLLM 部署是有效 TP=1、DP=4，并启用了独立 MTP draft。本次借用其源码理解模型结构；按照用户决定，不将该服务作为数值验收基准。

核实位置：checkpoint [config.json](/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16/config.json)、[权重索引](/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16/model.safetensors.index.json)、[部署脚本](/mnt/afs_agents/yankuihao/works/scripts/deploy_mhc_mtp_1010/deploy_mhc_mtp_tpdp.sh)。

## 2. 已核实的模型语义

### 2.1 参数层与执行层

checkpoint 有 32 个物理层，循环区间为 `[4, 28)`，`eval_count=3`，`use_gate=false`。执行计划为：

```text
physical_indices = [0..3] + [4..27] * 3 + [28..31]
```

| 执行层索引 | 物理层索引 | 阶段 |
| --- | --- | --- |
| 0–3 | 0–3 | 前段 |
| 4–27 | 4–27 | 循环第 1 轮 |
| 28–51 | 4–27 | 循环第 2 轮 |
| 52–75 | 4–27 | 循环第 3 轮 |
| 76–79 | 28–31 | 后段 |

物理层按三个 GDN、一个 full attention 重复排列。展开后需要 **60 个 GDN 执行状态槽和 20 个 full-attention KV 状态槽**；同一物理层跨轮次共享参数，状态互相独立。数字来自配置与源码推导，尚不是新 vLLM 的运行测量。

参考：[执行计划](/mnt/afs_agents/hongjiawei/code/loop_transformers/infer/LightLLM_mhc/lightllm/models/qwen3_5_loop/config.py:12)、[参数共享及运行配置](/mnt/afs_agents/hongjiawei/code/loop_transformers/infer/LightLLM_mhc/lightllm/models/qwen3_5_loop/model.py:22)、[状态槽映射](/mnt/afs_agents/hongjiawei/code/loop_transformers/infer/LightLLM_mhc/lightllm/common/state_cache_manager/linear_att.py:163)。该目录的 Loop README 仍描述共享缓存及最终轮覆盖，与当前源码不符，本方案以源码为依据。

### 2.2 每轮 mHC 的输入输出边界

每次进入物理层 4，把宽度 `D=4096` 的表示复制为四条流 `[tokens, 4, D]`；每次离开物理层 27，对四条流取均值恢复宽度 D。三轮均各自 expand/collapse，下一轮重新展开。

每个 mHC 层依次执行：

```text
R -> attention router -> 聚合 -> norm -> Attention/GDN -> 回写到 R
R -> FFN router       -> 聚合 -> norm -> MLP           -> 回写到 R
```

Attention/GDN 和 MLP 的输入宽度仍是 D，只有残差表示带四条流。prefill 与 decode 使用相同路由语义。

参考：[layer_infer.py](/mnt/afs_agents/hongjiawei/code/loop_transformers/infer/LightLLM_mhc/lightllm/models/qwen3_5_loop_mhc/layer_infer.py:8)。

### 2.3 路由公式与精度

令 `R` 为四条残差流，`z` 为路由器输出：

```text
u = flatten(R).float()
u = u * rsqrt(mean(u ** 2) + 1e-6)
z = linear(u, router_weight.float(), router_bias.float())
pre  = sigmoid(z[..., :4])
post = 2 * sigmoid(z[..., 4:8])
log_mix = reshape(z[..., 8:], [..., 4, 4])
重复 20 次：先沿最后一维、再沿倒数第二维做 logsumexp 归一化
mix = exp(log_mix)
branch_input = pre.to(R.dtype) @ R
branch_output = Attention/GDN/MLP(norm(branch_input))
R_next = mix.to(R.dtype) @ R + post.to(branch_output.dtype) * branch_output
```

矩阵方向是 **`mix @ R`**。公式省略了广播维度，实际 cast、广播和加法顺序按参考实现保留。checkpoint 中有 `mhc.pre_scale=0.5`，但已读路由代码没有消费该字段；本方案不会额外乘 0.5，若原训练版本另有语义则须重新核对。

router 的 weight/bias 是 FP32，每个路由器形状分别为 `[24, 16384]` 和 `[24]`，每个 mHC 物理层各有 attention、FFN 两套路由器。须显式按 FP32 分配，避免全局 BF16 模型初始化把参数降精度。

Qwen3.5 的可学习 RMSNorm 使用 Gemma 形式的 `(1 + weight)`；router 对展开向量的 RMS 归一化则不带可学习权重。两者不可混用。

参考：[routing.py](/mnt/afs_agents/hongjiawei/code/loop_transformers/infer/LightLLM_mhc/lightllm/models/qwen3_5_loop_mhc/routing.py:21)、[训练参考](/mnt/afs_agents/hongjiawei/code/loop_transformers/loop_mhc/mhc_experiment/model.py:77)、[GemmaRMSNorm](../vllm/vllm/model_executor/layers/layernorm.py:132)。

## 3. 拟实施结构

### 3.1 注册原生架构与构造路径

新增 `qwen3_5_loop.py`，注册 checkpoint 原有的 `Qwen3_5LoopForConditionalGeneration`。复用标准 Qwen3.5 的 embedding、算子、norm、lm_head 和外层处理接口。

在现有 `qwen3_5.py` 增加两处最小可覆盖构造入口，默认类保持不变：

1. `Qwen3_5ForCausalLMBase` 构造内层 `.model` 的入口。
2. `Qwen3_5ForConditionalGeneration` 构造 `.language_model` 的入口。

Loop 子类通过入口直接构造 Loop 模型。内层也要直接构造目标结构，不能先运行标准模型构造函数创建 32 层再替换；Attention/GDN 构造时已向 `static_forward_context` 注册，重复构造会造成重复注册或遗留状态。

**Loop 执行必须放在内层 `.model.forward`。** 现有多模态外层直接调用 `self.language_model.model(...)`，仅覆盖 CausalLM 的 `forward` 会被绕过。

保留原 checkpoint/HF 物理配置；另行派生执行计划、展开后的层类型和执行边界，必要时使用模型内部运行配置副本。运行时对执行层 32 及之后的类别查询，不能直接索引原始 32 项 `layer_types`。

参考：[内层构造](../vllm/vllm/model_executor/models/qwen3_5.py:332)、[外层构造](../vllm/vllm/model_executor/models/qwen3_5.py:505)、[外层 forward](../vllm/vllm/model_executor/models/qwen3_5.py:574)。

### 3.2 权重共享与状态隔离

- 按执行顺序构造 80 个执行层；相同物理层共享无状态的投影、MLP、norm 和 router，并共享 GDN 的 `A_log`、`dt_bias` 参数。
- Attention/GDN 的状态对象、执行层标识、缓存绑定各自独立，不能整体复用同一个模块。
- 构造后及时绑定已有物理层参数，避免先为全部执行层分配完整权重再统一替换。
- 每个 cache prefix 只包含一个数字索引，如 `...layers.52.linear_attn`。`...layers.4.loops.2.linear_attn` 会违反本地 `extract_layer_index` 约束。
- 最早出现的执行层是对应物理层的参数归属位置；权重后处理可能替换参数，需在完整加载及后处理之后再次检查共享关系。
- 缓存隔离检查依据实际槽映射及读写影响；不能仅凭底层 storage 指针是否不同判断，因为缓存可能来自同一个大 allocation 的不同 view。

参考：[索引提取](../vllm/vllm/model_executor/models/utils.py:905)、[加载后处理](../vllm/vllm/model_executor/model_loader/utils.py:96)、[缓存分配](../vllm/vllm/v1/core/kv_cache_utils.py:1325)。

### 3.3 前向残差接口

新的 Loop 内层模型统一传递完整残差表示。普通层语义为：

```text
h = x + Attention/GDN(norm1(x))
y = h + MLP(norm2(h))
```

mHC 层按第 2 节公式显式聚合及回写。不要把普通模型的延迟相加 residual tuple 直接接到多流表示上；每次进入/离开 mHC 时必须已有完整表示。复用原有 norm 和算子，不更改它们的数学含义，检查各边界形状与 dtype；不增加数值对照任务。

### 3.4 权重装载

checkpoint 共 857 个 tensor、11 个 shard：decoder 520 个、vision 333 个、embedding/norm/lm_head 3 个、关闭的 Loop gate 1 个。96 个 mHC router tensor 为 FP32，其余为 BF16；未发现目标模型 MTP 权重。

加载流程：先做既有外层名称映射，再把 decoder 的物理层索引映射到其首次执行位置，然后走标准 packed-weight 映射。每组物理参数只装载一次。

```text
first_occurrence[0..27]  = 0..27
first_occurrence[28..31] = 76..79
```

关键反例：checkpoint 的 `layers.28` 是后段物理层，必须落到执行层 76，不能误装到第二轮开始的执行层 28。

保留 QKV、GDN 的 qkv/z 与 b/a、MLP 的 gate/up 等标准融合映射。`load_weights` 返回的参数名必须与去重后的实际模型参数集合一致，不能为了通过缺失检查返回不存在的别名。

对额外或未使用权重实行明确处理：

- 仅当 `use_gate=false` 时允许忽略 `model.language_model.loop_gate.logits`。
- 第一阶段采用文本模式，通过现有 `language_model_only`/缺失视觉塔机制处理视觉参数。
- router 的 weight 和 bias 都必须检查完整装载，不能让默认忽略 bias 的行为掩盖缺失。
- 不因 `mtp_num_hidden_layers=1` 就假定此目录有可用 draft；用户脚本中的 MTP 来自独立目录。

### 3.5 配置和能力接口

在模型配置映射中让新架构复用 Qwen3.5 配置校验，保留 checkpoint 的 FP32 SSM cache 要求。保留 `HasInnerState`、`IsHybrid` 及 GDN state shape/dtype 接口，使 V2 runner 正确发现混合状态模型。

多模态外层的 processor 和 `_mark_language_model` 接口要保持可识别；这只是正确复用外层机制，不表示已验收图像输入。继承得到的 LoRA、PP、Eagle 等能力标志也不能自动当成已经支持，第一阶段须显式限制尚未验证的配置。

预计核心改动位置：

| 文件 | 拟改动 |
| --- | --- |
| `vllm/vllm/model_executor/models/qwen3_5_loop.py` | 新增 Loop/mHC 模型、执行映射、参数共享及加载逻辑 |
| `vllm/vllm/model_executor/models/qwen3_5.py` | 两处构造入口，默认行为保持原实现 |
| `vllm/vllm/model_executor/models/registry.py` | 注册新 architecture |
| `vllm/vllm/model_executor/models/config.py` | 配置校验映射及首阶段能力约束 |

该表是预期核心范围，不包含验收脚本；目前没有证据要求先修改全局 scheduler 或缓存管理器。若运行验证发现接口缺口，须记录原因再调整。

### 3.6 TP=8 下的分片与路由边界

用户已选择 TP=8、PP=1，按一个模型实例占用 8 张 GPU 实现，DP=1。静态源码核查未发现头数分片及残差宽度方面的结构性阻碍；这不等于已经完成 8 卡运行验证。

| 部分 | 全局布局 | 每个 TP rank 的布局 |
| --- | --- | --- |
| Full attention Q heads | 16 | 2 |
| Full attention KV heads | 4 | 1；每个原始 KV head 复制到两个 rank |
| GDN Q/K heads | 16 | 2 |
| GDN V heads | 32 | 4 |
| GDN `A_log` / `dt_bias` | 各 32 个元素 | 各 4 个元素 |
| Dense MLP 中间宽度 | 12288 | 1536 |
| mHC 残差表示 | 4 条流，每条宽度 4096 | 每个 rank 持有完整四条流 |
| mHC router | FP32，weight `[24,16384]`、bias `[24]` | 每个 rank 完整复制，不按 TP 分片 |

Full attention 的 KV 复制走现有 `QKVParallelLinear` loader；GDN 和 MLP 走各自现有 loader，不能把所有 checkpoint tensor 机械地八等分。参数跨 Loop round 的共享发生在同一 rank 内，仍需保留每个执行位置的独立状态身份。

目标模型是 dense `qwen3_5_text`。现有 Attention/GDN 的输出投影和 dense MLP 的 down projection 都可通过 TP all-reduce 返回完整 D 维结果，embedding 也恢复完整表示。因此，mHC router 在每个 rank 本地计算 route/merge，无需新增四条流之间的 TP 通信算法。这是源码数据流推导出的实现方式，后续要验证实际 collective 顺序和输出形状。

实现必须保留这些输出的 `reduce_results=True`，首阶段不引入改变残差接口的 sequence parallel/reduce-scatter 路径。Router 参数的完整复制不能受通用 TP 分片逻辑影响；相同物理层 router 在同一 rank 的不同循环轮次之间仍共享。

核实位置：[attention 的头数规则](../vllm/vllm/model_executor/models/qwen3_next.py:228)、[KV 复制配置](../vllm/vllm/model_executor/layers/linear.py:1018)、[KV 权重加载](../vllm/vllm/model_executor/layers/linear.py:1267)、[GDN 分片布局](../vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:493)、[dense 模型 reduce 设置](../vllm/vllm/model_executor/models/qwen3_5.py:136)、[MLP 输出投影](../vllm/vllm/model_executor/models/qwen2_moe.py:73)、[输出 all-reduce](../vllm/vllm/model_executor/layers/linear.py:1653)、[embedding all-reduce](../vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:503)。

## 4. 首阶段交付边界与运行检查

已确认范围：单机 8 卡、TP=8、PP=1、8K 文本目标模型推理；不做数值对照，不包含 TorchSpec 采集/训练和 MTP。BF16 主干、FP32 router 和 SSM、eager 作为首阶段实现基线。8K 作为 `max_model_len=8192`，计算输入与输出 token 总长度。

交付配置基线如下；这里只记录参数，不执行启动：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tensor_parallel_size=8, pipeline_parallel_size=1, data_parallel_size=1
max_model_len=8192, max_num_batched_tokens=8192
dtype=bfloat16, load_format=safetensors
language_model_only=true, enforce_eager=true
enable_prefix_caching=false, enable_chunked_prefill=false
mamba_ssm_cache_dtype=float32
```

首个 smoke request 可用较短输入和 `max_num_seqs=1`，但交付必须在 TP=8、PP=1 下完成加载、prefill 和多步 decode，并验证接近 8K 总长度的请求。随后以少量变长请求检查调度及状态释放；开发期单卡通过不能替代 8 卡验收。首阶段不设吞吐或时延指标。

沿用已激活环境的 V2 runner。关闭 chunked prefill 时，本地 scheduler 要求 `max_num_batched_tokens >= max_model_len`，上述配置满足此约束。8K prefill 的实际峰值显存仍由运行检查确认；不因静态头数条件成立就承诺内存容量。配置中的 `use_cache=false` 是导出字段，不能据此关闭 vLLM 的增量推理状态。

| 顺序 | 检查对象 | 通过条件 |
| --- | --- | --- |
| 1 | 配置与映射 | 架构能解析；32→80 映射、20/60 状态类别和后段权重位置正确 |
| 2 | 完整加载 | 8 个 rank 成功加载；KV 复制和 GDN 分片形状正确；缺失/多余参数均可解释；router 保持 FP32；加载后参数仍共享 |
| 3 | 执行层结构 | 无重复状态注册；mHC expand/collapse 的形状与 dtype 正确；各执行层缓存槽正确绑定 |
| 4 | 8 卡前向与生成 | TP=8、PP=1 下完成短输入及接近 8K 总长度请求的 prefill 和多步 decode；无 OOM、形状错误、非法索引、非有限输出或通信挂起 |
| 5 | 请求生命周期 | 变长请求、请求结束和槽复用正常；缓存状态绑定与释放正确，无运行错误 |
| 6 | 用户输入输出 | 使用已选 tokenizer、chat template、EOS 和停止条件，完成端到端文本请求并正常终止 |

不运行 LightLLM/训练实现对照，不采集参考 hidden states/logits，不建立 BF16 误差阈值，不用整段重算作为 decode 对照。上述项目是功能和结构检查；不会把它们记录成数学等价性结论。

checkpoint 外层 EOS 为 248046，text_config EOS 为 248044；实现时核对实际使用的 generation config 和模板，避免提前停止或无法停止。此项属于输入输出配置检查，不需要运行对照模型。

首阶段之外的功能包括 chunked prefill、prefix caching、多模态、编译/CUDA Graph、MTP、TorchSpec 采集和 256K 长上下文。若为解决实际运行问题确需调整实现基线，应记录原因，不默默扩大交付范围。

TorchSpec 采集问题保留为后续记录：现有逻辑从配置读取物理层数，接入前需要明确物理层、循环轮次、流收敛方式及 norm 前后语义；不能把 32 直接替换为 80 就认定采集正确。参考：[当前采集配置](../torchspec/inference/engine/vllm_engine.py:201)。

## 5. 设计树与当前问题

“已核实”描述源码事实；“方案基线”描述前述拟实现方式；“待决定”描述用户尚未回答的选择。两者都不替代运行验收。

```text
接入 ckpt-225-bf16
├─ 已核实：Loop/mHC 数学语义、参数布局、当前框架接口
├─ 方案基线：原生架构、共享参数、独立执行状态、两处构造入口
├─ D1 [已确认] 单机 8 卡目标模型推理，不含 TorchSpec/MTP ← 第 1 轮 Q1
│  ├─ D1.1 [已确认] TP=8、PP=1，一个 8 卡模型实例 ← 第 2 轮 Q3
│  └─ D1.2 [已确认] 首阶段 8K ← 第 2 轮 Q4
│     └─ 实现基线：eager、非分块 prefill；运行确认 8K 峰值显存
└─ D2 [已确认] 不需要做对照 ← 第 1 轮 Q2
   └─ 保留权重、结构、8 卡生成及请求生命周期检查
```

当前没有待回答的范围问题。实现基线依照上述已确认范围收敛；具体算子约束由源码核查，内存和通信行为留给实施后的运行检查，不要求用户提供环境中已有的事实。

### 第 1 轮：已回答

**Q1：首阶段交付范围是什么？**

用户回答：“先完成单机 8 卡的推理就行”。

影响：替换此前单卡目标；首阶段包含 8 卡推理，TorchSpec 训练和 MTP 留在后续阶段。8 卡的 TP/DP 拓扑尚未指定。

**Q2：若当前 LightLLM 与该 checkpoint 的训练语义存在差异，最终以谁为准？**

用户回答：“不需要做对照”。

影响：关闭“比较基准及容差”分支；已核实源码继续作为实现依据，原训练 checkout 同版确认不再是本次交付前置条件。用户没有选择某个实现的运行输出作为数值基准。

### 第 2 轮：已回答

**Q3：一个模型分布到 8 张卡（TP=8），还是 8 个单卡副本（TP=1、DP=8）？**

用户回答：“先 TP=8,PP=1 来运行”。

影响：交付为一个 8 卡 TP 模型实例，DP=1；TP 分片、KV 复制、FP32 router 全量复制和 collective 运行检查进入首阶段。

**Q4：首阶段上下文需要 8K，还是原脚本的 256K（262144 tokens）？**

用户回答：“先支持 8k”。

影响：交付使用 `max_model_len=8192`；256K 延后，不将其作为首阶段成功条件。较短输入仅用于开发期 smoke request，最终仍检查 8K 长度支持。

## 6. 8K 设计阶段的执行边界（历史）

当时只新增或更新设计文档、ADR 和术语表，尚未修改模型实现、配置、权重或启动服务。之后已完成 8K 实现和运行验收，见文件开头的运行报告；这段记录不再代表当前实现状态。256K 的决策与执行边界以新的扩展设计问答为准。
