# Qwen3.5 Loop/mHC：默认融合后的单机八卡训练功能验收

记录日期：2026-10-15。

**结果：单机八卡训练与独立恢复功能验收均通过。** 默认 20 条数据完成五次更新，再从本轮第 4 步恢复并实际完成第 5 步。两次 driver 均退出 0，八卡与本轮服务均已释放；清理告警及诊断证据限制见下文。

用户已确认 Q1、Q2 均采用推荐范围，验证 `examples/qwen3.5-loop-mhc-dspark` 能否正常运行。本轮不测速，也不追加满长边界输入。

## 已确定的目标与事实

- 使用示例原有 `run.sh` 和 `config.yaml`，单机八张 H800：四卡 TP=4 目标模型供样，加四卡 FSDP 训练 DSpark 草稿模型。这里的八卡不是独立目标模型 TP=8。
- 验证当前 vLLM 默认的 `VLLM_QWEN3_5_MHC_BACKEND=triton` 和 `VLLM_QWEN3_5_MHC_FUSION=all`。运行时清除这两个变量的继承覆盖，以检查库本身的默认值。
- 保留目标 checkpoint、当前五层草稿结构 Q=16/KV=4/head_dim=256、BF16 主干、FP32 SSM、8192-token 分块及现有监督与采集契约。
- 默认数据为 `mydata/test_20.jsonl`。既有预处理记录为 20 条有效输入，长度 7177–105647；当前文件 SHA256 与该记录一致。这是历史统计，本轮仍需独立预处理核验。
- 默认 microbatch=1、训练 world size=4、累积=1，一轮预期完成五次 optimizer 更新。第 2、4、5 步保存，最终保留第 4、5 步。
- 本轮只读预检时八张 H800 空闲，目标 checkpoint 的 11 个权重 shard 和 tokenizer/config 文件可读。实际启动前再次确认资源状态。
- 10/12 的当前草稿结构训练成功记录仅覆盖 Sinkhorn 融合；新 norm/pre/merge 融合的 TP=8 独立推理通过，不能替代本轮 TP=4 供样与四卡训练验收。

## 已确认的范围

| 问题 | 已确认范围 | 未选择范围 |
| --- | --- | --- |
| Q1：验收终点 | 默认一轮完整训练与保存；再从本轮第 4 步恢复到第 5 步 | 只验证默认一轮训练与保存 |
| Q2：数据与长度 | 默认 20 条数据完整一轮 | 默认数据通过后，追加四条满长边界输入的一次更新 |

这两项回答同时确认本文件对应的运行范围。没有新的领域术语或难以逆转的架构决定，沿用根术语表和现有采集契约 ADR。

## 执行方式

为本轮创建独立的输出、缓存和日志目录，保留既有运行产物。运行前记录有效配置、源码身份和数据摘要；清除继承的 `LOAD_PATH` 与 `CKPT_STEP`，确保主运行从头训练。日志保留训练、采集、保存和异常信息，通过命令行覆盖 `debug.enable_perf_metrics=false`。

主运行命令形态如下；目录后缀在启动时生成唯一值：

```bash
validation_dir="$PWD/outputs/loop_dspark_functional_<unique>"
env -u VLLM_QWEN3_5_MHC_BACKEND \
    -u VLLM_QWEN3_5_MHC_FUSION \
    -u LOAD_PATH -u CKPT_STEP \
    OUTPUT_DIR="$validation_dir" \
    CACHE_DIR="$validation_dir/cache" \
    LOG_DIR="$validation_dir/logs" \
    TRAIN_DATA_PATH="$PWD/mydata/test_20.jsonl" \
    TARGET_MODEL_PATH=/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    RAY_ADDRESS=local RAY_DEDUP_LOGS=0 TORCHSPEC_LOG_LEVEL=DEBUG \
    bash examples/qwen3.5-loop-mhc-dspark/run.sh \
    debug.enable_perf_metrics=false
```

本轮验证包装器在训练数据提交前、训练循环返回且引擎清理前收集只读状态，原示例入口负责配置与启动。逐 rank 计数读取尝试经过 Ray `__ray_call__` 转发到 vLLM `collective_rpc`，实际被 vLLM 默认序列化策略拒绝；最终证据范围见结果部分。包装器不改变模型算法、采样或训练步骤。

## 主运行通过条件

1. 本轮预处理确认有效数据及监督长度，20 条样本被训练实际消费，完成五次 optimizer 更新。后台额外预取与训练消费分别统计。
2. 四个目标 TP worker 的实际配置均为 `triton/all`；若计数诊断可用，四个 rank 的 Sinkhorn 与 norm/pre/merge Triton 计数增量均大于零，Torch 回退增量为零。
3. 五个条件特征与独立 post-norm 最终状态按采集槽 `[4, 28, 52, 76, 79, 80]` 交付，分块位置连续，行数与输入对齐，传输及训练消费无异常。
4. 各步 CE、L1、confidence loss 和梯度范数有限；草稿参数、optimizer 状态有限，真实可训练参数发生改变；冻结 embedding 与目标权重保持一致。
5. 第 4、5 步 checkpoint 的 model、optimizer、scheduler、RNG、meta 完整，步数与当前草稿/采集契约一致。不能仅以目录存在判定保存成功。
6. 主进程退出 0，退出后本轮 GPU 计算进程及相关服务清理完成。异常和清理告警保留并解释，不把功能通过描述为完全无告警。

不建立 TTFT、吞吐或加速比基线；有限步功能验收不用于声称模型质量或收敛。

## 恢复训练

从主运行 `checkpoints/iter_0000004` 显式恢复，使用新的 `OUTPUT_DIR`，保持原一轮、总步数五的配置，完成第 4→5 步。

不能直接从默认 latest=5 恢复：controller 在 `completed_steps == num_steps` 时不执行更新，仍可能生成新的第 5 步保存文件。也不通过单独增加总训练步数绕过，因为加载的 scheduler 仍保留原总步数，最后一步学习率已降为零。

验收需证明初始 global step=4，model/optimizer/scheduler/RNG 均实际恢复，结束 global step=5，Adam 状态由 4 推进到 5，部分可训练参数确实改变，冻结参数不变。scheduler 加载失败目前可能仅产生 warning，必须检查加载证据。异步样本队列恢复是 best-effort，不要求恢复后的第 5 步与主运行第 5 步逐样本或逐位一致。

## 本轮不执行的满长边界

候选数据为现有 `outputs/loop_dspark_verification_inputs/128k.parquet`，四条输入各 131071 tokens、监督各 512 tokens，执行当前草稿结构与默认融合的一次真实更新。运行前再次核验该文件和长度。

本训练示例的 `max_model_len=131072` 包含内部生成的一个 token，因此满长输入是 131071。独立推理测试中的 131072-token 输入加一个输出不属于此处默认配置边界。10/10 旧草稿结构的边界测试不能代替当前运行。

## 依据

- [示例配置](../examples/qwen3.5-loop-mhc-dspark/config.yaml)与[启动入口](../examples/qwen3.5-loop-mhc-dspark/run.sh)
- [现有训练规格](qwen3_5_loop_mhc_dspark_spec.md)与[采集契约 ADR](adr/0002-loop-mhc-dspark-capture-contract.md)
- [10/12 单机八卡训练记录](qwen3_5_loop_mhc_8gpu_revalidation_plan.md)
- [新增融合的独立推理结果与限制](qwen3_5_loop_mhc_fusion_results.md)

## 本轮实测结果

产物根目录为 `outputs/loop_dspark_functional_20261015_Nt1mh5/`，主运行和恢复分别使用 `main/`、`resume/`。完整启动脚本与只读检查工具保存在同一目录。

**主运行与独立恢复均通过。** [机器可读总结果](../outputs/loop_dspark_functional_20261015_Nt1mh5/verification.json)记录范围、检查工具摘要和证据位置。

| 主运行检查 | 实测结果 |
| --- | --- |
| 数据与消费 | 本轮独立预处理 20 条有效样本，7177–105647 input tokens，共 729182；监督共 6853 tokens。20 个不同样本各训练一次 |
| 训练与保存 | 完成五次 optimizer 更新；loss 与梯度均有限；保存第 2、4、5 步，最终保留第 4、5 步 |
| 参数与 optimizer | 63 个模型张量以及 optimizer 张量全部有限；62 组 Adam 状态从 4 推进到 5；context projection 的 27207796 个元素在第 4→5 步改变 |
| 冻结权重 | 保存的 embedding 与目标权重逐元素一致；optimizer 不含 embedding/LM head |
| 采集与传输 | 槽 `[4, 28, 52, 76, 79, 80]`，每个请求分块位置连续；训练端 20 条样本的条件特征、最终状态、输入和 loss mask 行数均对齐；后台额外预取两条，因此采集 22 次、862130 行，训练消费仍为 20 条 |
| 退出与资源 | driver 退出 0；八卡显存回到 0 MiB，GPU 计算进程为空 |

| 独立恢复检查 | 实测结果 |
| --- | --- |
| 恢复来源 | 本轮主运行 `iter_0000004`，新输出目录；四个训练 rank 均记录 model、optimizer、scheduler 加载成功 |
| 实际步数 | 只读快照为四个 rank 的 `[4,4,4,4] → [5,5,5,5]`；实际执行一次更新，CE/L1/confidence loss 与梯度均有限 |
| 数据消费 | 跳过前 16 条，实际消费 `sample_7/8/0/3` 四条，共 142024 input tokens；四条的训练特征行数均对齐 |
| 参数变化 | 63 个模型张量与 248 个 optimizer 张量全部有限；62 组 Adam 状态从 4 推进至 5；context projection 有 27206915 个元素相对恢复源改变 |
| 冻结与保存 | embedding 与目标逐元素一致；model、optimizer、scheduler、RNG、meta 完整，最终 global step=5 |
| 退出与资源 | driver 退出 0；八卡显存回到 0 MiB，GPU 计算进程、本轮目标进程与 Mooncake 服务均已释放 |

恢复的后台预取共采集八条、369937 行；只有上述四条进入本次更新。两次运行的 13 个训练/推理源码、示例与草稿配置、输入数据摘要一致，见[输入身份比较](../outputs/loop_dspark_functional_20261015_Nt1mh5/input_identity_comparison.json)。恢复源 RNG 文件已完整检查，正常恢复路径执行 RNG 加载；本验收仍不承诺异步队列的逐样本或逐位重放。

融合证据的限制：四个 TP worker 未设置 backend/fusion 环境覆盖，使用当前库默认值；TP0 的日志记录 Sinkhorn、normalize、pre_aggregate、residual_merge 实际执行 Triton。附加 callable RPC 被 vLLM 默认序列化策略拒绝，未取得逐 rank 调用计数；本轮不声称全 rank 零回退。该诊断异常由包装器捕获，未导致训练终止。恢复运行保留相同 vLLM 配置，跳过这项已知不可用的计数 RPC，仍读取四个 trainer 的实际起止步数。

收尾日志包含队列 owner `INTENDED_SYSTEM_EXIT` 导致的 fetcher 告警，以及 vLLM 终止剩余 EngineCore 的告警；恢复运行还出现 Mooncake segment/metadata 取消注册和析构卸载错误。它们均发生在更新和保存完成之后，最终退出与资源检查通过。本结果不表示无告警退出。首批派发曾等待 Ray 队列首次反序列化时的 Python 依赖导入；进程栈留档，随后正常推进，未修改训练代码。

主运行证据：

- [功能检查结果](../outputs/loop_dspark_functional_20261015_Nt1mh5/main/functional_result.json)
- [checkpoint 参数与 optimizer 检查](../outputs/loop_dspark_functional_20261015_Nt1mh5/main/parameter_verification.json)
- [训练消费与采集记录](../outputs/loop_dspark_functional_20261015_Nt1mh5/main/run_summary.json)
- [数据检查](../outputs/loop_dspark_functional_20261015_Nt1mh5/main/dataset_verification.json)
- [完整启动日志](../outputs/loop_dspark_functional_20261015_Nt1mh5/main/driver.log)

恢复证据：

- [功能检查与四个 rank 的起止步数](../outputs/loop_dspark_functional_20261015_Nt1mh5/resume/functional_result.json)
- [恢复后参数与 optimizer 比较](../outputs/loop_dspark_functional_20261015_Nt1mh5/resume/parameter_verification.json)
- [训练消费与采集记录](../outputs/loop_dspark_functional_20261015_Nt1mh5/resume/run_summary.json)
- [退出与服务清理](../outputs/loop_dspark_functional_20261015_Nt1mh5/resume/end_state.json)
- [完整恢复日志](../outputs/loop_dspark_functional_20261015_Nt1mh5/resume/driver.log)
- [恢复后的最终 checkpoint](../outputs/loop_dspark_functional_20261015_Nt1mh5/resume/checkpoints/iter_0000005/meta.json)
