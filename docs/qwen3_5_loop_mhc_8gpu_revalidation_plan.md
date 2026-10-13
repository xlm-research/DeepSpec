# Qwen3.5 Loop/mHC：加速实现的单机八卡 DSpark 训练验收

记录日期：2026-10-12。用户已明确本轮运行 `examples/qwen3.5-loop-mhc-dspark`，不是独立 vLLM 服务。此前本文件中的 TP=8 独立生成复验方案已被这一明确范围替代。

**结果：本轮训练通过。** 单机八张 H800 上启用 Triton 融合，四卡目标模型供样、四卡 FSDP 训练，20 条有效样本完成五次 optimizer 更新。进程退出码为 0，最终 checkpoint 检查通过，退出后八卡均回到 4 MiB、0% 利用率且没有 GPU 计算进程。

## 已确认范围

- 执行[单机训练入口](../examples/qwen3.5-loop-mhc-dspark/run.sh)，使用其现有配置：四卡 TP=4 目标模型供样，加四卡 FSDP 训练 DSpark。
- 启用已有 Qwen Loop FP32 log-Sinkhorn Triton 融合，保留 BF16 主干、FP32 router/SSM、三轮 Loop、四条流与 20 轮 Sinkhorn。
- 使用默认 `mydata/test_20.jsonl`，已有回答监督、完整对话上下文、仅最后一轮回答计算损失。五份条件特征与独立 post-norm 最终状态保持原契约。
- 完整运行一轮：20 条有效样本、全局 batch=4、五次 optimizer 更新；检查第 2、4、5 步保存及最近两份 checkpoint。
- 当前草稿配置为 Q heads=16、KV heads=4、head dimension=256。README 中旧的 32/8/128 训练记录不能替代此次运行。
- 本轮核验训练链路和融合实际执行，不重新建立独立生成 TTFT 基线。

## 已核实事实

- 启动前八张 H800 均空闲。`run.sh` 加载 `h800conda.sh` 后使用本仓库 vLLM 源码。
- 目标 checkpoint 的 11 个权重 shard 及 tokenizer/config 文件可读。
- 当前 CPU 预处理确认 20/20 样本有效、无过滤，输入长度 7177–105647，共 729182 tokens；监督 token 共 6853。该默认数据集不能证明完成了 131071-token 满长训练。
- `VLLM_QWEN3_5_MHC_BACKEND` 默认是 `torch`；本轮显式设置 `triton`。
- 发现 TorchSpec 的 Ray 环境白名单未显式转发该开关，已在 `torchspec/utils/env.py` 补充透传，不改变默认后端。相关五项环境测试通过。
- 使用新的输出目录与缓存目录，确保执行本轮预处理并保留既有产物。

## 运行与验收

本次运行目录：`outputs/loop_dspark_triton_20261012_231245/`。`invocation.json` 记录入口、关键环境值、源码及数据 SHA256；`config.yaml` 保存有效配置，`logs/` 保存启动日志，`gpu_samples.log` 每十秒记录资源用量。

运行命令等价于：

```bash
OUTPUT_DIR="$PWD/outputs/loop_dspark_triton_20261012_231245" \
CACHE_DIR="$PWD/cache/loop_dspark_triton_20261012_231245" \
LOG_DIR="$PWD/outputs/loop_dspark_triton_20261012_231245/logs" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_ADDRESS=local \
RAY_DEDUP_LOGS=0 TORCHSPEC_LOG_LEVEL=DEBUG \
VLLM_QWEN3_5_MHC_BACKEND=triton \
bash examples/qwen3.5-loop-mhc-dspark/run.sh
```

通过条件：

1. 核对四个目标 TP worker 的实际进程环境均选择 Triton，并保留实际 kernel 执行日志。vLLM 的 `info_once` / `warning_once` 默认仅节点首 rank 输出，因此日志不能证明所有 rank 逐请求零回退；本轮不宣称进行了调用计数审计。
2. 20 个有效样本完成采集与派发；采集槽为 `[4, 28, 52, 76, 79, 80]`，分块位置连续。
3. 五步 loss 与梯度均有限，完成实际 optimizer 更新，训练正常退出。
4. 最终 checkpoint 的模型、optimizer、scheduler、RNG 和元数据完整；最终 Adam 步数正确，草稿参数发生变化，冻结 embedding 保持目标权重。
5. 核对退出码与资源释放，保留运行中异常和范围限制。

## 本轮结果与原始证据

| 检查 | 实测结果 |
| --- | --- |
| 训练消费 | 五次派发，每次四条；20 个不同样本全部覆盖，无重复消费 |
| 融合启用 | 四个 TP worker 进程环境均为 `triton`；TP0 实际 kernel 执行日志存在，未观察到回退警告 |
| 采集契约 | 采集槽 `[4, 28, 52, 76, 79, 80]`，每个请求的分块位置连续 |
| 训练更新 | 五步 CE/L1/confidence loss 和梯度均有限；62 组 Adam 状态从第 4 步推进到第 5 步 |
| 参数变化 | 第 4→5 步 context projection 有 27207796 个元素发生变化；final norm FP32 master 相对初始化有 4094 个元素变化 |
| 模型与冻结权重 | 63 个模型张量全部有限；冻结 embedding 与目标权重逐元素一致；optimizer 不含 embedding/LM head |
| 保存与保留 | 成功保存第 2、4、5 步，最终保留第 4、5 步，latest=5；model/optimizer/scheduler/RNG/meta 完整 |
| 新草稿结构 | checkpoint 契约确认为 Q=16、KV=4、head_dim=256、五层；训练 world_size=4 |
| 资源 | 每十秒采样的训练卡峰值 48920–59732 MiB，目标卡 67340–67883 MiB；无 OOM，结束后全部释放 |
| 环境透传修复 | `tests/test_env.py` 五项通过；Ruff 检查、格式检查及 `git diff --check` 通过 |

各步指标如下，梯度值为日志中的裁剪前范数；此处仅验证有限性，不把五步 smoke 运行作为质量或收敛结论。

| optimizer step | CE | L1 | confidence | grad norm |
| --- | --- | --- | --- | --- |
| 1 | 13.135996 | 1.998700 | 0.701978 | 102.523926 |
| 2 | 10.027761 | 1.991825 | 0.102716 | 3.667074 |
| 3 | 28.511492 | 1.949789 | 0.382539 | 7.905530 |
| 4 | 22.735525 | 1.896617 | 0.660575 | 5.743868 |
| 5 | 17.189199 | 1.915719 | 0.431044 | 4.739573 |

证据位置：

- [运行汇总、逐步指标及采集分块](../outputs/loop_dspark_triton_20261012_231245/run_summary.json)
- [真实 checkpoint 参数与 optimizer 检查](../outputs/loop_dspark_triton_20261012_231245/parameter_verification.json)
- [启动环境、入口和源码/数据身份](../outputs/loop_dspark_triton_20261012_231245/invocation.json)
- [四个目标 TP worker 的实际环境](../outputs/loop_dspark_triton_20261012_231245/worker_environment.json)
- [退出、资源释放和融合日志证据](../outputs/loop_dspark_triton_20261012_231245/end_state.json)
- [完整 driver 日志](../outputs/loop_dspark_triton_20261012_231245/raw_logs/driver.log)
- [最终 checkpoint](../outputs/loop_dspark_triton_20261012_231245/checkpoints/iter_0000005/meta.json)

## 边界与收尾观察

- 实际训练消费 20 条样本，共 729182 输入 tokens。后台预取额外采集两条未训练样本，因此日志共记录 22 次采集、862130 行表示；不能将全部采集行数算作训练消费量。
- 本轮最长输入为 105647 tokens，配置上限为 128K；没有额外构造满长 131071-token 训练输入，也没有执行恢复训练或性能前后对照。
- 融合日志受节点首 rank 过滤，未读取逐 rank 的调用计数；本结果不声称完成了全 rank 零回退审计或新的数值对齐验收。
- 训练完成后的资源清理阶段出现 `_QueueActor` owner `INTENDED_SYSTEM_EXIT` 导致的 fetcher 警告，以及 vLLM 强制终止剩余 EngineCore 的警告。最终 driver 退出码为 0，checkpoint 检查通过，八卡资源全部释放；不将日志描述为无告警退出。

没有新的领域术语或难以逆转的架构取舍，沿用 [CONTEXT.md](../CONTEXT.md) 和[采集契约 ADR](adr/0002-loop-mhc-dspark-capture-contract.md)。
