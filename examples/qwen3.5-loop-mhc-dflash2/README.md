# Qwen3.5 Loop/mHC DFlash2 单机八卡训练

四张 H800 运行 TP=4、PP=1 的 vLLM V2 目标引擎，另外四张运行 FSDP2
`FULL_SHARD` 草稿训练。复用 `torchspec.train_entry`、`DFlash2Trainer`、
Loop/mHC direct capture、Mooncake TCP 和已有 checkpoint 保存/恢复实现。

默认目标为 `1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16`，
默认数据为 `mydata/test_20.jsonl`。完整对话作为上下文，只监督最后一个 assistant
回答；目标模型、目标 embedding 和 LM head 冻结，草稿使用 CE + selector loss。
目标端只对已有样本做 prefill，不加载或同步训练中的草稿权重。

## 运行

在仓库根目录执行：

```bash
bash examples/qwen3.5-loop-mhc-dflash2/run.sh --dry-run
bash examples/qwen3.5-loop-mhc-dflash2/run.sh --print-config-only
bash examples/qwen3.5-loop-mhc-dflash2/run.sh --smoke
bash examples/qwen3.5-loop-mhc-dflash2/run.sh
```

脚本先调用仓库根目录的 `cp_envs.sh`，激活
`/tmp/deepspec_vllm_torchtitan_envs`，再复用 `h800conda-tmp.sh` 设置本仓库
vLLM 的导入路径、CUDA 库和本地 Mooncake 的代理绕过地址。
如果 shell 尚未加载 Conda，默认加载
`/mnt/afs_share/miniconda3/etc/profile.d/conda.sh`；可用 `CONDA_SH` 指定其位置。
始终启用 `VLLM_USE_V2_MODEL_RUNNER=1`。

`--smoke` 保持 **4 卡目标 + 4 卡训练及完整草稿结构**，改用仓库现有的
`examples/data/sample_conversations.jsonl`，以 8K 长度、32 anchors 运行两次
optimizer 更新，每步保存，并开启已有的参数更新验证。
使用该数据是因为默认长对话裁到 8K 后，大多数最后一轮回答会被裁掉。
普通模式沿用 128K 长度、512 anchors、global batch=4 和一轮数据；每两步及
轮末保存，保留最新两份完整 checkpoint。128K 包含内部生成的一个 token，
实际训练输入最多为 131071 tokens。

输出目录默认带时间戳，日志位置在启动时打印，默认位于 `running_logs/`。
可用 `TARGET_MODEL_PATH`、`TRAIN_DATA_PATH`、`OUTPUT_DIR`、`CACHE_DIR`、
`LOG_DIR` 覆盖路径；命令末尾的 `key=value` 参数优先于普通及 smoke 默认值。
`RAY_ADDRESS` 默认 `local`，在当前机器创建 Ray 实例。

完整训练状态恢复到新的输出目录：

```bash
LOAD_PATH=/absolute/path/to/original-output/checkpoints \
CKPT_STEP=1 \
OUTPUT_DIR=/absolute/path/to/new-resume-output \
bash examples/qwen3.5-loop-mhc-dflash2/run.sh --smoke
```

不设置 `CKPT_STEP` 时读取最新 checkpoint。若已经完成 smoke 的第 2 步，
继续运行时追加 `training.num_train_steps=3`。smoke 的 LR 总步数固定为 5，
以便验证后续恢复时仍有非零学习率。脚本检查 model、optimizer、scheduler、RNG、
metadata 是否齐全，并拒绝将恢复输出写回原 checkpoint 目录。
原有训练契约还会检查目标身份、草稿结构、采集点和监督设置；异步样本队列的恢复
不保证逐样本重放。

## 模型与采集约定

专用草稿配置为
[`qwen3_5_loop_mhc_dflash2.json`](../../configs/draft_models/qwen3_5_loop_mhc_dflash2.json)：

- 五层、hidden size 4096、FFN 8192、Q/KV heads 16/4、head dimension 256。
- 沿用现有 DFlash2 配方：前四层 sliding attention（窗口 2048），最后一层
  full attention；块内双向 attention，block size=8（1 个已知 token + 7 个预测 token）。
- selector rank=256、top-k=16，完整词表 248320；mask token=248044，
  与已有 Loop/mHC 草稿约定一致，部署时须保持一致。
- 32 个物理层展开为 80 个执行层，采集零基执行层 `[3,27,51,75,78]`，
  对应辅助槽 `[4,28,52,76,79]`。草稿配置明确使用执行深度 80。

沿用 [Loop/mHC 采集契约](../../docs/adr/0002-loop-mhc-dspark-capture-contract.md)：
direct-capture connector 仍发布独立的 post-norm 最终槽 80；本示例 L1 权重为零，
损失不使用最终状态，也不加载目标 final norm；现有数据传输路径仍读取该槽。
保留该槽可复用既有采集路径，
不能把 DSpark 的 L1/confidence loss 配置直接复制到 DFlash2。

Loop/mHC 目标当前只支持 TP=4 或 TP=8；不要通过改成 TP=2 缩小调试。
四卡资源不足以同时运行本示例的目标和训练两个角色。

## 2026-10-27 调试结果

已使用 `cp_envs.sh` 的环境完成**单机八卡在线训练、完整恢复和默认长对话训练**。
实际分配为 GPU 0–3 运行四个 FSDP rank，GPU 4–7 运行一个 TP=4 目标引擎。
三个八卡运行均正常退出，退出码为 0。

| 检查 | 结果 |
| --- | --- |
| 自动测试 | 162 项测试、70 项子用例通过 |
| 八卡 smoke | 在线采集并完成两次更新，保存第 1、2 步；loss、梯度和实际参数更新检查通过 |
| 八卡完整恢复 | 从 smoke 第 1 步恢复模型、optimizer、FP32 master、scheduler 和 RNG，继续更新到第 2 步 |
| 默认长对话 | `mydata/test_20.jsonl` 的 20 条样本全部参与，512 anchors、global batch=4，完成一轮共 5 次更新；保留第 4、5 步 checkpoint |
| 分块采集 | 共 729182 输入 tokens，最长 105647 tokens；所有训练样本的采集长度与预处理结果一致，块间位置连续，最长样本跨 13 个 8192-token prefill 块 |
| checkpoint | 三个运行的最终 checkpoint 均通过检查：82 个模型张量、324 个 optimizer 张量均有限，81 组 Adam 状态推进到对应的第 2 或第 5 步；冻结 embedding 与目标完全相同，投影和 selector 参数均有实际更新 |

调试中修复了 vLLM 引擎将非连续 Ray GPU 分配错误展开为连续编号的问题，回归测试
先失败、修复后通过。此前四卡资源下的分阶段检查也已通过：先采集 8 条真实目标
特征，再复用 `MooncakeDataset`、`OfflineDataset` / `OfflineReplayEngine`
完成训练和恢复。临时适配器及 checkpoint 检查脚本位于 `.codex-tmp/loop-dflash2/`；
八卡入口直接使用在线采集链路。

完整记录见 [`verification.json`](verification.json)。实机产物：

- [smoke 参数检查](../../outputs/loop_dflash2_8gpu_smoke_20261027/parameter_verification.json)
  和[运行日志](../../running_logs/qwen3.5-loop-mhc-dflash2_20261027_210629.log)。
- [恢复参数检查](../../outputs/loop_dflash2_8gpu_resume_20261027/parameter_verification.json)
  和[运行日志](../../running_logs/qwen3.5-loop-mhc-dflash2_20261027_211004.log)。
- [长对话采集检查](../../outputs/loop_dflash2_8gpu_default_20261027/capture_verification.json)、
  [最终参数检查](../../outputs/loop_dflash2_8gpu_default_20261027/parameter_verification.json)、
  [训练指标](../../outputs/loop_dflash2_8gpu_default_20261027/metrics.json)
  和[运行日志](../../running_logs/qwen3.5-loop-mhc-dflash2_20261027_211342.log)。

退出时仍有 Mooncake 卸载和 Ray 队列关闭告警，日志已保留；本次 GPU worker
均已释放。当前验证长度最高为 105647 tokens，未验证满 128K 边界、长期收敛、
草稿接受率或解码加速比。
