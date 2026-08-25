# DSpark FSDP2 128K 100-step benchmark

本文记录 DeepSeek-V4-Flash DSpark 在单机 8 卡上的 128K 长序列训练速度与
loss 变化。实验于 2026-08-24 完成。这里的 100 轮指 100 个 optimizer
steps，不是 100 个 dataset epochs。

## 实验目的

- 验证 128K 真实长样本下 DSpark FSDP2 训练链路能够连续运行 100 steps。
- 测量不包含模型权重保存时间的纯训练耗时。
- 测试期间不保留 model/optimizer checkpoint。
- 观察 DSpark total loss 以及各 loss 分量的变化。

## 硬件与并行配置

| 项目 | 配置 |
| --- | --- |
| GPU | 8 × NVIDIA B300 SXM6 AC |
| 单卡可见显存 | 275,040 MiB（约 268.6 GiB） |
| Precision | BF16 |
| Sequence length | 131,072 tokens |
| Global batch size | 4 |
| Local batch size | 1 |
| Optimizer steps | 100 |
| Anchors | 512 |
| Draft model | EP 1 |
| Online target model | EP 8 |
| Context parallel | CP 2 |
| Tensor parallel | TP 1 |
| FSDP | FSDP2 full shard，dp_shard 4 |
| Learning rate | 1e-5，linear warmup + cosine decay |

训练数据使用：

```text
train_data/spec_o3_coldstartsft.repeat60.deepspec.packed_256k.jsonl
```

数据集包含 435 条 packed records，打包目标长度为 262,144 tokens。本实验的
collator 截取真实的前 131,072 tokens。抽检样本的 attention tokens 同样为
131,072，因此不是用 padding 伪造的 128K；该抽检样本包含 73,136 个有效
loss tokens。

## Benchmark 实现与复现

生产训练脚本保持不变：

```text
scripts/ fsdp/train_deepseek_v4_flash_dspark_fsdp2.sh
```

本实验使用独立 benchmark 脚本：

```text
scripts/ fsdp/benchmark_deepseek_v4_flash_dspark_fsdp2_128k_100steps.sh
```

运行命令：

```bash
bash "scripts/ fsdp/benchmark_deepseek_v4_flash_dspark_fsdp2_128k_100steps.sh"
```

benchmark 入口为 `scripts/benchmark_train_no_checkpoint.py`。它在 trainer
实例构建完成后，将 `save_and_eval_checkpoint` 替换为无操作函数，不修改生产
trainer，也不修改原始训练脚本。计时流程如下：

1. 完成模型加载、并行 mesh 构建和 trainer 初始化；
2. 所有 rank 执行 barrier 和 CUDA synchronize；
3. 开始 wall-clock 计时；
4. 执行 100 个完整 optimizer steps；
5. 再次 CUDA synchronize，并对各 rank 耗时取最大值；
6. 停止计时并清理进程，不写入 checkpoint。

因此计时包括 dataloader、在线 target forward、draft forward、backward、梯度
同步和 optimizer step，但不包括模型初始化、checkpoint 保存和进程清理。

## 速度结果

benchmark 输出：

```json
{
  "optimizer_steps": 100,
  "training_seconds": 5409.326002452988,
  "seconds_per_step": 54.09326002452988,
  "checkpoint_writes": 0
}
```

汇总如下：

| 指标 | 结果 |
| --- | ---: |
| 100-step 纯训练时间 | 5,409.326 秒 |
| 分钟表示 | 90 分 9.3 秒 |
| 全程平均速度 | 54.093 秒/step |
| 去除首步后的稳态速度 | 53.492 秒/step |
| Global samples/s | 0.07395 |
| 聚合输入 tokens/s | 约 9,692 |

聚合 tokens/s 按 `global_batch_size × sequence_length / seconds_per_step`
计算，即每个 step 处理 4 × 131,072 个输入 tokens。首步包含 CUDA allocator
和首轮 kernel 初始化，耗时约 113 秒；后续稳定在约 53–54 秒/step。

## Loss 曲线变化

Total loss 从 9.9471 下降到 2.4735，总降幅约 75.1%。最低值为 2.4643，
出现在 step 92。分段均值和总体波动如下：

| Step 区间 | 平均 loss | 标准差 |
| ---: | ---: | ---: |
| 1–10 | 6.8723 | 1.8238 |
| 11–20 | 3.8075 | 0.2960 |
| 21–40 | 2.9475 | 0.1601 |
| 41–60 | 2.6665 | 0.0632 |
| 61–80 | 2.5653 | 0.0583 |
| 81–100 | 2.5106 | 0.0519 |
| 91–100 | 2.5038 | 0.0514 |

曲线可以分成三个阶段：

1. step 1–20 快速下降，loss 从 9.9471 降至约 3.4；
2. step 21–45 继续下降，但下降速度明显变慢；
3. step 45 后进入 2.5–2.8 左右的平台区，10-step moving average 仍缓慢下降，
   未观察到发散或持续反弹。

各 loss 分量首尾值：

| Loss 分量 | Step 1 | Step 100 | 变化 |
| --- | ---: | ---: | ---: |
| Total loss | 9.9471 | 2.4735 | -75.1% |
| CE loss | 70.1165 | 6.5101 | -90.7% |
| L1 loss | 1.9938 | 1.9964 | 基本不变 |
| Confidence loss | 0.8684 | 0.0162 | -98.1% |
| Gradient norm | 190.0000 | 0.8555 | 显著下降 |

Total loss 的下降主要来自 CE loss 和 confidence loss。L1 loss 始终约为 2.0，
这是后期 total loss 形成平台的主要原因之一；后续如需继续优化 DSpark 收敛，
应重点检查 L1 目标、归一化方式和 `l1_loss_alpha`，而不是单纯增加 steps。

## 产物与 checkpoint 验证

本地实验产物位于：

```text
output/deepseek_v4_flash_dspark_fsdp2_128k_100step_benchmark/
```

其中：

- `benchmark/train.log`：完整训练日志；
- `benchmark/loss_metrics_100steps.csv`：100 个 step 的 loss 与分量原始值；
- `benchmark/loss_curve_100steps.png`：raw loss、10-step moving average 和分量曲线；
- `tensorboard/`：TensorBoard event 文件。

实验结束后递归检查输出目录：不存在 `.safetensors`、`.pt` 或 `.bin` 权重
文件，checkpoint 文件数为 0。空 checkpoint 目录可能由配置初始化阶段创建，
但其中没有保存 model、optimizer 或训练状态。

## 结论

当前 DSpark FSDP2 配置可以在单机 8 × B300 上稳定执行 128K 在线 target
训练。100-step 纯训练时间约 90.2 分钟，稳态约 53.5 秒/step。Loss 在前 40
steps 快速下降，之后逐渐进入约 2.5 的平台区；训练过程稳定，但 L1 分量没有
表现出学习趋势，值得在更长训练前单独排查。
