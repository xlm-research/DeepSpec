# DeepSeek-V4 DSpark FSDP2 通信/计算重叠优化报告

实验日期：2026-08-25 至 2026-08-26

实验主机：dev-73915620-0

代码分支：lzw/feat/qwen3.8_dflash2，审计时 HEAD 78778be

范围：单机 8 × NVIDIA B300 SXM6 AC，DeepSeek-V4-Flash DSpark draft 训练路径

目标脚本：scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh

## 1. 结论摘要

速度优先的 draft FSDP2 参数选择如下：

| 参数 | 原始基线 | 最终 draft 配置 |
| --- | --- | --- |
| reshard_after_forward | true | false |
| forward_prefetch | false | true |
| backward_prefetch | false | true |
| prefetch_depth | 1 | 2 |
| reduce_dtype | fp32 | bf16 |
| fsdp_wrap_granularity | block | block |
| last_backward_hint | false | false，实验候选已拒绝 |

在与生产拓扑一致的 accumulation=64、8 GPU、seq=131072 draft-only 基准中，最终配置相对原始基线：

- step 中位数从 12,400.632 ms 降到 6,411.548 ms，延迟降低 48.297%；
- p95 从 12,467.117 ms 降到 6,465.296 ms，降低 48.141%；
- draft token 吞吐从 18,497.122 提升到 35,775.449 token/s，提升 93.411%；
- PyTorch peak allocated 从 161.091 GiB 降到 119.442 GiB，减少 41.652 GiB；
- PyTorch peak reserved 从 221.871 GiB 降到 180.984 GiB，减少 40.873 GiB。

PyTorch Profiler 的 accumulation=4 时间线显示，完整 draft step 的 FSDP 暴露通信由 164.452 ms 降到 47.876 ms，减少 70.891%；draft backward 的 AllGather 从 12 次降为 0 次，末个 micro-backward 的 FSDP 通信隐藏率从 75.233% 提高到 95.590%。Nsight Systems 独立采集得到 48.246 ms 暴露通信和 64.526% 完整 step 隐藏率，与 PyTorch Profiler 的 47.876 ms 和 63.443% 交叉吻合。

必须同时保留一个重要的端到端结论：指定完整在线-target 脚本虽然成功运行，但单次 2-step 诊断没有验证出总吞吐提升。当前代码控制组耗时 1,719 秒，最终 draft 配置耗时 1,771 秒；优化配置延迟反而增加 3.025%，吞吐降低 2.936%。完整 step 约 14–15 分钟，而独立 draft step 只有 6–12 秒，在线 target 占据绝大部分总时间；预期的 draft 节省被单次完整运行的波动淹没。因此可确认的是 draft 路径和通信暴露优化，不应宣称完整训练已经加速。上线前应按第 12 节重复完整 A/B。

## 2. 测量边界和未修改范围

本任务只优化用户指定的 DSpark draft 路径。在线 target 由 deepspec/modeling/target/online.py 的独立包装和 train.target_parallel 配置管理，overlap 参数只从 train.parallel 进入 draft 的 apply_fsdp2。此次 overlap 工作没有修改 target 包装逻辑或 target 参数。

工作树在任务开始前已经很脏，并且 deepspec/modeling/target/deepseek_v4_cp.py 等文件已有未提交修改。本任务保留这些用户修改，不能把“本次未编辑 target”误写成“整个工作树的 target 与 HEAD 相同”。

draft-only 基准不会构造在线 target。它生成与真实 target hidden states 相同张量契约的合成特征，用于隔离以下路径：draft attention、grouped MoE、loss、backward、FSDP collective、gradient clipping 和 optimizer step。完整脚本运行则包含 dataloader、在线 target forward、draft forward/backward、通信、优化器及日志。

未测量或不应外推的项目：

- 未在真正的 128 GPU / 多节点拓扑运行；用户确认当前资源为单机 8 GPU；
- 没有每 expert 的 routed-token counter，只有 nvidia-smi 利用率采样，不能量化 expert imbalance；
- 完整训练只有每组 2 steps，并包含首步冷启动；
- 当前异步 metrics 在训练结束时一次性打印两个 step，不能从日志拆出可靠的单步稳态时间；
- profiler 下的 CPU step 时间包含 tracing overhead，不用于吞吐结论；
- nvidia-smi memory.used 是外部采样，不等同于 PyTorch allocated/reserved。

## 3. 环境与工作负载

| 项目 | 值 |
| --- | --- |
| GPU | 8 × NVIDIA B300 SXM6 AC |
| 单卡可见显存 | 275,040 MiB |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| NCCL | 2.28.9+cuda13.0 |
| Nsight Systems | 2025.3.1 |
| Sequence length | 131,072 |
| Global / local batch | 64 / 1 |
| Gradient accumulation | 64 |
| Context parallel | CP 8 |
| Draft FSDP shard | 8 |
| Draft EP / TP | 1 / 1 |
| Target EP | 8 |
| Anchors / block size / draft layers | 512 / 7 / 3 |
| Parameter dtype | BF16 |
| Optimizer master weights / moments | FP32 |

完整运行数据：

```text
train_data/spec_o3_coldstartsft.repeat60.deepspec.packed_256k.jsonl
```

目标模型：

```text
/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731
```

## 4. 执行图审计

DeepSeek-V4 DSpark draft 是 3 个按固定顺序执行的 decoder blocks。每个 block 包含 attention 和 DeepseekV4SparseMoeBlock；MoE 主计算使用 grouped_mm。实际 forward 调用 self.model(**kwargs)，所以 FSDP module hooks 在真实执行路径生效。

FSDP2 采用自底向上包装：先包装子 block，再包装 draft root。optimizer 在 fully_shard 将参数转换为 DTensor 后构造。一个 optimizer step 的 64 个 micro-batches 将 loss 除以 64，前 63 个 micro-batches 禁止 gradient sync，最后一个 micro-batch 执行同步；最终 reduction 完成后才进行 gradient clipping 和 optimizer step。

原始基线 reshard_after_forward=true 会在每次 forward 后释放 full parameters，随后 backward 为每层再次 AllGather。在 accumulation=4 的基线时间线中，完整 step 有 16 次 AllGather，其中 12 次位于 backward。最终 reshard_after_forward=false 使 3 个约 24.5 GB gathered draft blocks 在累积窗口内保持 materialized，backward AllGather 降为 0。B300 的显存预算允许这种交换。

## 5. 实现方法

### 5.1 保持 draft 参数跨累积窗口 materialized

对 draft blocks 和 root 继续使用 FSDP2 full shard，但将 DSpark 配置的 reshard_after_forward 设为 false。gradient accumulation 的非最终 micro-batch 同时关闭 gradient sync 和 backward reshard，最终 micro-batch 恢复同步。这是最大收益来源：消除每个累积窗口中重复的 backward AllGather。

### 5.2 静态 prefetch 顺序

DSpark decoder 顺序静态可知。包装过程返回真实 forward/backward module order，并通过 FSDPModule.set_modules_to_forward_prefetch 和 set_modules_to_backward_prefetch 安装深度为 2 的 schedule。搜索显示 prefetch 的额外收益很小但可复现：在 accumulation=64、last-backward-hint=false 下，median 比无 prefetch 再快 0.710%，代价是 peak reserved 增加 12.25 GiB。

### 5.3 BF16 gradient reduction，FP32 optimizer state

FSDP MixedPrecisionPolicy 的 reduce_dtype 从 FP32 改为 DSpark 显式选择的 BF16，减少 ReduceScatter / replicated-gradient reduction 字节数。MasterWeightAdamW 的 master_param、exp_avg、exp_avg_sq 和 step 继续保持 FP32。共享 ParallelConfig 默认仍为 fp32，避免把 DSpark 的数值/带宽选择扩散到其他 draft 路径。

### 5.4 block 粒度优于 block_components

额外测试了将 attention 和 MLP 分别 fully_shard 的 block_components 粒度。它降低了一部分 reserved memory，但增加 hook/collective 调度并使 median 从 933.676 ms 变成 1,005.923 ms，相对 block 慢 7.738%，因此拒绝。

### 5.5 避免 gradient clipping 的 host sync

clip_grad_norm_ 将 clipping coefficient 留在 device 上，并直接缩放 DTensor local shard，避免 total_norm.item() 把最终 collective 与后续 FP32 optimizer kernels 串行化。此修改同时存在于本报告所有当前代码 A/B 中，未单独归因到参数矩阵的 48.297% 提升。

### 5.6 拒绝 last-backward hint

测试了 FSDPModule.set_is_last_backward 的 accumulation hint。它保持 loss、grad norm 和参数 checksum 不变，但 accumulation=64 median 从 6,457.400 ms 退化到 6,546.301 ms，延迟增加 1.377%、吞吐降低 1.358%，所以生产默认关闭。实验开关仅保留在 benchmark 以便复现拒绝结论。

## 6. 参数搜索结果

第一阶段固定 8 GPU、seq=131072、accumulation=8、warmup=3、measured=9。每个 step 使用 8 rank 最大 CUDA Event 时间；warmup 排除在统计外。

| Variant | reshard | F/B prefetch | depth | reduce | wrap | hint | median ms | p95 ms | median token/s | allocated GiB | reserved GiB |
| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | true | off/off | 1 | fp32 | block | false | 1,746.698 | 1,753.650 | 16,414.974 | 161.091 | 221.871 |
| core_no_prefetch | false | off/off | 1 | bf16 | block | true | 933.676 | 944.920 | 30,708.719 | 119.442 | 168.734 |
| components_no_prefetch | false | off/off | 1 | bf16 | block_components | true | 1,005.923 | 1,017.028 | 28,503.168 | 119.437 | 155.475 |
| forward_prefetch_d1 | false | on/off | 1 | bf16 | block | true | 937.093 | 949.370 | 30,596.763 | 119.442 | 168.734 |
| prefetch_both_d2 | false | on/on | 2 | bf16 | block | true | 928.844 | 945.555 | 30,868.494 | 119.442 | 180.984 |

初步最终候选相对 baseline median 降低 46.823%、吞吐增加 88.051%。由于这一阶段 optimized variants 使用了后来被拒绝的 hint，第二阶段在真实 accumulation=64 且 hint=false 下重新确认。

## 7. 真实 accumulation=64 的最终选择

每组 warmup=2、measured=5。draft_tokens_per_optimizer_step=512 × 7 × 64=229,376。

| Variant | 关键差异 | median ms | p95 ms | stdev ms | median token/s | allocated GiB | reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_accum64_no_hint | reshard=true, FP32 reduce | 12,400.632 | 12,467.117 | 39.547 | 18,497.122 | 161.091 | 221.871 |
| core_accum64_no_hint | reshard=false, BF16, no prefetch | 6,457.400 | 6,472.091 | 25.647 | 35,521.415 | 119.442 | 168.734 |
| core_accum64_with_hint | 上一行加 last-backward hint | 6,546.301 | 6,634.152 | 51.216 | 35,039.025 | 119.442 | 168.734 |
| final_prefetch_d2_no_hint | core 加 F/B prefetch depth=2 | 6,411.548 | 6,465.296 | 34.178 | 35,775.449 | 119.442 | 180.984 |

最终配置相对基线 median 降低 48.297%，p95 降低 48.141%，median throughput 提升 93.411%。相对无 prefetch core，最终配置只有 0.710% median 收益和 0.715% throughput 收益，但 reserved memory 增加 12.25 GiB。因此：

- 速度优先、B300 显存充足：使用最终 prefetch depth=2 配置；
- 更重视显存余量或完整训练未复验：core_no_prefetch 是更保守候选。

## 8. 通信时间线结果

### 8.1 PyTorch Profiler / Kineto

可比 profile 使用 accumulation=4，每个配置采集 8 ranks 的一个额外 optimizer step。下表均为 8 ranks 中位数。

| Phase / metric | Baseline | Optimized | 变化 |
| --- | ---: | ---: | ---: |
| Full-step AllGather count | 16 | 4 | -75.0% |
| Full-step AllGather union | 291.304 ms | 63.728 ms | -78.12% |
| Full-step ReduceScatter union | 111.709 ms | 68.815 ms | -38.40% |
| Full-step FSDP comm union | 404.059 ms | 133.386 ms | -66.99% |
| Full-step FSDP overlap | 239.853 ms | 83.982 ms | 绝对量随总通信减少 |
| Full-step hidden percent | 59.303% | 63.443% | +4.140 pp |
| Full-step exposed FSDP comm | 164.452 ms | 47.876 ms | -70.891% |
| Full-step exposed ReduceScatter | 37.626 ms | 20.225 ms | -46.25% |
| Draft-backward AllGather count | 12 | 0 | 完全消除 |
| Draft-backward exposed FSDP comm | 93.722 ms | 1.750 ms | -98.13% |
| Final-micro backward hidden percent | 75.233% | 95.590% | +20.357 pp |

剩余最大的 FSDP tail 主要在 forward AllGather 和最终 ReduceScatter：optimized draft forward exposed 28.319 ms，full-step exposed ReduceScatter 20.225 ms。由于 backward AllGather 已消失，继续扩大 backward prefetch 不再是最高优先级。

### 8.2 Nsight Systems 交叉验证

Nsight 只采集最终配置的一个已预热 accumulation=4 optimizer step。8 ranks 中位数：

| 指标 | Nsight | PyTorch Profiler optimized |
| --- | ---: | ---: |
| AllGather count | 4 | 4 |
| AllGather union | 61.711 ms | 63.728 ms |
| ReduceScatter count | 4 | 4 |
| ReduceScatter union | 73.738 ms | 68.815 ms |
| FSDP comm union | 135.582 ms | 133.386 ms |
| Compute overlap | 87.318 ms | 83.982 ms |
| Hidden percent | 64.526% | 63.443% |
| Exposed FSDP comm | 48.246 ms | 47.876 ms |
| grouped_mm union | 101.540 ms | 101.160 ms |
| grouped_mm / FSDP overlap | 27.697 ms | 27.671 ms |

两种 profiler 的暴露通信只差 0.370 ms，证明 interval 分类和重叠结论不是单一工具的假象。Nsight CUDA kernel 汇总还记录了总计 32 个 BF16 ReduceScatter 和 32 个 AllGather kernel，即每 rank 各 4 个。

## 9. 数值正确性

BF16 reduction 与 FP32 baseline 不是 bitwise identical，这是预期的 reduction rounding 差异。真实 accumulation=64 的 5 measured steps 中：

| 指标 | 结果 |
| --- | ---: |
| 最大绝对 loss 差 | 0.0003011 |
| 最大绝对 grad-norm 差 | 0.0625 |
| 两组 initial parameter checksum | 完全相同，38676.34403485969 |
| Baseline final checksum | 38682.46319788352 |
| Optimized final checksum | 38684.13907878431 |
| 参数是否发生更新 | 是 |

相同 BF16 reduction policy 下，prefetch、wrap 和 last-backward hint ablations 的 loss、grad norm 和 final checksum 完全一致。单元/分布式测试验证了：

- BF16 reduced gradients 有限且 dtype 为 BF16；
- optimizer 的 master_param、exp_avg、exp_avg_sq 和 step 保持 FP32；
- FSDP2 forward/backward/update 与非分片基准匹配；
- gradient accumulation 和可选 hint 不改变 update；
- DSpark / DFlash2 draft 基础测试通过。

测试命令：

```bash
python3 -m unittest -v tests.test_optimizer_precision tests.test_deepseek_v4_drafts
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --module unittest -v tests.test_fsdp_numerics
```

结果：第一组 7 tests passed；第二组 3 tests 在两个 rank 上均通过。仅有 DeviceMesh 弃用和 SWIG 类型警告。

## 10. 指定完整脚本执行结果

### 10.1 最终 draft 配置

执行命令：

```bash
OUTPUT_ROOT=/mnt/afs-agentpro/lezewei/DeepSpec/output/deepseek_v4_flash_dspark_fsdp2_overlap_final_8gpu \
MAX_TRAIN_STEPS=2 SAVE_CHECKPOINTS=false \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29671 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/usr/bin/bash scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh
```

结果：exit code 0，launcher diagnosis 为 completed successfully，无 OOM、NCCL timeout 或 deadlock，checkpoint 明确关闭。

| 项目 | 结果 |
| --- | --- |
| Training from scratch | 2026-08-25 22:40:35 |
| Running training | 2026-08-25 22:41:07 |
| 两步完成 | 2026-08-25 23:10:38 |
| 训练计时 | 1,771 秒 |
| 平均每 step | 885.5 秒 |
| 聚合输入 token/s | 9,473.301 |
| Loss step 1 / 2 | 10.4417 / 7.0418 |
| Launcher exit | 2026-08-25 23:11:06，code 0 |
| nvidia-smi 最大采样 | 189,171 MiB，约 184.737 GiB |

日志：output/deepseek_v4_flash_dspark_fsdp2_overlap_final_8gpu/logs/node_rank_0.log

### 10.2 当前代码原始参数控制组

控制组只覆盖六个受测 FSDP 参数，其余代码、模型、数据、seed、拓扑和 2-step schedule 相同：

```bash
OUTPUT_ROOT=/mnt/afs-agentpro/lezewei/DeepSpec/output/deepseek_v4_flash_dspark_fsdp2_overlap_baseline_current_8gpu \
MAX_TRAIN_STEPS=2 SAVE_CHECKPOINTS=false \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29672 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RESHARD_AFTER_FORWARD=true \
FSDP_FORWARD_PREFETCH=false FSDP_BACKWARD_PREFETCH=false \
FSDP_PREFETCH_DEPTH=1 FSDP_REDUCE_DTYPE=fp32 \
FSDP_WRAP_GRANULARITY=block \
/usr/bin/bash scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh
```

| 项目 | 结果 |
| --- | --- |
| Running training | 2026-08-25 23:17:46 |
| 两步完成 | 2026-08-25 23:46:25 |
| 训练计时 | 1,719 秒 |
| 平均每 step | 859.5 秒 |
| 聚合输入 token/s | 9,759.870 |
| Loss step 1 / 2 | 10.4395 / 7.0407 |
| Launcher exit | 2026-08-25 23:47:00，code 0 |
| nvidia-smi 最大采样 | 236,047 MiB，约 230.515 GiB |

日志：output/deepseek_v4_flash_dspark_fsdp2_overlap_baseline_current_8gpu/logs/node_rank_0.log

### 10.3 端到端解释

优化组相对控制组 wall time 多 52 秒，延迟增加 3.025%，吞吐降低 2.936%；因此当前证据不支持“完整训练加速”的结论。优化组最大卡 nvidia-smi memory.used 少 46,876 MiB，减少 19.859%，显存收益明显。

独立 microbenchmark 估计 draft 从每 step 约 12.4 秒降到 6.4 秒，只占完整 859–886 秒 step 的约 1% 左右。若其他阶段完全不变，Amdahl 估计的端到端上限仅约 0.7%，小于这次 3% 的运行间波动。该估计使用合成 target features，必须标注为估算，不是完整 profiler 的实测 phase breakdown。

## 11. 代码和产物索引

主要实现：

- deepspec/distributed/config.py：overlap、reduction dtype、wrap 粒度配置及校验；
- deepspec/distributed/fsdp.py：BF16 reduction policy、静态 prefetch、block/component 包装、累积同步和 device-side clipping；
- config/dspark/dspark_deepseek_v4.py：DSpark 速度优先默认参数；
- scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh：环境变量、拓扑检查、profile 和最终参数透传；
- scripts/fsdp/benchmark_deepseek_v4_dspark_draft_overlap.py：draft-only 多次 CUDA Event 基准及可选 PyTorch/Nsight capture；
- scripts/fsdp/analyze_draft_overlap_trace.py：Kineto CUDA/NCCL interval 分析；
- scripts/fsdp/analyze_draft_overlap_nsys.py：Nsight SQLite interval 分析；
- tests/test_fsdp_numerics.py 和 tests/test_optimizer_precision.py：数值与 optimizer state 测试。

关键产物：

- output/draft_overlap_search_v2/baseline_accum64_no_hint/result.json；
- output/draft_overlap_search_v2/core_accum64_no_hint/result.json；
- output/draft_overlap_search_v2/prefetch_d2_accum64_no_hint/result.json；
- output/draft_overlap_ab/baseline/trace_summary.json；
- output/draft_overlap_ab/optimized_block/trace_summary.json；
- output/draft_overlap_nsys/final_d2/draft_overlap_final.nsys-rep；
- output/draft_overlap_nsys/final_d2/trace_summary.json；
- 两个完整运行日志目录，见第 10 节。

## 12. 后续续接建议

优先级 1：完整端到端重复 A/B。固定同一空闲节点和时钟条件，baseline 与 final 随机交错，各至少 3 次；或者每组执行完整 6-step epoch。报告 median/p95 和置信区间。只有这一步能判定小于 1% 的理论端到端收益是否存在。

优先级 2：给完整 trainer 增加无 host synchronize 的 CUDA Event phase timers，分别记录 target forward、draft forward、loss、backward、final collective/clip、optimizer，并在 step 末统一读数。这样能把 52 秒运行差异定位到 target、draft 还是系统噪声。

优先级 3：完整脚本比较 core_no_prefetch 与 final_prefetch_d2。prefetch 只有约 0.7% draft 收益，却增加 12.25 GiB reserved；若完整链路没有稳定收益，生产默认应采用 core_no_prefetch，把 prefetch 作为速度实验开关。

优先级 4：继续处理剩余 forward AllGather 和 final ReduceScatter tail。可研究更早的 root/block unshard、适当的 forward order、减小 reduction bucket tail；当前 PyTorch 2.11 本地 FSDP2 API没有独立 ReduceScatter group 或 buffer-depth 接口，不能声称已调过这些参数。

优先级 5：多节点验证。在 128 GPU 环境重新检查 dp_replicate mesh、跨节点 ReduceScatter/AllReduce、NCCL fabric 和 clip tail。本报告的单机 8 GPU 结果不能外推到跨节点。

## 13. 复现命令

真实 accumulation=64 基线和最终 draft 基准的核心差异如下。完整参数可从对应 result.json 和 shell history/本节恢复。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
scripts/fsdp/benchmark_deepseek_v4_dspark_draft_overlap.py \
--sequence-length 131072 --num-anchors 512 --block-size 7 \
--num-draft-layers 3 --accumulation-steps 64 \
--warmup-steps 2 --measure-steps 5 --cp 8 \
--reshard-after-forward --no-forward-prefetch --no-backward-prefetch \
--prefetch-depth 1 --reduce-dtype fp32 --wrap-granularity block \
--no-last-backward-hint \
--output-json output/draft_overlap_search_v2/baseline_accum64_no_hint/result.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
scripts/fsdp/benchmark_deepseek_v4_dspark_draft_overlap.py \
--sequence-length 131072 --num-anchors 512 --block-size 7 \
--num-draft-layers 3 --accumulation-steps 64 \
--warmup-steps 2 --measure-steps 5 --cp 8 \
--no-reshard-after-forward --forward-prefetch --backward-prefetch \
--prefetch-depth 2 --reduce-dtype bf16 --wrap-granularity block \
--no-last-backward-hint \
--output-json output/draft_overlap_search_v2/prefetch_d2_accum64_no_hint/result.json
```

Kineto 分析：

```bash
python3 scripts/fsdp/analyze_draft_overlap_trace.py \
output/draft_overlap_ab/optimized_block/torch_profile \
--output-json output/draft_overlap_ab/optimized_block/trace_summary.json
```

Nsight SQLite 分析：

```bash
nsys stats --force-export=true \
output/draft_overlap_nsys/final_d2/draft_overlap_final.nsys-rep
python3 scripts/fsdp/analyze_draft_overlap_nsys.py \
output/draft_overlap_nsys/final_d2/draft_overlap_final.sqlite \
--output-json output/draft_overlap_nsys/final_d2/trace_summary.json
```

## 14. 最终判定

draft FSDP2 优化已完成并有两种 profiler、真实 accumulation、多次 CUDA Event 和数值测试支持。可报告的优化量是 draft median latency -48.297%、draft throughput +93.411%、FSDP exposed communication -70.891%，以及完整运行 sampled memory.used -19.859%。

不可报告为正收益的是完整在线-target wall time：本次严格对照实测为 latency +3.025%、throughput -2.936%。因此最终配置是“经验证的 draft 路径速度最优候选”，不是“已经验证的完整训练最优配置”。后续应从第 12 节的完整重复 A/B 和 phase timer 开始续接。

## 15. 简历表述建议

建议使用下面这种有边界的表述：

“面向 8 × NVIDIA B300、128K sequence 的 DeepSeek-V4 DSpark 训练，完成 PyTorch FSDP2 通信/计算重叠优化与 profiling 基础设施建设；通过跨 gradient-accumulation 保留参数、BF16 gradient reduction 和静态 prefetch，将 draft-only optimizer-step median latency 降低 48.3%、吞吐提升 93.4%、暴露 FSDP 通信降低 70.9%，并用 PyTorch Profiler 与 Nsight Systems 交叉验证。完成完整在线-target 训练和数值正确性回归，同时识别 target 主导下端到端收益尚未显著，设计后续 phase timing 与重复 A/B 方案。”

不建议写“完整训练提速 93%”或“端到端提速 48%”；这些数字只属于隔离后的 draft 路径。若后续重复完整 A/B 得到稳定正收益，再新增端到端指标。
