# DSpark / TorchTitan 实施前基线

日期：2026-09-14。对应本地票 `01-baseline-training-entry`。

本记录覆盖实施前快照、环境兼容性核对及真实 Qwen 小规模数值验证入口。
它不是 TorchTitan 接入、阶段卸载或 128K 验收完成的声明。

## 工作树与运行环境

实施开始时分支为 `lzw/feat/qwen3.8_dflash2`，HEAD 为
`932be431ee8be510a6c238fd19bbc24242f22cd7`。工作树已有相关未提交修改，
因此该 HEAD 单独不能复现基线。

在任何实现文件修改前，250 个相关源码、配置、测试、规格文件已归档到
`output/dspark_torchtitan_baseline_20260914/working-tree.tar.gz`。
归档 SHA256：
`c24484a47b7c5111c3b62f147f7b8dc5fab12ed35e255e8cedb6b337e4bb3a04`。
同目录的 `files.sha256.json`、`git-diff.patch`、`git-status.txt`、`head.txt`
记录文件身份和初始 Git 状态。大体积归档留在本地输出目录；迁移到其他机器时须一起传递。

后续实验可通过 `scripts/capture_dspark_baseline.py <新的输出目录>` 捕获新的实际工作树。
工具拒绝覆盖既有目录，避免替换数值参考；应使用同一个训练解释器运行它。

| 项目 | 实际检查结果 |
| --- | --- |
| `env.sh` 选择的解释器 | `/mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python` |
| Python | 3.12.14 |
| PyTorch / CUDA | 2.13.0+cu130 / 13.0 |
| Transformers | 5.16.1 |
| GPU | 单机 8 × NVIDIA B300 SXM6 AC；数值测试使用其中 2 个实际 rank |
| TorchTitan 源码 | `f6b9152e9bedcc18f5dc339b9f88265e5a07e988`，工作树干净 |
| vLLM 源码 | `1ee54c40df7ffe2c8934f5bd1c79917f34cb954e`，工作树干净 |
| vLLM 扩展与 engine | 从本地 `vllm/` 导入 `_C_stable_libtorch` 和 `vllm.v1.engine.core` 成功 |
| TorchTitan 实施前导入 | 失败：`ModuleNotFoundError: No module named 'spmd_types'`；依赖已在下述后续安装中补齐 |

`environment.log` 保存实际导入证据，`packages.json` 保存该环境的已安装版本，
`vllm-binaries.sha256.json` 保存 12 个已有编译扩展的散列，`gpu.csv` 保存 GPU 身份与驱动。
shell 默认 `/usr/local/bin/python` 使用 PyTorch 2.11.0，不能替代上述训练环境。

TorchTitan 的固定 checkout 要求 `spmd_types==0.2.5`，其 AC 模块还导入
实施前缺失的 `torch_remat`、`tyro`。2026-09-14 经用户授权，已在同一训练环境中
新增 `spmd_types 0.2.5`、`torch_remat 0.2.0`、`tyro 1.0.16`、`typeguard 4.6.0`。
`torch_remat` 固定到 checkout 要求的完整 commit
`d302699b1c58f83fa2c7b03bc2593967e9530335`。

安装前重新记录全部 212 个已安装包版本并作为 constraints，确认解析计划只包含上述
四个新包后，使用 `--no-deps` 安装。安装后逐项核对，原有包没有删除或版本变化；
安装前后的 `pip check` 均通过。证据保存在同一基线输出目录下的
`dependency-install-20260914T061119Z-e0wrsjkd/`，包括安装 requirements、constraints、
前后包清单、`package-diff.json`、pip 解析与安装报告及日志。
pytest 和 mypy 仍仅位于 `/tmp/deepspec-validation-tools`，没有安装进训练环境。

安装后 TorchTitan AC 组件、DSpark trainer、现有 vLLM 编译扩展与 engine 的导入均通过。
vLLM 继续从本地 `vllm/` 源码及已有编译产物加载，本次没有安装、重装或编译 vLLM。
使用现有 PyTorch 2.13.0+cu130 / CUDA 13.0 在 B300 上执行小型 MLP 的 SelectiveAC
前向与反向，对照未启用 AC 的同权重模型，输出、输入梯度和全部参数梯度均通过比较。
详情见该安装证据目录的 `smoke.log`；这只验证组件环境可用，不代表 DSpark SAC 接入验收。

## 数值入口

```bash
DRAFT_PYTHON=/mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python
DEEPSPEC_BASELINE_OUTPUT="$PWD/output/new-dspark-baseline" \
OMP_NUM_THREADS=1 "$DRAFT_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m tests.test_dspark_training_baseline
```

输出目录必须尚不存在。本次最终证据目录为
`output/dspark_torchtitan_baseline_20260914/numerics-final/`。
后续回归必须另外读取这份固定参考，不能只比较两条同时修改过的实时执行路径：

```bash
DEEPSPEC_BASELINE_REFERENCE="$PWD/output/dspark_torchtitan_baseline_20260914/numerics-final" \
OMP_NUM_THREADS=1 "$DRAFT_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m tests.test_dspark_training_baseline
```

回归入口加载归档的初始权重、features、CPU/CUDA RNG，并断言输出、梯度、
optimizer/scheduler、日志指标及进度与归档结果一致，用于捕获两条实时路径
共享的 GAS、裁剪或 optimizer 实现回归。它不代表 DCP 新进程训练恢复验收。

测试通过 `Qwen3_8DSparkTrainer.run_batch` 和继承的 `BaseTrainer.train` 执行，
仅用固定 features 替代 target 生产及其启动初始化；没有新建训练循环。
模型为真实 `Qwen3_8DSparkModel` 类的小规模配置：2 层、hidden 64、
4 Q heads / 2 KV heads、128 词表、2 个 anchors、block size 3、16 tokens。
Markov rank 为 8，开启 confidence head；embedding 和 LM head 冻结。

每个 rank 使用 local batch 1，FSDP shard2、TP1、CP1、GAS2，完成两次更新。
先 FP32，再 BF16，通信 reduction 为 FP32。固定种子生成初始参数、features 和
训练 RNG；四个 microbatch 分别包含全有效、稀疏 mask、零分母和尾部边界监督。
目标为 CE × 0.1 + 分布 L1 × 0.9 + confidence BCE，位置衰减 gamma=4，
分母 epsilon=1e-6，confidence target detach。

独立参考在相同训练循环中按每个 microbatch 的全局分母计算 loss，
因此比较中没有替换 optimizer、FSDP 或梯度累积。比较模型输出、各 loss、
每个 trainable 参数裁剪后梯度、公开训练日志中的裁剪前 norm、参数、
FP32 master weights、Adam moments/计数、scheduler、RNG 及最终进度。
冻结权重要求逐元素完全一致，并检查所有 trainable 参数参加反向。
CE/L1/confidence 分项从实际 TensorBoard 输出读取，并与按日志分母重建的
参考值分别比较；日志的整窗口 token 加权统计与训练的 microbatch 等权目标不同。

预设浮点容差为 rtol=1e-4、atol=1e-6：此处两条路径仅改变最终标量 loss
的组合顺序，使用同一模型、FSDP、精度与核，不采用跨不同并行实现的 BF16 宽容差。
该容差不能自动推广到后续 SAC、TP、CP 或 PP。

`numerics-final/` 下每个精度/rank 的 `.pt` 保存初始权重、输入、RNG、配置和完整观测，
配套 `.json` 保存实际 rank、精度、分母、容差及生成该文件的测试源码 SHA256。
此前 `numerics/`、`numerics-nonzero/` 目录为开发过程记录，不是后续回归的参考。

## 验证状态

- 最终 FP32/BF16 两卡捕获通过（每个 rank 的测试约 137.8 秒），
  日志为 `numerics-final.log`。两种精度的全局分母均为
  `[9.541325569152832, 5.557601451873779, 0.0, 2.0]`。
- 另起两个实际 rank，读取固定归档的初始权重、features、RNG 和预期更新，
  完成 FP32/BF16 重放对照，通过（每个 rank 的测试约 67.0 秒），
  日志为 `numerics-replay.log`。这些时长为测试耗时，不是 draft 性能验收数据。
- `numerics-final.sha256.json` 记录所有最终数值产物的 SHA256；四份精度/rank
  清单中的测试源码散列均与提交的测试文件一致。
- mypy（包括未标注函数体）、Ruff：通过新增的两个 Python 文件。
- 快照检查：逐项验证归档内容与 SHA256 清单一致，确认既有目录不能被覆盖。
- 全量 `tests/`：249 passed、27 skipped、3 failed（35 subtests passed），
  耗时 431.75 秒；使用同一训练解释器、`CUDA_VISIBLE_DEVICES=2` 单进程运行。
  多 rank 的新基线另由上面的真实两卡命令验证，不能把 suite 中的 skip 计作多卡通过。
  失败项位于原有 GLM 文件：两个配置断言期待 `[2, 22, 42]`，而当前基线为
  `[40, 41, 42]`；另一个 launcher 断言期待 preflight 诊断文案，实际输出
  node-log 创建诊断。详细栈见 `full-suite.log`。250 个实施前文件的 SHA256
  复核全部一致；本次没有修改这些配置、测试或 launcher。

## Standards

独立 Standards review：没有文档标准违规或可操作的代码异味发现。
评审使用初始 HEAD `932be431ee8be510a6c238fd19bbc24242f22cd7` 与本次三个新增文件
的暂存 diff，排除了原有未提交工作。依据为 CONTEXT、ADR-0001/0002/0003 及
code-review skill 的 smell baseline。

## Spec

独立 Spec review 初次指出三项问题：两条实时路径共享 GAS/optimizer 可能掩盖回归，
没有单独观察 production loss 分项，以及文档引用了早期 fixture 目录。
已分别补充固定归档结果比较、实际 TensorBoard 分项校验和最终目录/源码散列记录。
复审无剩余源码级发现；运行证据状态以上面的最终捕获和重放结果为准。

最终源码审查结果：Standards 0 项，Spec 0 项。

本次完成基线验证入口与证据归档，没有修改生产训练实现；当前不将同一 FSDP
路径内的 loss 参考对照当作新组件验收。票 02–23 尚待实施。
真实 5 层、128K、八卡候选 D、DCP/卸载/恢复、性能统计及后续并行矩阵仍待实施验收。
