# B 机 Mooncake 特征池扩容与对照方案

日期：2026-09-17。状态：**方案已整理，扩池配置尚未进行真实训练验证**。
用户当前只有一台机器；本次仅编写方案和检查命令，等第二台机器到位后再调试。
用户随后确认：申请的 A、B 两台机器**每台均为 2048 GiB（2 TiB）CPU RAM**。
这是已确认的资源规格；运行时仍需核对实际可见内存、cgroup 限额和剩余可用内存。

目标是判断扩大 B 机缓冲能否减少训练等待、提高持续吞吐。先评估
**128 GiB 池 / window 12**，通过对照后再决定是否作为默认配置。
已有启动参数支持扩池，本阶段无需先修改训练 Python 或脚本默认值。

## 1. 固定实验条件

沿用已验收的整机分离布局：

```text
A：8 GPU，vLLM DP2 × TP4
        ↓ TCP 写入
B：Mooncake CPU 特征池 → 本机读取 → 8 GPU，TorchTitan DP2 × TP4 / GAS2
```

- A 机继续负责生产暂存，B 机提供唯一特征池；Master 仍随 driver 在 A 启动。
- 每次更新四个样本，128K 实际长度为 131072；模型、层选择、BF16 特征、输入顺序、
  训练目标和 checkpoint 路径语义沿用现有配置。
- 保留有限值检查、完整 SHA256、单副本/hard pin、四个所属 TP rank 读取确认后释放。
- 沿用 TCP/CPU、相同预取深度；RDMA、A 机供池、多节点池、读取方式优化另行评估。
- 本轮只改变池容量和窗口。每次更改都启动新一轮作业，不对运行中的池热扩容。

单机阶段可以检查文档、源码和 dry-run；不要用同机生产/消费的测试结果替代本方案的跨机验收。

## 2. 容量与实验矩阵

历史 128K 每样本描述符总字节数为 `8,055,160,848`，约 **7.502 GiB**。
实际调试时以新一轮 `pipeline.json` 的 `samples[*].nbytes` 为准。
对象额度为池的 75%；样本数量还受 window 限制。

| 编号 | B 池 `--pool-gib` | 对象额度 | `--window` | 按当前样本大小最多可预留 | 实验目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| P0 | 64 GiB | 48 GiB | 8 | 6 个 | 在新机器上建立基线 |
| P1 | 128 GiB | 96 GiB | 8 | 8 个 | 单独评估容量变化 |
| P2 | 128 GiB | 96 GiB | 12 | 12 个 | 在 P1 基础上增加窗口，首选候选 |
| P3 | 256 GiB | 192 GiB | 16 | 16 个 | 可选；P2 有收益且仍受缓冲限制时再试 |

以上是静态最大可预留数量，不是保证始终驻留或同时推理的数量；高低水位和更新组准入也会影响调度。
`window` 不等于 GAS，调整窗口不改变每次更新的样本数。

先只跑 P0、P1、P2，逐轮检查后再进入下一轮，不自动连续启动全部实验。
P3 同时改变容量和窗口，只用于探索；如需分别归因，再补 256 GiB/window 12 的中间对照。

## 3. 扩池前核对两端内存预算

扩大 B 池不只影响 B：增加 window 会提高 A 机的生产暂存上界。
按当前 `memory.py::feature_budget`、两端 DP2、TP4、GAS2 和上述样本大小计算：

| 配置 | A 特征相关估算上界 | B 特征相关估算上界 |
| --- | ---: | ---: |
| P0：64/8 | 约 361.13 GiB | 约 305.33 GiB |
| P1：128/8 | 约 361.13 GiB | 约 369.33 GiB |
| P2：128/12 | 约 541.17 GiB | 约 369.33 GiB |
| P3：256/16 | 约 721.22 GiB | 约 497.33 GiB |

这是代码用于准入的保守上界，**不是预计 RSS 或实测峰值**；消费者的读取/预取上界在当前
公式中取决于读者数和 GAS，不随全局 window 增加。上界之外，代码还要求留出 64 GiB 余量，
并同时受物理内存 80%、cgroup 上限 80% 和实时可用内存约束。

按 A、B 各 2048 GiB 规划，128/256/512 GiB 的 B 池分别占 B 机 RAM 的 6.25%/12.5%/25%。
若 cgroup 也提供完整额度，单节点 80% 的预算上限为 1638.4 GiB，仍需同时满足实时余量约束。
P2 和 P3 的上述估算均低于该上限，容量条件支持逐级对照，无需因申请内存大小限制而停在 64 GiB。

作为后续容量参考，512 GiB/window 32 按当前公式得到 A 约 1441.41 GiB、B 约 753.33 GiB
的特征相关上界；加上 64 GiB 预留，启动时分别至少需要约 1505.41/817.33 GiB 的 headroom。
这只是预算计算，不是实测可用性或性能结论；该配置不加入初始实验矩阵，是否测试由前几档结果决定。

机器到位后重新核对两端物理内存、cgroup、空闲余量和 GPU 占用，不能沿用历史机器的空闲状态。
若 P2 因 A 机预算不足被拒绝，先停在 P1，记录原因；不要通过移除预算检查强行启动。

本次检查时 `/tmp/deepspec_vllm_torchtitan_envs/bin/python` 不存在。
调试前需在两台机器恢复固定环境并核对依赖、共享源码、模型和输入路径；继续不使用 uv。
环境恢复方式应按实际可用安装包确定，不直接假定旧归档与验收版本一致。

## 4. 机器到位后的执行顺序

1. 按 [训练操作指南](TRAINING_GUIDE.md) 建立或复用 Ray：A Head、B Worker。
   使用新机器的 Ray IP；A/B 必须是不同物理节点，两个 DP 参数均为 2。
2. 新环境先做既有 **4K/3 步、4 GiB/window 8** 冒烟检查，确认模型、连接与清理正常。
3. 做 **128K/3 步、128 GiB/window 12** 扩池短测，核对实际容量、校验、checkpoint 和清理。
   只有 12 个样本的短测不能证明 window 12 的持续吞吐收益。
4. 在同一批机器、同一份冻结代码和同一输入计划下，依次做 **P0 → P1 → P2，各 20 步**。
   每轮需要 80 个不同的有效样本、一个 epoch；不得用重复扩充数据凑满 80 条。
5. 若 P2 看起来有收益，再重跑 P0 和 P2 复核。收益不稳定则补测，不能直接采用历史短测比值。
   只有 P2 仍频繁因容量/窗口等待且训练还在等特征，才考虑 P3。

历史整机分离全十六卡只完成 3 步验证；历史十二卡混布 TCP20 不能替代这次 P0。
每轮完成 checkpoint、对象释放和本轮进程清理后再进入下一轮，Ray 可以保留。

## 5. 命令模板

以下模板在项目根目录、A 机训练终端使用。Ray 启动方式见操作指南。
环境变量替换为新机器地址后再执行真实作业：

```bash
cd /mnt/afs-agentpro/lezewei/DeepSpec
export PRODUCER_NODE="替换为_A_机_Ray_IP"
export CONSUMER_NODE="替换为_B_机_Ray_IP"
export RAY_HEAD_ADDRESS="${PRODUCER_NODE}:26379"
```

单机当前只可预览，例如候选 P2 的 20 步命令；这个模式不调用训练 Python、不建输出目录、不启动服务：

```bash
DRY_RUN=true bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train_multinode.sh \
  --producer-dp 2 --consumer-dp 2 \
  --context-length 131072 --steps 20 --pool-gib 128 --window 12 \
  --protocol tcp --receive-device cpu --timeout-seconds 7200
```

两台机器就绪后，每轮单独设置下面四个变量，再执行同一模板。
扩池短测使用 `POOL_STEPS=3`；正式对照按矩阵使用 20 步。
默认模型/输入路径继承现有多机脚本；需要覆盖时，所有对照轮使用相同的 `--model`、`--source`。

```bash
# 以下为 P2 正式对照示例；按矩阵逐轮修改，上一轮验收后再运行下一轮。
POOL_GIB=128
POOL_WINDOW=12
POOL_STEPS=20
POOL_TAG=p2

: "${PRODUCER_NODE:?先设置 A 机 Ray IP}"
: "${CONSUMER_NODE:?先设置 B 机 Ray IP}"
: "${RAY_HEAD_ADDRESS:?先设置 Ray Head 地址}"
set -o pipefail
POOL_RUN_DIR="outputs/dspark_bpool_${POOL_TAG}_${POOL_GIB}g_w${POOL_WINDOW}_s${POOL_STEPS}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p outputs/launch_logs

bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train_multinode.sh \
  --producer-dp 2 --consumer-dp 2 \
  --context-length 131072 --steps "${POOL_STEPS}" \
  --pool-gib "${POOL_GIB}" --window "${POOL_WINDOW}" \
  --protocol tcp --receive-device cpu --timeout-seconds 7200 \
  --output "${POOL_RUN_DIR}" \
  2>&1 | tee "outputs/launch_logs/${POOL_RUN_DIR##*/}.log"
POOL_PIPESTATUS=("${PIPESTATUS[@]}")
printf 'Training exit: %s; tee exit: %s\n' "${POOL_PIPESTATUS[0]}" "${POOL_PIPESTATUS[1]}"
```

不要预先创建 `POOL_RUN_DIR`。训练和 tee 退出码都应为 0。
Mooncake Master 和 B 池由训练自动启动，无需手动运行 `start_mooncake.sh`。
`--prepare-only` 会实际准备数据，不是 dry-run；正常训练自带传输探针，无需每轮重复单跑。

## 6. 每轮验收和比较指标

先通过正确性验收，再看性能：

- 3 步短测：12 样本、48 次 rank 读取确认、八 rank 各 3 次更新、native cursor 6、fc optimizer step 3。
- 20 步对照：80 样本、320 次 rank 读取确认、八 rank 各 20 次更新、native cursor 40、fc optimizer step 20。
- 所有实际长度为 131072，样本及 input identity 顺序在各轮一致；所属四 TP rank 校验并 ACK 后才释放。
- 首次 context 梯度检查八 rank 全部通过，loss/梯度有限；对象全部释放、remaining=0。
- checkpoint metadata/hash、八个 DCP 分片引用范围及独立读取的 optimizer step 通过。
- 无内存压力、无本轮 GPU/模型进程残留；记录源码 hash、配置和两节点内存趋势。

现有 `outputs/dspark_full16_20260916/verify_run.py` 会将输入计划硬编码对比历史 **12 样本**短测，
不能直接用于新的 80 样本长测。调试前应另建参数化核验工具，显式接受对应的基准计划/运行目录，
按实际 steps 检查游标、更新、读取、checkpoint 和清理；不能仅删除样本一致性断言。
本次未修改该历史核验脚本，也未实现新工具。

性能比较使用同一个 B 机 FeatureBuffer 的事件时钟，避免直接相减两台机器的本地时钟。
令 `t(k)` 为第 k 步八个 rank 的 `optimizer_update_complete` 事件时间最大值。
20 步初筛剔除前三步，以第 4–20 步为一致的比较窗口：

- 更新间隔：`t(k)-t(k-1)`，k=4…20，报告中位数、均值和范围。
- 样本吞吐：`4 × 17 / (t(20)-t(3))`；实际长度统一为 131072 时再换算 token/s。
- 生产背压：统计此窗口中的预留等待；跨窗口等待截取交集。事件次数不能代替等待时长。
- 训练等特征：各 rank 的 `gpu_ready.feature_wait_seconds` 分布，并按同一窗口比较。
  不将多个并发 rank 的耗时相加当作全局墙钟时间。
- 缓冲和内存：预留字节峰值、样本积压、两端 RSS/匿名页趋势、最低 headroom、memory pressure。
- put/get、SHA256、转换耗时作为定位依据；这些主机调用时间不是纯网络带宽。

首次推理到消费者结束的总时长作为辅助指标，单列 checkpoint/退出影响。
历史 3 步的 64.60/66.00 秒更新间隔不能当作本轮 20 步稳态基线。

## 7. 如何选择最终配置

- **吞吐稳定提高、训练等待减少、内存安全**：采用其中较小的有效配置。
- **A 等待减少，但训练速度不变**：说明主要增加了积压，不能据此宣称训练加速。
- **窗口越大，put/get 或更新越慢**：检查 Store/内存带宽竞争，回到较小窗口。
- **P1 与 P2 相近**：优先 P1；若 P0 也相近，保留 P0，停止扩大缓冲。
- **P2 仍有明确缓冲限制且重复测量显示收益**：再测 P3。

建议将重复对照约 5% 以上的改善作为初步采用参考；这不是统计显著性保证。
差异较小时优先较小配置，必要时使用更多真实样本延长测试。
最终默认值和操作指南只在实际验收后更新；回退始终使用 `--pool-gib 64 --window 8` 的新一轮作业。

参考：[当前容量/节点预算](../../../deepspec/pipeline/memory.py)、
[生产准入与释放](../../../deepspec/pipeline/buffer.py)、
[历史整机分离 128K 核验](../../../outputs/dspark_two_node_20260916_128k_full16_separated1/verification-status.json)。
