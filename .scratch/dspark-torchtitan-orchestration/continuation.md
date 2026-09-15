# 自动续接记录

## 2026-09-15 补验完成：优先于下方历史状态

- 本轮独立改动已提交 `937e3c9`（当前分支 `dev/vllm_torchtitan`）：CPU HF修复、
  缺shard回归、失败测试stop_update修正和常驻报告，共4文件。Ruff与最终Pyrefly
  均通过（0 errors）；两轴本地review在 `/tmp/deepspec-implementation-checks/export-review.md`。
  其余共享工作树改动仍由原工作及并行会话保留，未纳入本提交。
- 原生5+5和连续10步均完成；常驻expandable尝试也已完成10步、完整DCP及worker退出。
  常驻总成本1517.764秒，原生分阶段3561.700秒；固定监督/样本/RNG精确一致。
  规模、拓扑、allocator及计时边界见 `doc/benchmarks/dspark_native_128k_h800_resident.md`。
- 完整128K checkpoint的FP32 CPU HF导出验证64个参数、consumer、幂等和DCP哈希。
  两GPU真实保留/恢复、FP32/BF16导出及第4步提交失败保护均通过，日志索引见上述报告。
- 缺失GPU HF shard的真实恢复测试发现旧索引残留；`export.py`现于重新保存前清理
  旧单文件/索引名称。`test_torchtitan_pending_export`的FP32/BF16真实回归均通过；
  无fixture时显式skip，验收运行均提供fixture。保留失败测试显式设置stop_update=4。
- 本轮全量discovery记录307项、20 failures/26 errors/51 skips；其中新测试的
  无fixture错误已修复并单测验证。其余45条失败记录在原HEAD快照复现，涉及旧模型
  路径、GPU前提、缺少pytest及已有GLM配置断言。日志在 `/tmp/deepspec-implementation-checks/`。
- 17:53发现另一CLI会话同时修改本工作树的DP/FSDP、配方和票11验收runner；本会话
  已询问用户由谁继续这些共享编辑/GPU检查，尚未收到答复。继续独立export修复审查，
  DP/FSDP重叠工作等待归属答复；其他会话和其他工作区进程保持运行。
- `/tmp/deepspec-implementation-checks/ticket11.patch`已由另一会话应用，不能重复应用。
  `h800-gqa-dp8-reference`已生成；`h800-replicate8-red.log`实际通过FP32两更新，
  因补丁并发落地，它是green证据，不是red证据。票11仍需核对另一会话的最终验收。
- 本会话未创建持续goal；下方“goal active”及旧PID属于历史。恢复工作时先核对
  当前文件、进程和用户归属答复，已完成的128K训练与导出无需重复运行。

## 2026-09-15 17:07 复核：优先于下方历史运行状态

- 原分阶段运行和连续重放均已完成 10 updates；两者根目录 `complete.json`
  齐全，所有 worker 已退出。不要再启动原编排或重复连续重放。
- 最终 DCP 比较 `h800-continuous-checkpoint-comparison.json`：754 个字段、
  344 个张量精确一致；监督/RNG 的 160 对 forward 记录和 rank0 的 250 对
  训练指标也精确一致。完整结果见 `doc/benchmarks/dspark_native_128k_h800_comparison.md`。
- 任务 10 仍缺旧常驻参考测量和剩余 HF 导出/失败验证，任务 11 尚未实施。
- 旧常驻参考 `qwen38-128k-h800-resident/` 于17:11在第一次L1 loss中CUDA OOM，
  0 updates，worker已退出；`h800-resident-failure.json`保存错误和请求。
  当前新尝试为 `qwen38-128k-h800-resident-expandable/`，日志
  `h800-resident-expandable.log`（均相对下方 run 父目录）。唯一变化为
  `PYTORCH_ALLOC_CONF=expandable_segments:True`，用于排查碎片化。保持原128K输入、
  逻辑DP2/GAS2、物理GPU0、4；先检查进程和日志，不能重复启动或并发GPU测试/大文件IO。
- 本次复核 `python -m unittest tests.test_torchtitan_scale_summary -v`：8项通过。
- 17:31复核：expandable尝试worker为108999、109000，日志已报告第6步；仍在运行。
  该尝试通过了原L1失败点，但每个microbatch仍出现可恢复allocator retry。
  必须等待完整10步、checkpoint和worker退出，不能把单步进展当成性能验收。
- summary及其回归测试补充JSON字典类型注解，8项测试、Ruff、Pyrefly均通过；
  用真实分阶段和连续产物重算后，与已有summary逐值相同。
  `/tmp/deepspec-implementation-checks/`保存检查日志和HEAD代码快照；native trainer
  的类型错误在HEAD快照中也复现，未借本票改写其运行逻辑。
- `/tmp/deepspec-implementation-checks/ticket11.patch`是尚未应用的票11候选。
  它基于既有dense-candidate，额外拒绝replicate与shard>1或TP>1组合；后续HSDP
  仍属票12。补齐票10后，先用匹配DP8参考跑现有phase-entry测试观察red，再应用。

更新时间：2026-09-15 15:24 Asia/Shanghai。后续轮次先读本文，再核对进程和文件。
本文覆盖旧 `active-run-notes.md` 的机器、路径和当前运行状态；旧记录仅作历史证据索引。

## 目标和边界

用户已要求设置自动续接。线程持续目标已创建，状态 active，无指定 token 预算：
完成任务 10（真实 Qwen 128K 两阶段训练、恢复正确性与性能验收），随后完成
任务 11（八卡 DP replicate8、DP shard8 的训练与同拓扑保存退出恢复验证）。
任务 11 完成后汇报；任务 12–30 保留为后续计划。

- 所有依赖安装使用 `python -m pip`，用户明确要求不用 uv。
- 不启动子代理；不停止其他工作区的进程。
- 运行期间保留模型、输入、特征、初始状态和 checkpoint。先确认现有进程，避免重复启动。
- `Training completed` 表示一个草稿阶段完成；以 run 根目录 `complete.json` 判断整个运行完成。

## 当前运行：先检查这里

仓库：`/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm`。

- 原主 PID4130253于14:50失败退出。**当前正式编排 PID21016**，15:07:18从原step5重启，
  15:18:15启动新阶段，worker **25533–25540**，监督器25522。15:24仍在运行。
  `h800-orchestrator.pid` 已更新；新attempt为`attempt-1789456695677558196`。
- 正式worker于15:19:20完成DCP加载（约42.9s）；**15:22:48完成第6步，loss3.72804**，
  与隔离探针相同，已越过原失败位置。正在继续第7–10步。
  rank0发生一次PyTorch缓存分配重试后成功，并非训练退出；NCCL致命OOM未再出现。
  运行时已完成6步，但progress和最新持久checkpoint仍为step5，阶段结束才提交step10。
- Run ID：`native-qwen38-128k-h800-20260915`。
- Run 根目录：`outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800`。
- 主日志：`outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800.log`。
- 请求：同上父目录的 `128k-h800-request.json`。
- `phase-0-5/complete.json` 已存在；step 1–5 loss：
  3.73530、3.73460、3.73396、3.73274、3.73085。
- 第 5 步在 14:05:50 完成；DCP 保存约 14.55 秒，`checkpoints/step-5/commit.json`
  已提交，metadata SHA 匹配；progress 为 completed_updates=5。
- `phase-5-10/target/features` 已有 20 个 .pt（约 161GB）。两个 producer 于 14:21
  完成请求并退出；target 父进程校验字节后于 14:33 提交 `target-result.json` 并退出。
- 第二阶段于 14:45:37 启动八个新 worker（4189312–4189319），14:46:44
  完成 step-5 DCP 加载（约 47s）。但 14:50:26 在第6步梯度裁剪中失败：
  `torch.nn.utils.get_total_norm -> DTensor _NormPartial all_reduce`，
  NCCL `Cuda failure 2 'out of memory'`。rank0 首发，其他 worker 被监督器清理。
  原主进程也已退出，所有 GPU 释放；没有第6步提交，progress 保持5。
- 失败摘要：run 父目录 `h800-phase2-failure.json`，完整堆栈在原主日志 521 行起。
  失败的 `phase-5-10/attempt-1789454737180297105/` 和全部原始产物保持原样。
- 首个探针 `h800-resume-probe-1/`（PID7743、worker7826–7833）在14:58再次复现同样OOM，已退出。
- 第二探针 `h800-resume-probe-2-warm/`（PID14072、worker14138–14145）已通过：
  15:04:39完成真实第6步，loss3.72804；15:04:52保存step6；全部worker和父进程退出。
  从同一个 step5 和原始第二分区启动，只尝试完成第6步，保持真实模型和128K输入；
  checkpoint/result 写入 probe 独立目录，不推进原编排。
- 修复已接入正式 `DSparkTrainer._initialize_parameter_collectives`：提前用零标量初始化
  参数DTensor mesh每个非单例轴的通信组；不改参数、梯度、RNG和training_identity。
  正式恢复仍用普通ScaleTrainer，不带诊断钩子、NCCL_DEBUG或probe环境。
  不要因为校验特征时GPU空闲而并发启动其他GPU测试。
- producer 日志：`phase-{0-5,5-10}/target/producer-{0,1}.json.log`。
  完成后的 vLLM teardown 会出现 SIGTERM / force killing EngineCore 日志，不能单据此判为任务失败。

只读状态命令（先 cd 仓库）：

```bash
ps -p 21016 -o pid,etime,stat,args
ps -p 25533,25537 -o pid,etime,stat,args
cat outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800/progress.json
tail -30 outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800.log
nvidia-smi
```

当前编排正在验证正式修复；若退出，先核对完整错误、checkpoint和complete.json，不能盲目重启。
启动/恢复命令：`bash .scratch/dspark-torchtitan-orchestration/resume-128k-after-gpu-release.sh`。
step-5 已存在时恢复会加载 checkpoint；不会覆盖首次初始化。

## 环境、数据和模型

- 8×H800 80GB；之前旧验收运行是 B300，不继承其完整规模验收结论。
- `source ./env.sh` 激活 `.envs/orchestration`，基于
  `/tmp/deepspec_vllm_torchtitan_envs` 的 system-site-packages venv。
- Python 3.12.14，torch 2.13.0+cu130，Triton 3.7.1，Transformers 5.16.1。
  vLLM 0.26.1rc1.dev719 使用本仓库 `vllm/` 源码和已有 H800 扩展。
- CUDA_HOME 由 env.sh 指向基础环境中的 nvidia/cu13；不要误用系统 CUDA 12.9。
- pip check、8 卡 NCCL/BF16、FlexAttention 前后向、vLLM CUDA RMSNorm 已通过。
  版本、安装约束、日志详见 `environment-setup/README.md`。
- attn-gym 安装基础包 0.0.8，省略与 vLLM tvm-ffi==0.1.11 冲突的 linear extras。
  当前 DSpark 是 full attention；不宣称其他 TorchTitan 模型环境已完整支持。
- 教师（用户指定）：`/mnt/afs_agents/hongjiawei/share_models/Qwen/Qwen3.8-27B`。
  18 个分片和 tokenizer 已验证；env.sh 设置 TARGET_MODEL_PATH，recipe 支持该变量。
- 数据（用户迁入）：`outputs/dspark_torchtitan_orchestration_20260914/128k-source.jsonl`。
  注意目录是 **outputs 带 s**。80 条 packed records，81,557,568 字节。
- native plan 实际选择 40 条，全部 131072 tokens；plan SHA
  `efd6377dd1b18e4f2dadf32161d4ce151ee59933068609434642cd7275ae1987`。
- 原迁入 `128k-request.json` 保留原样；其旧路径和旧 run ID 不用于新运行。

## 实际训练配置

冻结 Qwen3.8-27B 由 vLLM 生产特征；TorchTitan 训练 5 层 DSpark 草稿模型。
5120 hidden、17408 FFN、248320 vocab、24Q/4KV/head256，512 anchors、block7、Markov256。
BF16 参数 / FP32 master+reduce，TP4 × DP shard2，CP1/PP1，SelectiveAC，SP 关闭。
本地 batch1、全局 batch4、GAS2；10 updates 分 5+5。LR 6e-4，保留生产调度
total_steps=1000 / warmup40。本次运行 HF export 关闭。

训练入口：`torchtitan.models.dspark_draft.train`；配置入口：
`tests.torchtitan_scale_fixtures.qwen38_128k_acceptance`，包含初始化和监督状态记录。
核心代码在 `torchtitan/torchtitan/models/dspark_draft/`；DeepSpec orchestration 调度阶段，
`deepspec/trainer/qwen3_8_vllm.py` 和 GLM helper 负责现有教师生产路径。

## 续接顺序与完成条件

1. 跟进当前运行直到 10 updates，检查 step-5 恢复、step-10 提交、两阶段 complete、所有 worker 退出。
   结束条件不是 loss 下降，也不是单个阶段打印 Training completed。
2. 按 `issues/10-qwen-128k-first-acceptance.md` 补齐验收：实际状态连续性、固定特征对照、
   保存/恢复失败保护及资源交接证据，计时分开报告训练、准备/IO、保存、启动恢复和退出。
   复用仍有效的历史证据；旧机器产物未迁入时明确缺失并补测。不要据 10 步完成就关闭任务 10。
3. 真实初始化位于 run 父目录的 `scale-initialization-h800/`：initial-weights.pt 约 9GB，
   rng-rank0..7.pt 齐全。保持不变，用于固定状态重放。
   `tests/run_torchtitan_scale_replay.py` 和 resident benchmark 已改为从原始 plan 的
   `resolved_recipe.capture_initialization` 读取目录并提前检查权重存在，GPU 执行仍待完成。
   对照工具：`tests/compare_torchtitan_checkpoints.py`、`tests/summarize_torchtitan_scale.py`；
   常驻参考工具：`tests/run_torchtitan_resident_benchmark.py`。H800 显存限制需据实验证和记录。
4. 任务 10 通过后阅读 `issues/11-dense-dp-fsdp.md` 和 `next-dense-notes.md`，再处理
   `dense-candidate/candidate.patch`（未应用、未 GPU 验证）。不要把候选当成已交付能力。
   任务 11 要求 DP replicate8 和 shard8，TP/CP/PP1；各自匹配逻辑 DP8/GAS 的参考，
   FP32/BF16 loss、全梯度/master/Adam/scheduler 对照，以及真实 DCP 退出恢复和失败保护。
5. 更新本文、相关 ticket 和 `doc/benchmarks/` 验收证据。任务 11 真正完成后结束持续目标。

按需参考：`ticket-proposal.md`（任务依赖），`integration-followups.md`（HF 导出待测），
`doc/benchmarks/dspark_native_128k.md`（性能与恢复验收契约）。
HF export dtype 修复已在旧记录末尾说明应用过，不能重复盲目套用 hf-export-candidate。

## 本轮完成的验收工具修正与证据

- `doc/benchmarks/dspark_native_128k_h800.md` 保存当前机器的配方、路径、已测结果和待验项。
- `h800-first-phase-timing.json`（run 父目录）：首阶段 draft 准备 684.794s、worker
  生命周期 1105.434s、release 0.155s，总计 1790.383s，排除 target。
  最长 rank 训练 1018.189s、save 14.601s、初始化 29.102s、capture 14.740s。
  **更正**：其中 phase-end allocated=25,043,702,784 bytes 不是全程峰值；native metrics
  每步重置 CUDA 峰值计数器。`h800-first-phase-timing-v2.json` 已明确这些计数器的时间范围。
- `h800-first-phase-memory.json` 从真实 TensorBoard 读取每步重置之前的数据：rank0
  active 峰值61.627GiB、reserved73.271GiB，重试/OOM均0；原运行只记录了 rank0。
  summary 新增 `training_memory_at`，按实际 worker PID 匹配文件并严格要求全部 update。
  replay 使用原生 `--metrics.save-for-all-ranks` 补全八卡指标。仅改变指标配置，不改变数学或
  training_identity；CPU解析/初始化路径预检已通过。8项 summary回归通过，日志
  `h800-summary-memory-regression.log`；全规模重放仍未执行。
- `h800-step5-state-audit.json`：metadata SHA、step5、全局微批次游标10、scheduler epoch5，
  八 rank checkpoint CPU/CUDA RNG 与最后一次 forward 记录完全相等，Python/NumPy RNG 存在。
  此小状态 CPU 检查不代替后续完整 model/master/Adam 等 DCP 比较。
  可复用 `.scratch/dspark-torchtitan-orchestration/environment-setup/audit_checkpoint_boundary.py`
  （参数为 phase complete.json 和输出 report 路径）检查最终 phase。
- 原始 audit 断言误把 forward 钩子里的 dataloader 游标当成已消费序号。native Trainer
  在每个 update 先预取两个微批次，再执行两次 forward，所以八 rank 的原始记录都是
  `2,2,4,4,6,6,8,8,10,10`，forward 共10次；这是预取行为，不是重复消费的证据。
- 已修正 `tests/summarize_torchtitan_scale.py`：严格校验每阶段位置、forward 数量和每次
  预取游标，再以 forward 顺序建立比较序号；原始 pt 文件、监督/RNG tensor 和运行中训练器
  完全保留。六个 `tests.test_torchtitan_scale_summary` CPU 回归测试通过，包含拒绝缺失、
  重复、游标跳跃、phase 缺口和 worker 错配。red/green 日志存于 run 父目录。
  完整运行结束后仍须用真实两阶段产物执行 summary；此时尚未声称全运行验收通过。
- audit 的第一版失败是上述游标误解；v2 是临时脚本错误的 scheduler 键路径；v3 成功。
  DCP 的准确键为 `lr_scheduler.0.last_epoch`。三份日志均保留，不把前两次计作通过。
- `tests.test_torchtitan_hf_export` 新增可选环境变量 `DEEPSPEC_HF_EXPORT_CHECKPOINT`，
  可直接从完整规模 DCP 重建模型权重并检查 CPU HF导出、dtype、旧consumer、幂等和原文件哈希。
  原小规模 forward-observation 模式保留。step5 metadata已证明64个真实模型tensor键齐全、
  meta构造不初始化CUDA；真实完整导出测试尚未运行，等待 GPU性能测量结束以免IO竞争。

## OOM 诊断与已接入修复

按 diagnosing-bugs 排序假设：① NCCL首次建立梯度范数通信时，PyTorch缓存占满空闲显存；
② 恢复留下额外活跃GPU tensor；③ 额外进程占用显存。先在完整128K单update探针中定位。

隔离诊断 fixture `resume_probe.py` 继承原ScaleTrainer，副本已分别存入两个probe目录。
environment-setup 下的活动诊断文件已删除；原生产代码中没有 `[DEBUG-h800-resume]` 插桩。
只记录初始化、恢复后和梯度裁剪前后的 CUDA allocated/reserved/free 及失败点 memory_summary，
启用NCCL_DEBUG=INFO和全rank指标。调试前缀 `[DEBUG-h800-resume]`；结束诊断后不要带入生产路径。
注意 memory() 会 synchronize CUDA，这可能影响异步释放时机；若探针不复现，不能视为修复。

已见 rank0：初始化allocated=7,371,787,776、reserved=7,553,941,504；恢复后
allocated仍为7,371,787,776，reserved=9,607,053,312，driver free=72,565,850,112。
首个探针同样失败，排除了仅增加同步就修复的情况。before_clip rank0：
allocated=25,043,701,248，reserved=79,968,600,064，peak_allocated=66,172,147,712，
driver free=157,286,400（约150MiB）；其余rank free约1.365GB。NCCL日志明确是随后
首次参数mesh TP范数all-reduce连接分配触发OOM；所有rank恢复前后allocated相同。
摘要 `h800-resume-oom-diagnosis.json`，各rank原始jsonl和memory_summary保存在probe1。

第二探针唯一处理变化：在初始化结束时，对所有可训练DTensor参数的device_mesh各非单例轴
去重执行一次零标量 all_reduce，以提前建立已有通信资源。环境
`DEEPSPEC_RESUME_PROBE_WARM_COLLECTIVES=1`；未改变模型、数据、更新公式和checkpoint。
probe2使用自己目录中的fixture副本；完成后同一通信预热逻辑已接入native trainer。
warmup后allocated保持7.371GB，driver free减少约1.06GB，表明通信内存已提前建立；
probe2实际第6步clip、optimizer和DCP均通过；metadata SHA、training_identity不变、
八worker全部退出且GPU池释放均已验证，证据 `h800-resume-warmup-acceptance.json`。
`h800-resume-probe-2-boundary-audit.json` 核对了step6、global cursor12、scheduler6，
八rank checkpoint CPU/CUDA RNG等于最后forward，Python/NumPy RNG存在。
audit 支持 `--observations-dir`，用于探针复用原manifest、观察结果在独立目录的情况；
会先核对该目录draft-result的worker和完整commit，不能任意混用目录。
第一个probe2 audit因按manifest定位观察文件而失败，指定真实观察目录后的v2通过；日志均保留。
原trainer无预热版本已归档至probe1的`native-trainer-before-fix.py`；修复补丁
`h800-resume-warmup-fix.patch` 保存于run父目录。语法/Ruff/diff检查通过。
探针数据不代替完整10update验收；修复后仍须原编排完成、连续重放和所有对照。

原运行根目录 complete.json 存在、主进程退出、八卡确实无人使用后，再依次执行：

```bash
source ./env.sh
export OMP_NUM_THREADS=1
unset TORCHINDUCTOR_COMPILE_THREADS
python -m tests.summarize_torchtitan_scale \
  outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800 \
  outputs/dspark_torchtitan_orchestration_20260914/h800-phased-summary.json
python -m tests.run_torchtitan_scale_replay \
  outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800 \
  outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800-continuous 10
```

重放是长进程，按已有方式单独记录 log/PID 并持久运行；不要同 GPU 任务并发。
完成后使用 `compare_torchtitan_checkpoints.py` 比较两边 step-10，使用 summary 的
`--reference` 比较全部监督和 RNG，然后测 resident，再补齐其他缺失验收。

## 已修改文件与原工作区

本会话修改 `env.sh`、恢复脚本、native config_registry.py 的 TARGET_MODEL_PATH 支持，
以及本页上述验收工具、文档和native trainer通信预热修复；新增本地环境、记录和H800产物。
原 vllm 脏子仓库及其他未跟踪目录要保留。
git 因共享目录 ownership 需用 `git -c safe.directory="$PWD" ...`，无需修改全局配置。
