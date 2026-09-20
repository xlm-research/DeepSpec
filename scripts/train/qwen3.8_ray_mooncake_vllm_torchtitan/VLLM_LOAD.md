# 导出草稿头的 vLLM 加载验证

日期：2026-09-16。固定 Python：`/tmp/deepspec_vllm_torchtitan_envs/bin/python`。

## 结论

**修复一处 vLLM 配置识别问题后，用户导出的草稿头已在 GPU 4–7、TP4、BF16 下
完成原生 vLLM 模型加载、KV cache 分配和 warmup，`LLM(...)` 构造成功返回。**

加载后的参数抽查与两条短生成尚未完成：一次在检查脚本的 RPC 序列化处失败；
修正脚本后的两轮分别因外部 GPU 任务进入、进程收到 SIGTERM 而中断。
最后一次 SIGTERM 的来源尚未定位，不能据此判断模型存在推理错误，也不能报告生成通过。

所有四轮自己启动的子进程均已清理。以下证据明确区分加载成功与生成未验证：

- [独立核验结果](../../../outputs/dspark_vllm_load_20260916_run4/verification-status.json)。
- [加载及 warmup 成功的日志](../../../outputs/dspark_vllm_load_20260916_run2/vllm.log)：草稿架构为 `Qwen3DSparkModel`，四个 rank 均报告加载完成，11:28:43 完成 engine 初始化。
- [初始化之后的检查脚本错误](../../../outputs/dspark_vllm_load_20260916_run2/failure.json)：失败阶段为 `inspect_loaded_draft`，调用点在 `LLM(...)` 成功返回之后。

## 验证对象

- Target：`/mnt/afs_agents/hongjiawei/share_models/Qwen/Qwen3.8-27B`。
- 用户已导出的 draft：`outputs/dspark_pipeline_4plus4_20260916_run4/draft-hf-step-3`。
- 来源：同一运行的 `checkpoints/step-3`，run ID `dspark-94a613ec6136`，3 次优化器更新；导出记录的源 metadata 校验通过。
- HF 文件：64 个 tensor、4 个 safetensors 分片，导出精度 FP32；实测按 BF16 加载。
- 模型：`Qwen3DSparkModel`，5 层，hidden size 5120，`fc.weight` 为 `[5120, 25600]`，特征层 `[1, 16, 31, 46, 61]`，含 Markov head 和 confidence head。

以上数值来自实际导出配置和权重。检查中没有修改导出的配置或权重；配置、索引和
`export.json` 的 SHA256、四个分片的大小分别与测试前记录核对。

## 发现并修复的加载问题

原 vLLM 分支 `lzw/support_dspark_v.0.1.1`，HEAD
`1ee54c40df7ffe2c8934f5bd1c79917f34cb954e`，首次真实 TP4 加载报错：

```text
AttributeError: 'Qwen3_5TextConfig' object has no attribute 'hc_mult'
```

调用链：

```text
导出配置：architectures=[Qwen3DSparkModel], model_type=qwen3_5_text
  → SpeculativeConfig.hf_config_override() 原实现按 model_type 改成 Qwen3_5MTP
  → method=dspark 分支失去 Qwen3DSparkModel 标记，选中 DeepSeek DSpark
  → DeepSeek 构造器访问 Qwen 配置中不存在的 hc_mult，加载失败
```

修复位于 [`vllm/config/speculative.py`](../../../vllm/vllm/config/speculative.py)：
Qwen3.5 MTP 配置映射现在保留显式声明的 `Qwen3DSparkModel`。
这个修改仍在工作区，单独 checkout 上述 HEAD 不包含修复。

回归测试位于
[`test_speculative_draft_hf_overrides.py`](../../../vllm/tests/config/test_speculative_draft_hf_overrides.py)：
验证 DSpark 配置保持不变，以及四种普通 Qwen3.5 配置的 MTP 映射保持原行为。
加入测试后旧实现 **1 failed、10 passed**，修复后 **11 passed**。
真实导出配置的 CPU 探针、Ruff 检查和格式检查也通过。

在项目根目录复查配置回归：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/vllm:$PWD" \
  /tmp/deepspec_vllm_torchtitan_envs/bin/python -m pytest -q \
  --confcutdir=vllm/tests/config \
  vllm/tests/config/test_speculative_draft_hf_overrides.py
```

## 实测参数与范围

| 参数 | 实际设置 |
| --- | --- |
| GPU | 4、5、6、7，native multiprocessing，TP4 |
| Runner | `VLLM_USE_V2_MODEL_RUNNER=1` |
| 推理精度 / 上下文上限 | BF16 / 2048 |
| Speculative method / draft 长度 | `dspark` / 7 |
| Draft sampling / attention | `probabilistic` / `FLASH_ATTN` |
| 短生成计划（尚未完成） | 2 条短请求，每条 32 token，temperature=0，ignore_eos=True |
| 执行模式 | eager，prefix caching 关闭 |

检查脚本还计划通过私有本机 worker RPC 读取模型类型、层数、两个 head 是否存在和
`fc.weight[0, :8]`，与 HF 原值转 BF16 后逐值对照。
该参数抽查需要在这次本机测试中设置 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`，
以便传递检查函数；CPU 序列化预检已通过，GPU 参数抽查尚未完成。这个设置只用于检查脚本。

## 运行记录

| 目录（项目根目录下 `outputs/`） | 结果 |
| --- | --- |
| `dspark_vllm_load_20260916_run1` | 原实现加载失败，保留 `hc_mult` 完整日志和配置回归的失败/通过记录 |
| `dspark_vllm_load_20260916_run2` | 修复后模型加载、KV 分配和 warmup 成功；检查脚本的 callable RPC 序列化未启用，生成未执行 |
| `dspark_vllm_load_20260916_run3` | 检查脚本已修正；外部任务进入 GPU 后，监控器停止自己的测试进程，未完成加载 |
| `dspark_vllm_load_20260916_run4` | target 和 draft 分片读取后进程收到 SIGTERM，返回码 241；监控器未报告资源冲突或超时，信号来源尚未定位 |

后续需在可持续使用的 GPU 窗口完成参数抽查和短生成。测试脚本为
[`smoke.py`](../../../outputs/dspark_vllm_load_20260916_run4/smoke.py)，
监控和清理入口为 [`launch.py`](../../../outputs/dspark_vllm_load_20260916_run4/launch.py)。
新一轮应使用新的输出目录，保留已有失败和成功初始化记录。

短生成用于确认加载后的 DSpark 路径可以执行；即使两条请求完成，也不能作为接受率、模型质量或
加速效果结论。原生 vLLM 推理结果与 [4+4 训练验收](../../../deepspec/pipeline/VALIDATION.md)
分别记录。
