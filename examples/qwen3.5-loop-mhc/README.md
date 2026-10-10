# Qwen3.5 Loop + mHC：单机八卡文本推理

使用原始 `ckpt-225-bf16`，架构名保持 `Qwen3_5LoopForConditionalGeneration`。
环境为仓库的 `h800conda.sh`，需要八张可用 H800。配置采用 TP=8、PP=1、DP=1、
BF16 主干、FP32 mHC router 和 SSM 状态，以及 V2 runner 的 eager 文本模式。
输入与输出合计上限为 8192 tokens。

从 TorchSpec 根目录生成短文本：

```bash
source h800conda.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
uv run --no-project --python .venv/bin/python .venv/bin/python \
  examples/qwen3.5-loop-mhc/generate.py --prompt '请用一句话介绍你自己。'
```

使用实际 token 数检查 8K 总长度，保留 16 个生成 token：

```bash
uv run --no-project --python .venv/bin/python .venv/bin/python \
  examples/qwen3.5-loop-mhc/generate.py \
  --prompt-tokens 8176 --max-tokens 16 --ignore-eos
```

脚本使用 checkpoint 的 tokenizer 和 chat template，通过标准 `LLM.generate`
输出输入/输出 token 数、token IDs、完成状态和文本。`--ignore-eos` 仅用于长度压力运行；
普通文本运行沿用 `generation_config.json`，其中 EOS 包含 `248044` 和 `248046`。
可用 `--model /path/to/checkpoint` 覆盖默认路径，原权重无需转换或修改。

通过标准 vLLM OpenAI API 服务入口部署：

```bash
bash examples/qwen3.5-loop-mhc/run.sh --host 127.0.0.1 --port 8000
```

`config.yaml` 同时供服务和离线脚本使用。默认 `max-num-seqs=1`；少量变长请求可在同一
引擎中排队运行。非分块 prefill 要求 `max-num-batched-tokens >= max-model-len`。

模型按 `[0..3] + [4..27] * 3 + [28..31]` 执行，共 80 个执行位置；
同一 TP rank 上的 32 组物理权重共享，20 个 attention 和 60 个 GDN 执行状态各自独立。
每轮 mHC 都展开四流并在末尾取均值收敛，路由器在各 TP rank 完整复制。
加载后各 rank 会记录权重共享、路由器精度及执行状态对象检查结果。

当前配置校验会拒绝其他并行拓扑、量化、LoRA、推测解码、多模态输入、prefix caching、
chunked prefill、非 eager 运行和超过 8K 的模型长度。这里的功能与结构验证不代表
与训练实现或其他推理框架的数值等价性证明。

2026-10-10 在八张 H800 上通过原 checkpoint 的 `LLM.generate` / `LLM.chat`
完成功能验收，实际配置、各 rank 诊断、请求 token 和错误信息保存在
[verification.json](verification.json)。主要结果如下：

| 场景 | 实际结果 |
| --- | --- |
| 权重与执行状态 | 八个 rank 均通过共享参数、FP32 router 校验；每 rank 注册 20 个 attention 和 60 个 GDN 状态，SSM 为 FP32 |
| 短请求与 8K | 15 输入 + 16 输出；8176 输入 + 16 输出，均完成多步生成 |
| 总长度约束 | 8188 输入、申请 16 输出，实际生成 4；非法 token 和 8193 长输入被拒绝 |
| 停止行为 | 分别强制采样两个 EOS，均以 `stop` 完成；显式停止 token 也生效 |
| 变长与复用 | 32/257/2048 输入分别返回 4/6/8 输出，随后 15 输入 + 8 输出成功 |
| 正常聊天 | 重启后使用默认 EOS 设置完成 12 输入 + 16 输出，以长度预算结束 |
| 配置边界 | TP=4、FP16、LoRA、推测解码、多模态、非 eager、两个缓存开关、超长上下文及 BF16 SSM 配置均被拒绝 |
| 标准 Qwen3.5 回归 | 同一示例以标准 `Qwen3_5ForConditionalGeneration` checkpoint 完成 15 输入 + 16 输出，进程正常退出 |

上述生成请求所选 token 的 logprob 均为有限值。EOS 场景验证停止机制，不代表模型自然
生成 EOS 的概率；短样例也不用于评价模型质量。现有 Qwen3.5 注册用例为 8 项通过，
vLLM 暂存文件 hooks 与本次新增代码的类型检查通过。

缓存诊断首次直接返回 `torch.dtype` 时触发了既有 RPC 序列化限制。重启后在 worker 内
转换为可序列化摘要完成诊断；诊断进程的额外序列化开关未写入启动配置。八个 GPU worker
正常退出，EngineCore 父进程由既有超时机制清理，之后每卡仅占用 4 MiB。

完整 TorchSpec 测试命令 `pytest tests --maxfail=1 --timeout=120 -q` 运行至 69 项通过、
1 项失败、30 个 subtests 通过后停止，后续用例未执行。失败为既有全局 Ray mock
影响 `test_capacity_handling.py::TestAsyncTrainingControllerPoolBytes::test_initial_pool_bytes_is_zero`，
抛出 `module 'ray' has no attribute '_private'`。另一个既有问题是 vLLM 全局注册覆盖检查
缺少三个 `Glm5Next` 架构的测试条目。本次未修改这些无关测试，因此不宣称全套测试通过。
