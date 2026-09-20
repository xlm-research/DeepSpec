# H800 环境与调试入口

本机地址：`10.120.5.46`，8 × H800 80 GB。项目目录：
`/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm`。

## 激活与启动

```bash
cd /mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm
source h800conda.sh

# 环境检查、64 GiB Mooncake 池读写、4K 三步、128K 三步。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/h800_debug.sh

# 单独运行真实 4K 训练；每次自动创建新输出目录。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/h800_debug.sh --stage 4k

# 单独运行 128K；增加 --steps N 可调整更新次数。
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/h800_debug.sh --stage 128k
```

使用 `h800conda.sh` 激活的共享 conda 环境；项目内的 vLLM 和 TorchTitan 源码优先。
CUDA 13 编译器和 Mooncake 需要的 CUDA 12 runtime 均取自同一环境。
`h800_debug.sh` 显式选择本地模型路径、64 GiB 池、window8、producer batch4、三次更新，
采用四卡 vLLM TP4 + 四卡 TorchTitan TP4/GAS4、TCP/CPU 传输。
不要同时启动两轮 GPU 训练；预检查要求八张卡空闲。

## 保留的调试服务

2026-09-20 启动于以下 tmux 会话：

| 会话 | 用途 | 地址 / 日志 |
| --- | --- | --- |
| `deepspec-h800-ray` | 常驻 Ray Head，8 GPU / 24 CPU | `10.120.5.46:26379` |
| `deepspec-h800-mooncake` | 常驻独立 Mooncake Master | RPC `127.0.0.1:50051`；metrics `http://127.0.0.1:9003/metrics` |
| `deepspec-h800-check` | 本次完整模型短测及退出后的调试终端 | `debug_logs/h800_setup_20260920/` |

```bash
tmux attach -t deepspec-h800-check
# Ctrl-b 再按 d：离开会话，保留运行。

source h800conda.sh
python -m ray.scripts.scripts status --address 10.120.5.46:26379
curl --noproxy '*' -fsS http://127.0.0.1:9003/metrics
nvidia-smi
```

独立 Mooncake Master 只提供控制服务，不分配特征池。
单机训练按项目设计创建自己的私有 Ray 和 Mooncake Master/FeatureBuffer，结束后自动清理。
常驻服务供单独组件调试，训练不复用它们。TorchTitan 是有限步数的训练任务；
验收成功后保留 checkpoint，模型进程释放 GPU，再次调试使用上面的启动命令。

重启独立服务时，在对应 tmux 会话中 Ctrl-C 停止原进程，再执行：

```bash
source h800conda.sh
RAY_NODE_IP=10.120.5.46 RAY_HEAD_PORT=26379 PIPELINE_RAY_BLOCK=true \
  bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/start_ray.sh head

# 在另一个终端执行。
source h800conda.sh
MOONCAKE_RPC_ADDRESS=127.0.0.1 MOONCAKE_RPC_PORT=50051 MOONCAKE_METRICS_PORT=9003 \
  bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/start_mooncake.sh
```

## 本次运行证据

- 环境版本和依赖检查：`debug_logs/h800_setup_20260920/environment.json`、`pip-check.txt`。
- Ray 真实远程任务：`debug_logs/h800_setup_20260920/ray-health.json`。
- 64 GiB 池：`outputs/h800_setup_20260920_pool/debug-result.json`；独立进程读取 251,695,120 字节、SHA256 校验及 34 块删除通过。
- 4K 短测已通过：`outputs/h800_setup_20260920_4k/`。12 条真实 4096-token 样本、48 次逐 rank 校验、四 rank 各三次更新、optimizer step=3，特征全部释放，退出码 0，无残留进程。独立证据为 `debug-result.json` 和 `4k/verification.json`；vLLM 日志另存 `4k/ray-actor-logs/`。
- 128K 复测已完整通过：`outputs/h800_setup_20260920_128k_fixed/`。12 条真实 131072-token 样本、48 次逐 rank 校验、四 rank 各三次更新、optimizer step=3，启动器退出码 0，无残留进程。独立证据为 `debug-result.json` 和 `128k/verification.json`；vLLM 日志另存 `128k/ray-actor-logs/`。三次 loss 为 4.30922、3.03616、4.06866，训练日志显存峰值 69.56 GiB/卡。
- 常驻 Mooncake 真实读写检查：`debug_logs/h800_setup_20260920/mooncake-health.json`。
- 最终汇总：`debug_logs/h800_setup_20260920/setup-result.json`。本次是单机 TCP/CPU 短测，不代表多机或长时间稳定性验收。

首轮 128K `outputs/h800_setup_20260920_128k/` 完成了三次更新和 checkpoint，但 GPU 监控在训练进程退出时读取 `/proc/<pid>/environ` 遇到 `PermissionError`，启动器退出码为 1，不能当作完整验收通过。
`deepspec/pipeline/cluster.py` 已修复此非 root 退出竞态：环境不可读时只沿用已确认的 PID/启动时间归属，未知或复用 PID 仍受检查；同时处理读取期间进程消失的 `ProcessLookupError`。
修复前定向测试为 2 failed / 1 passed；修复后 `tests/test_pipeline_cluster.py` 全部 15 项通过，日志在 `debug_logs/h800_setup_20260920/ownership-{red,green}.log`。

依赖检查通过，现有环境的依赖已经齐全，本次无需额外下载安装包。
修复还包括激活后优先加载本项目源码、配置 CUDA 编译器/runtime，以及让启动与导出脚本接受 `PIPELINE_PYTHON`，避免继续使用旧的 `/tmp` 环境。

`debug-result.json` 的 `status=passed` 表示该轮结束且通过独立验收。
启动日志、`consumer.log`、`events.jsonl` 和 `checkpoints/step-3/` 保留在对应输出目录。
