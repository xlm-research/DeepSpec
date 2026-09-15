# H800 环境准备（2026-09-15）

## 使用

在仓库根目录执行：

```bash
source ./env.sh
python -m pip check
```

本仓库的环境位于 `.envs/orchestration`，基于已恢复的
`/tmp/deepspec_vllm_torchtitan_envs` 创建，使用 `--system-site-packages`
复用 Python 3.12.14、PyTorch 2.13.0+cu130、Triton 3.7.1 和现有 vLLM。
基础环境必须保留；这个虚拟环境不能单独复制到另一台机器使用。
`env.sh` 设置 CUDA 13 工具链和当前仓库的 Python/vLLM 源码路径。

所有补充依赖通过 `python -m pip install` 安装。重建补充依赖的命令：

```bash
python -m pip install \
  -c .scratch/dspark-torchtitan-orchestration/environment-setup/constraints.txt \
  -r .scratch/dspark-torchtitan-orchestration/environment-setup/requirements.txt
```

DSpark 使用 full attention。此环境安装基础 `attn-gym==0.0.8`，省略
与 vLLM 的 `apache-tvm-ffi==0.1.11` 要求冲突的 `linear` 可选依赖。
该环境配置只针对当前 DSpark 任务。
安装日志、版本和完整包清单保存在本目录。

## 已完成验证

- `python -m pip check`：无依赖冲突。
- TorchTitan DSpark recipe、128K fixture、编排器和 vLLM 扩展导入通过。
- `gpu-smoke.log`：8 卡 NCCL all-reduce、BF16 矩阵运算，以及
  Inductor/Triton FlexAttention 前向数值比较和反向传播全部通过，退出码 0。
- `vllm-smoke.log`：当前仓库编译扩展的 CUDA RMSNorm 数值检查通过，退出码 0。
- `env.sh` 和恢复脚本通过 `bash -n`；缺少运行输入时，恢复脚本在启动前明确退出。

## 数据与运行状态

用户已将数据和请求迁入 `outputs/`（带 s）：

- `outputs/dspark_torchtitan_orchestration_20260914/128k-source.jsonl`：
  81,557,568 字节，80 条合法 JSON packed records，每条含 111–139 段对话。
  native preparation 已用指定模型的 tokenizer 选出 40 条样本，全部为
  131072 tokens；输入计划与预检摘要见运行目录和 `h800-preflight.json`。
- `outputs/dspark_torchtitan_orchestration_20260914/128k-request.json`：
  已收到，解释器、源码、模型和输出路径仍指向旧机器。
- 两份 acceptance/resource-conflict JSON 是运行记录，不包含实际特征。

恢复脚本已适配新机器，当前使用以下路径（均在仓库内）：

- `outputs/dspark_torchtitan_orchestration_20260914/128k-source.jsonl`
- `outputs/dspark_torchtitan_orchestration_20260914/128k-h800-request.json`
- 同目录下的 `qwen38-128k-h800/` 保存新运行的状态和特征。
- 同目录下的 `qwen38-128k-h800.log` 为主日志，`h800-orchestrator.pid` 为启动 PID。
- 同目录下的 `scale-initialization-h800/` 保存这次运行的初始权重与 RNG。

原始 packed 数据文件的记录路径：
`train_data/spec_o3_coldstartsft.repeat60.deepspec.packed_256k.jsonl`。
128K 验证使用 40 条真实样本、每条截取 131072 tokens。

另一个已确认存在的数据集：
`/mnt/afs_agents/hongjiawei/code/DeepSpec_basemain/train_dataset/sensenova-flash-lite-v42-text-all.jsonl`。
尚未验证这个数据集是否满足 128K 长度要求。

模型使用用户指定目录中的：
`/mnt/afs_agents/hongjiawei/share_models/Qwen/Qwen3.8-27B`。
已验证模型几何、18 个权重分片存在，以及本地 tokenizer 加载。
`env.sh` 设置 `TARGET_MODEL_PATH`，native recipe 使用该变量。
原迁入请求保持不变，新请求使用独立 run ID 和输出目录；本次重新生成
输入计划和特征，随后可通过同一个恢复脚本继续本次运行。

编排已启动，完整训练验收仍在进行。当前进程、恢复状态及故障处理以
`../continuation.md` 和仓库 `doc/benchmarks/dspark_native_128k_h800.md` 为准。
生成阶段的详细日志位于 `qwen38-128k-h800/phase-*/target/producer-{0,1}.json.log`。
