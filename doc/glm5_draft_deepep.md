# GLM-5.3 DSpark draft 的 DeepEP 接入

本次实现只替换 GLM draft 的 pure-EP token dispatch/combine，保留当前 FSDP2/HSDP、专家参数布局、router、grouped GEMM、DSpark loss 和 checkpoint 路径。vLLM target 的代码、权重、推理配置和特征协议未修改。这里没有迁移整个 TorchTitan Trainer。

## 实现边界

| 代码 | 职责 |
| --- | --- |
| `deepspec/distributed/deepep_dispatch.py` | DeepEP V2 compact BF16 dispatch/combine、自定义 backward、接收端 expert-major 排序；每次调用独立 handle |
| `deepspec/distributed/draft_expert_dispatch.py` | draft 后端选择、可选依赖检查、模型拥有的 buffer 释放 |
| `deepspec/modeling/glm5_next_parallel.py` | 仅 `draft=True` 时创建 dispatcher，所有 draft MoE 层共享一个实例 |
| `deepspec/modeling/deepseek_v4_parallel.py` | 在原生 pure-EP chunk 循环中接入可选 dispatcher；继续使用原专家 forward |
| `deepspec/modeling/pure_ep.py` | 专家 replica 梯度平均后补除 EP，修复专家梯度相对 dense 梯度放大的问题 |
| `deepspec/trainer/dspark_trainer.py`、`base_trainer.py` | 完成 backward 后、卸载 draft/退出训练前，collective 销毁 buffer |
| `scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh` | draft 后端/容量参数及依赖预检查；显式保留原 vLLM Python |

TorchTitan 的 compact training 设计作为实现参考，相关文件保留 BSD 3-Clause 声明。依赖锁定 DeepEP commit `01dc3aaac82068020353dce2c302e38153c0bfaa`（构建版本 `2.1.0+01dc3aa`），不依赖工作区 TorchTitan 包。

通信流程为：

```text
源 rank: hidden + global top-k IDs + scores
    → DeepEP compact dispatch（同目标 rank 内去重 token）
    → 本地 expert-major 展开
    → 原 grouped GEMM（传入 unit weights）
    → 真实 routing score 乘一次
    → 按接收 token 合并 → DeepEP combine → 源 rank
```

backward 用 combine 归还输入和 routing-score 梯度，用带原 handle 的 dispatch 传回输出梯度。零 routing score 仍保留计算路径；不能删除这些 pair，否则其 router 导数会丢失。首版保留原 MAX token 数协商及 padding/chunk 逻辑，所有 rank 进入相同次数的 collective；仅 DeepEP 的 padding 使用无效 expert ID，跳过 padding 的通信和专家计算。DeepEP compact 接口仍有接收 token 计数的 CPU 同步，不能宣称消除了全部主机同步。

buffer 首次 forward 延迟创建，不进入 state_dict；多层、多 chunk 的 autograd graph 分别持有 handle。分区卸载时先等待 GPU 工作，再按 dispatcher 身份去重销毁，之后释放模型、优化器和 CUDA cache。下一分区创建新 dispatcher。

## 环境与启动

已在本机创建独立环境：

```text
/mnt/afs-agentpro/lezewei/.venvs/deepspec-deepep-v2-01dc3aaa/bin/python
PyTorch 2.11.0+cu130 / CUDA compiler 13.0
DeepEP 2.1.0+01dc3aa / 实际加载 NCCL 2.30.4 / NVSHMEM 3.4.5
```

从兼容的 CUDA 13 PyTorch 环境复现：

```bash
DEEPSPEC_DEEPEP_BASE_PYTHON=/usr/local/bin/python \
  bash scripts/setup_deepep_env.sh /path/to/new-draft-venv
```

安装脚本继承基础环境的 PyTorch，在新 venv 中安装私有 NCCL/NVSHMEM 并构建锁定版本 DeepEP。它不修改原环境，不是可直接搬迁的环境压缩包：私有 `.pth` 含 NCCL 绝对路径，多节点应在相同路径安装或共享可访问环境。

PyTorch wheel 的 RPATH 可能优先加载旧 NCCL，因此 venv 私有 `.pth` 在 import torch 前预加载自己的 NCCL，不向进程环境导出 `LD_PRELOAD`/新的 `LD_LIBRARY_PATH`。依赖检查通过已映射库的 `ncclGetVersion` 查询实际版本，不能用 `torch.cuda.nccl.version()` 的编译版本替代。本机验证从该环境启动原 `/usr/local/bin/python` 子进程仍加载 NCCL 2.28.9，且没有 DeepEP。

在**已有启动参数**上增加下列配置，`VLLM_PYTHON_BIN` 必须保持为原来实际使用的 target 解释器：

```bash
PYTHON_BIN=/mnt/afs-agentpro/lezewei/.venvs/deepspec-deepep-v2-01dc3aaa/bin/python \
VLLM_PYTHON_BIN=/path/to/existing-vllm-env/bin/python \
DRAFT_EP_BACKEND=deepep \
DRAFT_EP_MAX_TOKENS=4096 \
  bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
```

该 launcher 使用 `PYTHON_BIN -m torch.distributed.run`。手动启动测试也应这样调用，避免 PATH 中 `torchrun` 的 shebang 指向另一套 Python。原有 quick launcher `glm5.3-flash_dspark.sh` 固定两侧使用同一解释器；进行本实验时直接调用上面的生产 launcher，保留原数据、输出和 target 参数。

| 配置 | 默认/含义 |
| --- | --- |
| `DRAFT_EP_BACKEND` / `train.parallel.expert_dispatch_backend` | `native`；可选 `deepep`、`auto` |
| `DRAFT_EP_MAX_TOKENS` / `train.parallel.expert_dispatch_max_tokens_per_rank` | `4096`，一次 DeepEP dispatch 的每个源 rank 容量；长输入分 chunk |
| `DRAFT_EP` | 保留现有自动推导；节点 8 卡时通常为 8 |

显式 `deepep` 在依赖或布局不支持时失败；`auto` 在依赖/布局预检查失败时警告并使用 native。依赖可用性在 EP group 内达成共识，防止不同节点镜像导致同组混用两种后端；开始 token collective 后不会尝试切换后端。native 路径的 chunk 仍由原 `DEEPSPEC_V4_EP_TOKEN_CHUNK` 控制，做性能对照时两侧需设成相同值。

首版支持 CUDA BF16、GLM-5.3 DSpark、`EP>1`、`TP=CP=1`，hidden size 为 256 的倍数。关闭覆盖 dispatcher 的 `torch.compile`、activation checkpoint 和 CUDA graph；现有 attention 内部独立编译不受此限制。

## 多机通信

现有 torchrun、mesh 和 FSDP2/HSDP 继续负责多机训练。典型 `N` 节点、每节点 8 卡配置保持 `dp_replicate=N, dp_shard=8, ep=8, tp=cp=1`：DeepEP 通信在节点内，dense/expert replica 的梯度同步跨节点。DeepEP 不替代这部分 AllReduce，也不改变专家 GEMM 算子。

本机容器的 NCCL communicator 没有 GIN 能力，真实测试命令使用 `EP_DISABLE_GIN=1`，走节点内 NVLink。该变量没有写入生产 launcher 默认值。只有确认 EP group 完全位于同一 NVLink 域时才使用此设置；跨节点 EP 需要另行验证 RDMA、NCCL GIN 和拓扑配置，不能照搬节点内测试参数。上游要求见 [DeepEP README](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/README.md)。

## 梯度与恢复

原 DSpark loss 为 dense FSDP 的梯度平均补偿了归约组大小，而 pure-EP 专家只在正交 replica 维度平均，剩余梯度倍率为 EP。本次在现有 replica 平均后补除 EP，且只处理专家参数；router、shared expert、attention 和输入梯度保持原归约路径。梯度累积仍在整个 optimizer step 结束时归一化一次。

这一修正同时影响 native 和 DeepEP。旧训练的专家梯度、裁剪比例及 Adam 状态已受原尺度影响，因此新实验应使用独立输出目录和一致初始状态比较。权重/优化器的 checkpoint 格式未改，但本次修正不保证延续旧错误尺度的训练轨迹；已有分区 journal 的身份校验也不应绕过。

## 验证

2026-09-08 已完成：

* CPU：配置/launcher、可选依赖 fallback、target 不创建 dispatcher、分区卸载资源顺序、GLM draft 前后向和 HF checkpoint 回归。
* 专家归一化：7 种 2/4 rank CPU 拓扑，对真实 MoE 和 DSpark CE 比较集中参考的输入/全参数梯度及 SGD 更新，覆盖 CP、TP、expert_fsdp、dp_replicate。旧同步实现会失败；CPU 用等价 AllReduce 平均模拟 dense FSDP。
* 真实双卡 B300：DeepEP 输出、hidden/router/expert 梯度对集中参考，包含不等 token 数、空 rank、全部 rank 输入为空，以及所有路由集中于一个 rank。
* 真实双卡 B300：完整 tiny GLM draft + FSDP2，native/DeepEP 比较 CE/L1/confidence、Markov 输出、全部参数梯度和 SGD 更新；两层共享 buffer、7/21 token、容量 8 的多 chunk、销毁再创建两轮。
* 真实四卡 B300：`dp_replicate=2, dp_shard=2, ep=2` 的 HSDP 训练重复上述测试；最终共识与无效 padding 路径通过。两个创建/训练周期中，专家、router、其余 dense 参数的梯度及 SGD 后参数 relative L2 均为 0。这是同一节点内的 HSDP 拓扑验证，不是跨节点网络实测。

复现双卡测试（先选择空闲 GPU）：

```bash
CUDA_VISIBLE_DEVICES=3,6 EP_DISABLE_GIN=1 DEEPSPEC_TEST_DEEPEP=1 \
  /mnt/afs-agentpro/lezewei/.venvs/deepspec-deepep-v2-01dc3aaa/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=2 -m unittest \
  tests.test_deepep_dispatch.DeepEPHardwareTest \
  tests.test_glm5_deepep_integration -v
```

四卡 HSDP 使用 `--nproc-per-node=4`、四张空闲 GPU，并增加 `DEEPSPEC_TEST_DEEPEP_DP_REPLICATE=2`，仅运行 `tests.test_glm5_deepep_integration`。

## 性能结果

本机 B300 双卡、BF16、EP2、节点内 NVLink，固定每 rank buffer 容量为 4096。计时取每轮最大 rank 的 forward + backward 墙钟时间，再求均值；不包含初始化和 JIT/warmup。

| 专家几何和输入 | Native | DeepEP | 观察 |
| --- | ---: | ---: | --- |
| E8 / H4096 / I512 / top-k 2 / T256；warmup 5、计时 20 轮 | 5.156 ms | 5.507 ms | DeepEP 约慢 6.8% |
| GLM 实际专家几何 E288 / H4096 / I2048 / top-k 8 / T3584；warmup 3、计时 10 轮 | 14.324 ms | 14.309 ms | 基本持平 |

第二组 PyTorch allocated 峰值为 native 14.242 GiB、DeepEP 14.462 GiB；DeepEP 通信 buffer 容量另为 66 MiB，不计入表述中的 PyTorch allocator 峰值。这些测试只执行 routed experts、dispatch/combine 和 backward，不包含 router projection、shared experts、attention、DSpark loss、FSDP/expert replica 梯度同步、optimizer 或 vLLM 阶段。

原始数据：[小输入固定容量](benchmarks/glm5_ep_microbenchmark_fixed_capacity.json)、[GLM 专家几何](benchmarks/glm5_ep_production_geometry_microbenchmark.json)。此前小输入使用容量 256 的初测曾显示较低 DeepEP 耗时，但容量与生产默认不同，因此不作为默认配置的性能结论。容量需结合实际 draft token 数调优。

复现实际专家几何：

```bash
CUDA_VISIBLE_DEVICES=3,6 EP_DISABLE_GIN=1 \
  /mnt/afs-agentpro/lezewei/.venvs/deepspec-deepep-v2-01dc3aaa/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/benchmark_glm5_draft_ep.py \
  --num-experts 288 --hidden 4096 --intermediate 2048 --top-k 8 \
  --tokens 3584 --chunk 4096 --warmup 3 --steps 10
```

当前没有证据支持默认切换 DeepEP，所以保留 `native` 默认值。下一阶段应在空闲 EP8 资源上按实际 anchors/长度分布比较完整 draft step，并分别测 dispatch/combine、排序/展开和 grouped GEMM 的占比，定位是否值得进一步优化。此次未占用其他任务正在使用的 GPU。

尚未进行真实多节点/RDMA、完整规模 GLM draft 与 vLLM 分区交替的端到端吞吐验证。这些与小模型正确性测试、单 MoE microbenchmark 分开记录，不能将局部加速率当作完整训练加速率。
