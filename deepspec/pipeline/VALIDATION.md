# 单机 4＋4 卡验收记录

2026-09-16，真实 Qwen3.8-27B target 与五层 DSpark draft 的流水线运行正常退出。
本轮证明单机短程训练链路已打通；跨机接入、128K 长程运行和性能对照仍待后续验证。

## 基线与配置

| 项目 | 实际值 |
| --- | --- |
| 主机 | `app-e110ba357a9f4d969bc3d910774f334e-64776bd647-2n7s6` |
| 项目 | `/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm` |
| 分支 / commit | `dev/vllm_torchtitan` / `ee5d0c82f89358ef03f735af9ac3fd17feb33a39` |
| vLLM commit | `1ee54c40df7ffe2c8934f5bd1c79917f34cb954e` |
| Python 环境 | `/tmp/deepspec_vllm_torchtitan_envs`，未使用 uv |
| 生产者 | 一个 vLLM 实例，TP4，Ray 分配 GPU 0–3 |
| 消费者 | 一个 TorchTitan 实例，DP1 / TP4 / CP1 / PP1，GPU 4–7，GAS4 |
| Target | `/mnt/afs_agents/hongjiawei/share_models/Qwen/Qwen3.8-27B` |
| Draft | 原生五层 DSpark，H=5120，512 anchors，block size 7 |
| 特征层 | `[1,16,31,46,61]`，并保留最终 norm 后隐藏状态用于监督 |
| 样本 | 原生预处理生成的 12 个真实样本，每个 4096 token |
| 传输 | Mooncake Store，TCP 配置，pinned CPU 接收后传入 GPU |
| 内存池 | 单节点共享 4 GiB，特征预留窗口 8 批 |

上述 commit 是未提交改动的基线。本次增加 `deepspec/pipeline/` 与测试，修改原生
数据读取、微批物化接口及归一化初始化。原有文档删除、其他输出和 vLLM 子模块未提交
改动保持原状。实际执行代码的 SHA256 见运行目录的 `environment.json`、
`initialization-fix.json` 和 `verification-status.json`。

## 实际结果

运行目录：[`dspark_pipeline_4plus4_20260916_run4`](../../outputs/dspark_pipeline_4plus4_20260916_run4)。

| 检查项 | 结果 |
| --- | --- |
| 正常退出 | 启动器退出码 0；生产者、消费者均完成 |
| 生产 / 释放 | 12 / 12 批，剩余对象 0 |
| 每 rank 数据顺序 | 四个 rank 均按位置 0–11 消费，无遗漏或重复 |
| 读取校验 | 48 次读取全部通过 SHA256 校验 |
| 源对象释放 | 每批均在四个 rank 全部读取确认后释放 |
| 训练更新 | 四个 rank 均完成 3 次更新，消费位置依次为 4、8、12 |
| Loss | 4.56851、3.03412、4.12592；对应不同更新组，不据此判断收敛 |
| Context 梯度 | 四个 rank 的 fc、第一层 K/V 投影梯度均非零、有限 |
| Checkpoint | 完整 DCP `checkpoints/step-3`，训练步 3，消费位置 12 |
| Checkpoint 独立读取 | metadata SHA256 匹配；从 DCP 实际读取的 fc Adam step 为 3 |
| 背压 | 记录到 9 次等待；最大特征预留 2,013,790,336 字节，约 1.88 GiB |
| 组件与回归 | 10 项通过，47.57 秒 |
| 清理 | 本次模型 GPU 进程、Ray、Mooncake 均退出；临时暂停的 CI 已恢复 |

首个完整累积组的 fc 梯度范数为 116.49，四个 TP rank 一致。第一层 K、V 的本地
分片梯度范数分别位于 2.32–3.39 和 51.95–58.24，符合各 rank 持有不同分片的布局。
隐藏特征经过原生 `fc → hidden_norm → 各层 context K/V` 路径参与了反向传播。

事件记录中，推理区间与训练计算区间的交集约 0.783 秒，后台特征读取与计算区间的
交集约 2.333 秒。这些是主机事件区间，包含调用与同步开销；不能作为 GPU kernel
重叠比例或相对原流程的加速比。

## 跑通前修复的两个问题

1. **TP 归一化层漏初始化。** 指定环境的 Transformers 5.16.1 通用初始化器按
   类名识别 RMSNorm，`DraftNorm` 不符合名称条件。模型从 meta 存储物化后，这些
   权重未被正确设为 1，真实运行出现 fc 零梯度。新增回归测试将未初始化存储填充为
   NaN，稳定复现漏初始化；原生 `init_weights()` 按 `Qwen3RMSNorm` 继承关系显式
   初始化后，测试与真实四卡反向检查均通过。没有改变模型结构、loss 或优化器规则。
2. **检查未识别 SelectiveAC 包装路径。** `named_parameters()` 中出现
   `layers.0._checkpoint_wrapped_module...`，原检查找不到裸路径下的 K/V 参数。
   现通过模块访问取得参数，保留非零、有限梯度门槛。

复现和修复记录：

- [初始化失败测试](../../outputs/dspark_norm_initialization_red.log)
- [初始化修复测试](../../outputs/dspark_norm_initialization_green.log)
- [最终 10 项测试](../../outputs/dspark_pipeline_component_tests_final.log)
- [最终运行结果](../../outputs/dspark_pipeline_4plus4_20260916_run4/result.json)
- [独立验收](../../outputs/dspark_pipeline_4plus4_20260916_run4/verification-status.json)
- [训练日志](../../outputs/dspark_pipeline_4plus4_20260916_run4/consumer.log)
- [时序与生命周期事件](../../outputs/dspark_pipeline_4plus4_20260916_run4/events.jsonl)
- [CI 恢复记录](../../outputs/dspark_pipeline_4plus4_20260916_run4/ci-restoration.json)

## 验证边界与复现

本轮没有运行跨机 RDMA、CPU→远端 GPU 直传、128K 性能对照或长时间稳定性测试。
内存池及队列容量受到现有预算控制，但本轮的池内预留峰值不等于整机 CPU RSS 峰值。
Checkpoint 的持久化和进度已检查；没有执行训练重启恢复实验。按当前约定，特征
消费后释放，不实现特征恢复、重新生成或故障重放。

复现前预留 8 张 GPU。运行命令见 [README](README.md)，输出目录须使用新名称。
原有文件特征读取路径仍保留，选择原生 recipe 即可使用原流程；归一化初始化修复
同时适用于该路径在当前环境中的 TP 运行。
