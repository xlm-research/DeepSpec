# 96 卡训练“突然停机”排查与解决

排查日期：2026-08-27（Asia/Shanghai）

涉及输出：

- `output/dspark_96gpu_speed_test`
- `output/dspark_96gpu_new_dataset_speed_test`
- 启动脚本：`scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh`

## 1. 结论

这次 96 卡任务没有发生 OOM、NCCL 超时、Python 异常或节点故障。第一次任务是在外部环境变量 `MAX_TRAIN_STEPS=10` 的控制下正常完成 10 个 optimizer steps 后退出；同时 `SAVE_CHECKPOINTS=false`，所以退出时没有保存 checkpoint。

随后针对新数据集的尝试也没有真正启动训练。该次启动继承了 `DRY_RUN=true`，脚本执行的是 `echo torchrun ...`，只打印命令便以 0 退出。

根因是速度测试/诊断环境变量被带进了后续启动，而不是分布式训练代码崩溃。

## 2. 日志证据

### 2.1 第一次 96 卡任务正常达到人为步数上限

`output/dspark_96gpu_speed_test/logs/node_rank_0.log` 中有以下连续证据：

```text
schedule=fixed diagnostic override, max steps=10
Train dataset size = 435
Global batch size = 48
Steps per epoch = 9
Max train steps = 10
epoch=2 step=10/10 loss=4.2687 | ... | remaining=0.0min
Checkpoint saving is disabled for this training run.
[deepspec-launch-exit] ... node_rank=0 exit_code=0
```

汇总 12 个节点日志后：

- 12/12 个节点都有 `exit_code=0`；
- 没有命中 `Traceback`、CUDA OOM、NCCL watchdog/collective timeout、`ChildFailedError` 或非零退出码；
- 节点结束时间不同是各节点进程组清理耗时不同，不是逐台掉卡。

因此，“第 10 步附近突然停止”就是 `MAX_TRAIN_STEPS=10` 的预期行为。

### 2.2 新数据集任务只是 dry run

`output/dspark_96gpu_new_dataset_speed_test/logs/node_rank_0.log` 的实际执行行为是：

```text
+ echo torchrun --nproc_per_node 8 --nnodes 12 ...
[deepspec-launch-exit] ... exit_code=0
```

正常训练应显示 `+ torchrun ...`，这里多出的 `echo` 证明 `DRY_RUN=true`。该日志也仍带有：

```text
schedule=fixed diagnostic override, max steps=10
logging.save_checkpoints=false
```

即三项诊断设置都未清除。

### 2.3 新数据集的完整训练规模

新 JSONL 文件在排查时大小约 33.68 GB，共享索引文件与源文件 key 匹配，记录数为 713,144。当前训练调度的计算方式是：

```text
steps_per_epoch = floor(dataset_size / global_batch_size)
```

在 `GLOBAL_BATCH_SIZE=48` 时，1 epoch 应为：

```text
floor(713144 / 48) = 14857 steps
```

实际值仍应以启动后 rank 0 打印的 `Train dataset size`、`Steps per epoch` 和 `Max train steps` 为准；源文件 mtime 或训练参数变化后会重新计算。

## 3. 已实施的修复

启动脚本新增了以下保护，且不改变已有诊断/性能测试的默认兼容性：

1. 新增 `PRODUCTION_RUN=true` 生产安全模式。
2. 生产模式会在启动 GPU 进程前检查并拒绝：
   - `DRY_RUN` 不是 `false`；
   - `MAX_TRAIN_STEPS` 非空；
   - `SAVE_CHECKPOINTS` 不是 `true`。
3. 固定步数启动时明确打印“达到该 optimizer step 后会正常退出”。
4. dry run 时明确打印“只显示 torchrun 命令，不会启动训练”。
5. 成功退出诊断现在区分三种情况：
   - dry run 完成，未启动训练；
   - 达到 `MAX_TRAIN_STEPS`；
   - 数据集/epoch 推导出的完整 schedule 完成。

这样，生产任务即使继承速度测试环境变量，也会立即以非零状态和明确原因失败，不会再消耗 96 卡运行十步后才被误认为宕机。

## 4. 正确启动方式

### 4.1 SenseCore 生产环境变量

在 12 节点、每节点 8 卡的生产任务中设置：

```bash
PRODUCTION_RUN=true
DRY_RUN=false
SAVE_CHECKPOINTS=true
PROFILE_ENABLED=false
NUM_TRAIN_EPOCHS=1
GLOBAL_BATCH_SIZE=48
```

必须从任务环境中删除 `MAX_TRAIN_STEPS`，或将它设置为空字符串。不要设置成 `0`、`10` 或其他整数；生产安全模式会拒绝固定步数覆盖。

节点数、节点 rank、master 地址和端口继续由 SenseCore 注入。不要在所有节点上手工写死同一个 `NODE_RANK`。

### 4.2 新数据集生产启动示例

如果平台启动命令支持 `env -u`，可使用：

```bash
env -u MAX_TRAIN_STEPS \
  PRODUCTION_RUN=true \
  DRY_RUN=false \
  SAVE_CHECKPOINTS=true \
  PROFILE_ENABLED=false \
  NUM_TRAIN_EPOCHS=1 \
  GLOBAL_BATCH_SIZE=48 \
  OUTPUT_ROOT=/mnt/afs-agentpro/lezewei/DeepSpec/output/dspark_96gpu_new_dataset_train \
  TRAIN_DATA_PATH=/mnt/afs-agentpro/hongjiawei/code/DeepSpec-old/scripts/data/sensenova_flash_v15_use_data_json.thinking_toolcall.readable_full.flash_conversations.jsonl \
  bash scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh
```

如果使用 SenseCore 的环境变量表单，则删除 `MAX_TRAIN_STEPS` 项，并把其余变量按上面的值填写。脚本必须由平台在 12 个节点上各执行一次；不能只在开发机手工模拟 `NNODES=12`。

### 4.3 启动后的必查项

先检查每个节点日志头部。生产启动应出现：

```text
training world size=96
schedule=dataset-derived, epochs=1
profiler=false, save checkpoints=true
production safety=true
```

并且不能出现：

```text
schedule=fixed diagnostic override
WARNING: DRY_RUN=true
+ echo torchrun
```

rank 0 完成数据加载后，还应确认：

```text
Train dataset size = 713144
Global batch size = 48
Steps per epoch = 14857
Max train steps = 14857
```

如果这些数字与预期不符，应立即停止任务并检查数据路径、global batch 和 epoch，而不是等到训练结束。

### 4.4 仍需运行 10-step 性能测试时

性能测试可以继续显式使用诊断设置，但不要开启生产安全模式：

```bash
PRODUCTION_RUN=false \
DRY_RUN=false \
MAX_TRAIN_STEPS=10 \
SAVE_CHECKPOINTS=false \
OUTPUT_ROOT=/mnt/afs-agentpro/lezewei/DeepSpec/output/dspark_96gpu_speed_test \
bash scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh
```

脚本会明确提示它将在 10 steps 后正常退出。

## 5. 如何判断以后是否真的崩溃

先查聚合日志：

```bash
rg -n -i 'deepspec-launch-exit|deepspec-launch-diagnosis|traceback|out of memory|watchdog|collective.*timeout|ChildFailedError' \
  output/<任务名>/logs/node_rank_*.log
```

判断规则：

- `exit_code=0` 且最后一步等于 `Max train steps`：正常完成；
- `dry run completed` 或 `+ echo torchrun`：未启动训练；
- `exit_code` 非 0：按照紧邻的 `deepspec-launch-diagnosis` 和最早 traceback 排查；
- 没有任何 `deepspec-launch-exit`，日志被硬截断：再检查 SenseCore 容器 termination reason、节点驱逐、SIGKILL 和宿主机事件。

## 6. 验证结果

修复后已完成：

- `bash -n` shell 语法检查；
- `git diff --check` whitespace 检查；
- `PRODUCTION_RUN=true` 分别搭配 `DRY_RUN=true`、固定 `MAX_TRAIN_STEPS`、`SAVE_CHECKPOINTS=false`：三种情况均在启动前以 1 退出；
- `DRY_RUN=true + MAX_TRAIN_STEPS=10`：退出码为 0，但明确报告未启动训练并显示两条警告；
- dataset-derived dry run：命令包含 `train.num_train_epochs=1`、`train.max_train_steps=null` 和 `logging.save_checkpoints=true`；
- `python3 -m unittest -v tests.test_jsonl_dataset`：共享 JSONL 索引缓存测试通过；
- 原 96 卡日志复核：12 个节点正常退出，0 个故障特征命中。

没有在本地重复提交新的 96 卡作业；最终端到端验证需要按第 4 节在 SenseCore 重新启动生产任务，并确认日志中的四项生产配置和推导出的 14,857 steps。
