# H800：128K 常驻参考与原生阶段成本

2026-09-15，单机8张H800 80GB。任务10的固定输入、模型、GAS、精度和计时契约
见[128K验收说明](dspark_native_128k.md)，原生连续/恢复一致性见
[H800对照报告](dspark_native_128k_h800_comparison.md)。

## 结果

旧常驻参考完成10次更新、完整DCP保存及所有worker退出。两个逻辑DP rank各20次
前向的监督、样本顺序和CPU/CUDA RNG，与原生rank0、4的记录精确一致。
参考消费同一份真实128K特征、初始权重和随机状态，保持5层DSpark、512 anchors、
block7、BF16参数/FP32优化状态、GAS2以及1000步调度和40步warmup。

参考使用物理GPU0、4执行FSDP2；其余6张GPU空闲。原生路径使用DP shard2×TP4，
占用8张GPU。**本次原生阶段路径的累计成本高于常驻参考，没有得到加速结果。**
该对照同时包含拓扑、特征校验和分阶段执行的差异，不能把全部差额归因于DCP恢复。

| 成本，秒 | 常驻FSDP2 | 原生连续10步 | 原生5+5阶段 |
| --- | ---: | ---: | ---: |
| 启动前校验/准备 | 0.022 | 1396.581 | 1341.556 |
| Worker启动至退出 | 1517.573 | 2132.553 | 2219.869 |
| 其中训练，含各自输入读取 | 1383.535 | 2023.132 | 2044.203 |
| 退出后GPU检查 | 0.169 | 0.140 | 0.275 |
| 完整累计成本 | **1517.764** | **3529.274** | **3561.700** |

常驻初始化最长rank为85.765秒，最终保存28.458秒，均已计入worker生命周期。
原生阶段总成本为常驻参考的2.347倍。每种配置只有一次成功运行；常驻参考的输入
读取/reader校验包含在训练区间，原生路径另有昂贵的启动前全量特征字节校验。
本表排除target生产、等待以及此前失败的尝试，嵌套的训练时间不重复相加。

## H800内存边界

默认allocator尝试在第一次L1 loss的概率差绝对值运算中失败，0次更新。
失败时请求3.32GiB，PyTorch活跃分配67.63GiB，另有5.43GiB未使用的保留显存。
错误和完整请求保存在`h800-resident-failure.json`，原始失败目录保持原样。

第二次尝试唯一调整为`PYTORCH_ALLOC_CONF=expandable_segments:True`。模型、
算法、优化器、特征和批次分组未变；该尝试完成全部更新。运行中仍记录可恢复的
分配失败，不能把它报告为零OOM/零重试。两个rank的最大allocated为70.924GiB，
最大reserved为74.680GiB；这两个指标不能直接与原生TensorBoard的active值混用。

本次参考的allocator设置与原生测量不同，应随结果保留。原生同拓扑的连续/分阶段
对照仍是衡量阶段边界成本的依据。

## 证据与复现

以下产物位于`outputs/dspark_torchtitan_orchestration_20260914/`：

- [常驻完成记录](../../outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800-resident-expandable/complete.json)：逐rank时间线、10步完成和GPU池释放。
- [监督/RNG对照](../../outputs/dspark_torchtitan_orchestration_20260914/h800-resident-supervision-comparison.json)：两逻辑DP rank与原生对应rank精确一致。
- [验收摘要](../../outputs/dspark_torchtitan_orchestration_20260914/h800-resident-acceptance.json)：DCP包含64个模型tensor键，所有元数据引用的storage文件存在且长度完整。
- [常驻日志](../../outputs/dspark_torchtitan_orchestration_20260914/h800-resident-expandable.log)及[默认allocator失败记录](../../outputs/dspark_torchtitan_orchestration_20260914/h800-resident-failure.json)。

本节未声称常驻与原生的全部模型/Adam张量逐位相同；完整状态逐位对照是在原生
连续与原生分阶段两轮之间完成的。常驻结果建立固定工作负载的性能参考。

在GPU池空闲时，以一个尚不存在的输出目录重新执行：

```bash
source ./env.sh
OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m tests.run_torchtitan_resident_benchmark \
  outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800 \
  outputs/dspark_torchtitan_orchestration_20260914/qwen38-128k-h800-resident-repeat
```

运行使用原生计划记录的初始化目录，不覆盖首次捕获的权重、RNG或特征。

## HF与失败补验

以下测试均显式提供真实checkpoint运行，没有skip；日志与上面的产物同目录。

- `h800-full-hf-export.log`：完整step10 DCP导出FP32，现有consumer的64个模型
  tensor逐值一致；重复调用不改写文件，源DCP文件哈希不变，CUDA未初始化。
  测试含读取、导出和校验共328.275秒，不能把该时间当作纯导出耗时。
- `h800-retention-{float32,bfloat16}.log`：真实两GPU、SelectiveAC、BF16训练，
  两种HF精度、保留策略和两恢复点的连续轨迹对照均通过，分别83.091和86.431秒。
- `h800-small-bf16-cpu-export.log`：BF16提交的CPU导出精度、consumer、幂等及
  源文件保护通过。
- `h800-retention-failure-green.log`：真实第4步DCP数据写入后，commit marker
  rename失败；前三份提交不变，无成功phase结果，worker全部退出。旧测试请求错误地
  沿用step3停点，已显式改为step4；`h800-retention-failure-red.log`保留该失败。
- `h800-pending-export-test-{float32,bfloat16}.log`：移走实际GPU导出的一个shard，
  经DeepSpec的恢复入口调用CPU导出后，consumer权重一致，后续调用幂等且DCP不变。

缺失shard测试先发现了分布式索引残留问题：CPU修复写入单个
`model.safetensors`，旧索引仍指向缺失分片。导出器现在在重新保存前移除旧单文件/
索引命名文件，再由本次保存生成正确布局；有效导出的幂等返回路径保持不变。
回归测试红日志为`h800-pending-export-test-red.log`，两种精度的绿日志见上。

小规模补验使用真实24Q/4KV DSpark类、DP2/GAS2固定特征和独立目标参考
（`h800-gqa-dp2-reference.log`）。全规模训练和完整状态连续性仍由真实128K记录证明。
