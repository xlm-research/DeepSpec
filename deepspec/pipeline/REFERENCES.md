# 接入边界的源码核实

阅读日期：2026-09-16。下载的源码和 commit 记录位于
`outputs/dspark_pipeline_reference_20260916/`。

## 版本边界

| 项目 | 本次依据 |
| --- | --- |
| 当前项目 | `dev/vllm_torchtitan`，`ee5d0c82f89358ef03f735af9ac3fd17feb33a39`，加当前未提交的接入改动 |
| 当前 vLLM | 子模块 `1ee54c40df7ffe2c8934f5bd1c79917f34cb954e` |
| slime 源码 | `4c193f1f37509cca70f0e88807a9305b70f63f4e` |
| Mooncake 参考源码 | `9fb95ed0339e15b8f969d32df599f3c8ebcbba2a` |
| 实际执行的 Mooncake | 指定 Python 环境中的 wheel `mooncake-transfer-engine==0.3.13.post1` |

没有证据证明该 wheel 对应上述 Mooncake 源码 commit。源码用于检查调用机制；
wheel 的实际行为以当前安装包和组件测试为依据，不将两者视为相同构建。

## slime：采用资源与生命周期机制

| 已读文件、函数 | 源码行为 | 本项目接入 |
| --- | --- | --- |
| [placement_group.py](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/ray/placement_group.py)：`_create_placement_group`、`create_placement_groups` | 等待资源可分配，取得实际节点/GPU 编号；非 colocate 布局使用不同 bundle 区域 | `run.launch` 为两个框架分配不同 placement group，运行事件记录实际 GPU ID |
| [actor_group.py](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/ray/actor_group.py)：`RayTrainGroup._allocate_gpus_for_actor`、`async_train` | 将进程放入指定 bundle，以 futures 协调任务完成 | 复用资源隔离、异步调用和统一失败处理的机制；TorchTitan 由每节点 launcher 调用其原生入口 |
| [train_actor.py](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/ray/train_actor.py)：`TrainRayActor.__init__`、`init` | actor 设置 rank 环境并初始化通信组 | 本项目的通信初始化继续由 torchrun/TorchTitan 执行，避免重复初始化 |
| [train_async.py](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/train_async.py)：`train` | 在训练当前 rollout 前启动下一轮生成 | 本项目通过有界的对象窗口允许后续生产继续进行；消费仍遵守完整 GAS |
| [rollout.py](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/ray/rollout.py)：数据分区与包装路径 | 包含将训练数据放入 Ray object store 或使用 NIXL 的分支 | 此数据传输路径不用于本项目隐藏层；这里仅传递 Mooncake 对象描述符 |

当前 vLLM 的 `EngineArgs.create_engine_config` 取得所在 placement group，
`RayExecutorV2._init_executor` 在其 bundle 中创建 worker，并解析实际 GPU 映射。
因此生产 frontend 只占 CPU，四张 GPU 全部由 vLLM 创建的 worker 使用。
没有额外启动一层 torchrun 来包装 vLLM worker。

## Mooncake：注册缓冲、完成判定与保护

已读调用链：

```text
store_py.cpp 的 batch_put_from / batch_get_into
  → RealClient::batch_put_from / batch_get_into
  → *_internal 构造 Slice、查询副本
  → Client::BatchPut / BatchGet
  → TransferSubmitter → Transfer Engine 或符合条件的本进程复制
```

- [Python binding](https://github.com/kvcache-ai/Mooncake/blob/9fb95ed0339e15b8f969d32df599f3c8ebcbba2a/mooncake-integration/store/store_py.cpp)
  把整数地址转为缓冲指针，批量读写和注册接口在调用 C++ 时释放 GIL。这为后台预取提供源码依据；
  当前 wheel 的多进程加载与有界后台读取已通过组件测试，GPU 重叠仍需实测。
- [RealClient](https://github.com/kvcache-ai/Mooncake/blob/9fb95ed0339e15b8f969d32df599f3c8ebcbba2a/mooncake-store/src/real_client.cpp)
  在内存副本路径上把调用者提供的缓冲区组成 Slice，交给 `Client::BatchGet`。
  其他存储层存在临时缓冲路径，因此不能把所有 `get` 接口都描述为相同复制路径。
- [Client](https://github.com/kvcache-ai/Mooncake/blob/9fb95ed0339e15b8f969d32df599f3c8ebcbba2a/mooncake-store/src/client_service.cpp)
  的 `BatchPut` 依次派发、等待传输、完成写入提交，再返回结果；`BatchGet` 等待传输 future。
  本项目据此在全部写入成功后发布 READY，在读取成功和字节校验完成后确认消费副本。
- [TransferSubmitter](https://github.com/kvcache-ai/Mooncake/blob/9fb95ed0339e15b8f969d32df599f3c8ebcbba2a/mooncake-store/src/transfer_task.cpp)
  的 `canUseLocalMemcpy` / `isSameProcessEndpoint` 要求完整 endpoint 匹配；同一主机不等于同一地址空间。
  配置 TCP、同机运行或设置 memcpy 开关，都不能单独证明实际传输绕过网络栈。
- [MasterService](https://github.com/kvcache-ai/Mooncake/blob/9fb95ed0339e15b8f969d32df599f3c8ebcbba2a/mooncake-store/src/master_service.cpp)
  的 `EvictGroupOrObject`、租户内存淘汰路径检查 hard pin。显式删除是另一条路径，
  因此本项目同时使用 hard pin 与全部读者确认；hard pin 本身不代表故障持久化。

## 当前已验证与尚未验证

## TorchSpec Mooncake 设计借鉴

本轮还对照了 TorchSpec commit `6c042a87140a84d13839e341ece2c5c3ada918bc`：

- [`buffers.py`](https://github.com/lightseekorg/TorchSpec/blob/main/torchspec/transfer/mooncake/buffers.py)：复用注册 host buffer、CUDA event 生命周期和有界异步 put；本项目对应 `mooncake/buffers.py` 与 `TensorStore.put_async()`。
- [`store.py`](https://github.com/lightseekorg/TorchSpec/blob/main/torchspec/transfer/mooncake/store.py)：能力探测、`batch_exists`/可见性等待、native client 串行化和 partial put 清理；本项目对应 `MooncakeCapabilities`、`wait_for_keys()` 和 `_cleanup_keys()`。
- [`eagle_store.py`](https://github.com/lightseekorg/TorchSpec/blob/main/torchspec/transfer/mooncake/eagle_store.py)：把发布提交放在 transfer 完成之后；本项目保持 READY 发布晚于 put handle 完成，并将描述符校验留在 FeatureBuffer。
- [`deferred_delete.py`](https://github.com/lightseekorg/TorchSpec/blob/main/torchspec/transfer/mooncake/deferred_delete.py)：删除失败不提前释放所有权；本项目用 `DeleteManager` 重试删除，成功后才调用 `ledger.deleted()`。

这些实现只借鉴生命周期和错误处理模式，不把 TorchSpec 的 Eagle 数据格式或 GPU/RDMA 运行假设带入当前六字段协议。

已验证：真实安装包的 CPU 注册缓冲读写、SHA256 对齐、分块对象、容量压力下 hard-pin
对象保留、全部读者确认后删除、有界预取、四个独立 CPU rank 的原生加载器与 GAS 背压。

2026-09-16 后续的 `outputs/dspark_pipeline_4plus4_20260916_run4` 已完成真实 Qwen3.8-27B
与 DSpark 的单机 4+4 整链路验证：12 个 4096-token 微批、48 次读取校验、3 次更新、
有效 context 投影梯度、完整 checkpoint 和全部源对象释放。详见 [验收记录](VALIDATION.md)。

尚未验证：跨机 CPU→GPU RDMA、安装包在该路径的实际复制次数、GPU kernel 重叠、
128K 长程运行及相对原流程的性能收益。已有重叠数值来自主机事件区间。
早期因外部 CI 占卡而中止的运行记录仍保留，成功证据以上述 run4 为准。
