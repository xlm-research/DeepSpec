# 目标 CUDA 节点传输验证（2026-09-18）

本轮在 `dev-951e3d4b-0` 使用固定环境
`/tmp/deepspec_vllm_torchtitan_envs` 完成了真实 Mooncake wheel、GPU 目标缓冲区和
RDMA 路径的节点级探针。环境路径是符号链接，实际目录为
`/tmp/deepspec_vllm_torchtitan`；探针使用的解释器仍是符号链接路径下的
`bin/python`。

## 环境和结果

| 项目 | 结果 |
| --- | --- |
| Python | `/tmp/deepspec_vllm_torchtitan_envs/bin/python` |
| Mooncake distribution | `mooncake-transfer-engine 0.3.13.post1` |
| Native module | `mooncake/store.so`，实际加载成功 |
| Torch / CUDA runtime | `torch 2.13.0+cu130` / CUDA 13.0 |
| GPU | 8 × NVIDIA B300 SXM6 AC |
| TCP Mooncake put/get | 通过 |
| 同节点 GPU 目标缓冲区 get | 通过，`cuda:0`，SHA256 一致 |
| RDMA client setup | 通过，发现 `mlx5_10` 和 GID index 3 |
| RDMA QP/数据传输 | 失败，`No such device`，不能记作 RDMA 通过 |

原生库的 SHA256、版本和 GPU 清单见
[`environment.json`](../../../outputs/target_cuda_transport_validation_20260918_175320/environment.json)。

## Mooncake wheel 和 GPU 直收

探针启动真实 `mooncake_master`，创建 256 MiB Store 池。CPU writer 写入六个字段，
总计 6,300,688 字节；reader 使用 `TensorStore.get(..., device="cuda:0")` 接收两个
特征字段，共 6,291,456 字节。两个 GPU tensor 均为 contiguous、设备为 `cuda:0`，
逐字段 SHA256 与 CPU 源数据一致；显式删除后六个对象均不可见。

结果见 [`result.json`](../../../outputs/target_cuda_transport_validation_20260918_175320/result.json)。
这证明当前节点和当前 wheel 可以把 Mooncake 数据直接写入 GPU 目标缓冲区；它不等同于
跨节点 RDMA，也不等同于完整 Qwen 训练链路的 GPU 直传验收。

## RDMA

`protocol=rdma`、`rdma_devices=mlx5_10` 的客户端初始化和 128 MiB segment 挂载通过，
但节点只读预检显示：

- `mlx5_10/1` 是 `ACTIVE`、`Ethernet`，但当前网络命名空间没有可见 netdev；
- sysfs 的 GID 3 为全零，`gid_type`、`netdev` 和 IPv4 RoCE 地址均不可读；
- 当前进程没有 `CAP_NET_ADMIN` 或 `CAP_SYS_ADMIN`；
- 独立进程中的 RDMA client 做本机 put 时，QP 在 RTR 阶段报 `No such device`，所有
  put 返回 `-800`。

预检摘要见 [`rdma-preflight.json`](../../../outputs/target_cuda_transport_validation_20260918_175320/rdma-preflight.json)，
初始化和失败传输的完整记录见
[`rdma-attempt.json`](../../../outputs/target_cuda_transport_validation_20260918_175320/rdma-attempt.json)
及 [`rdma-cross-process-local-result.json`](../../../outputs/target_cuda_transport_validation_20260918_175320/rdma-cross-process-local-result.json)。

因此当前 RDMA 阻塞点是目标节点的 RoCE 网口/GID 没有进入容器网络命名空间，不能用设置
GID index 或仅看到 verbs 设备替代平台网络配置。需要平台先挂载可见 RoCE netdev，
并使 GID、IPv4、路由和对端匹配；随后才能进行两节点 `--transport-only` 和 RDMA20。

## 回归

在同一环境补跑：

```bash
CUDA_VISIBLE_DEVICES='' /tmp/deepspec_vllm_torchtitan_envs/bin/python -m pytest -q \
  tests/test_mooncake_transport.py tests/test_pipeline_store.py
```

结果为 **13 passed**（128.53 秒，14 条上游弃用警告）；`compileall` 也通过。
