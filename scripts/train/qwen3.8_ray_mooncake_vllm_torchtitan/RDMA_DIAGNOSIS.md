# 两机 RDMA 检查（2026-09-16）

**后续用户已明确暂时跳过 RDMA 测试。** 以下保留诊断证据与将来重测方法；当前无需继续处理
此阻塞。TCP 的十二卡跨机 DP2 已完成 4K、128K 验收，最新进度见 [HANDOFF.md](HANDOFF.md#04-跳过-rdma接入消费者跨机-dp2026-09-16-后续)。

当前跨机 TCP 可用，RDMA 尚未通过。第一台容器缺少可见的 RoCE 网络接口；
将来恢复 RDMA 工作时，先补齐平台网络配置，再重跑传输检查。
Python/Mooncake 初始化成功不能代替跨机传输成功。

## 平台复查（2026-09-16 17:47，Asia/Shanghai）

用户提供的平台 RDMA 网络名为 **`roce-cluster-01`**。尚未确认第一台是否已保存该配置、
是否已按平台要求重新启动；不能仅凭页面存在该网络名判断挂载已完成。

- 当前会话位于第二台，两台 Ray 节点仍为 `172.20.1.195`、`172.20.5.39`。
- 第一台仍只有 `lo/eth0`；`mlx5_10/1` 显示 ACTIVE/Ethernet，但没有可读的 GID 3、
  GID 网口和 IPv4。到第二台 `100.93.128.84` 的路由仍走 `eth0`，源地址 `172.20.1.195`。
- 第二台 `mlx5_10/1` 的 GID 3、`net1`、`100.93.128.84/22` 一致；当前两端无 GPU 计算进程。
- 两端均为 `netns shared copy-on-fork on`，均无 `CAP_NET_ADMIN` / `CAP_SYS_ADMIN`，
  没有挂载宿主机控制 socket 或 kubeconfig。容器内 root 无法据此移动宿主机 VF 网口。
- 当前 ServiceAccount 对两台 Pod 的读取均返回 **403 Forbidden**。完整权限自查只允许
  身份/权限自查及 API discovery，没有 Pod/工作负载的读写权限。本会话也没有平台管理连接。
  因此本轮没有执行平台变更、重建容器或启动 RDMA20。

本轮证据位于 [platform 诊断目录](../../../outputs/dspark_rdma_platform_20260916)：
`network-preflight.json`、`platform-access.json`、`baseline-contract.json`。
权限报告不包含 ServiceAccount token。TCP20 的运行源码快照在此次 17:47 复查时与训练源码匹配；
后续消费者 DP2 接入修改了流水线源码，各轮以自身快照为准。
80 条样本的身份、顺序与对照参数已经保存在 `baseline-contract.json`。

平台侧需要完成的具体操作：

1. 核对第一台绑定的 **`roce-cluster-01`** 是否与第二台相同，并确认修改已经保存和生效。
   若平台要求重启或重建才能附加网卡，使用平台支持的流程，保留共享目录并重新核对节点 IP。
2. 若第一台已保存并重启仍缺网口，检查其工作负载网络申请和实际附加状态。
   若平台使用 Multus/SR-IOV，对照两台的 `k8s.v1.cni.cncf.io/networks`、
   `k8s.v1.cni.cncf.io/network-status`、RDMA 资源 requests/limits 及 CNI 事件。
   资源键名、NetworkAttachmentDefinition 名称和数量必须取自平台及第二台实际配置，
   **不能把页面展示名直接猜成 Kubernetes 资源名**。
3. 第一台对应的 RoCE VF 网口需要进入当前容器网络命名空间，并由平台分配独立 IP；
   不复制第二台的 IP。验收需要 GID、网口、IP 和对端路由一致，随后实际跨机传输成功。

这些检查方向与上游 [SR-IOV CNI](https://github.com/k8snetworkplumbingwg/sriov-cni)
及 [RDMA CNI](https://github.com/k8snetworkplumbingwg/rdma-cni) 的设备和网络命名空间配置方式一致；
尚未取得该平台的 Pod 配置，不能断言平台使用了哪一种 CNI 实现。

## 实测证据

| 项目 | 第一台：生产端 | 第二台：消费端 |
| --- | --- | --- |
| 主机 | `dev-73915620-0` | `dev-3e072ae7-0` |
| Ray 地址 | `172.20.1.195` | `172.20.5.39` |
| 当前容器网口 | `lo`、`eth0` | `lo`、`eth0`、`net1–net8` |
| `mlx5_10` 对应网口 | 容器内不可见 | `net1`，`100.93.128.84/22` |
| `mlx5_10` GID | sysfs 无可读条目；verbs 返回 GID 3，IPv4 部分为 `100.93.129.30`，但无关联网口 | GID 3 为 RoCE v2，IPv4 部分为 `100.93.128.84` |
| 到对方 RoCE 地址的路由 | 经 `eth0` 默认网关，源地址为 Ray IP | 经 `net1` 直连 |
| Mooncake RDMA 初始化 / 1 MiB 注册 | 均返回 0 | 均返回 0 |

独立 CPU 探针使用 `mlx5_10`，两端自动选中 GID 3：

- 第二台两个进程间的 1 MiB RDMA write 成功，接收 SHA256 与发送内容一致。
- 第一台向第二台写入失败，返回 `-1`。第一台日志在 QP 转 RTR 时明确报
  `Failed to modify QP to RTR ... No such device`；第二台缓冲区保持原内容，校验未通过。
- 因此，本机 RDMA 证明该节点的绑定和内存注册可以工作；**尚无成功的跨机 RDMA 数据传输**。
  第一台网口不可见与该错误一致，平台分配/容器网络命名空间是下一步检查点。

证据保存在 [诊断目录](../../../outputs/dspark_rdma_stability_20260916)：
`verbs-inventory.json`、`route-and-vpd.json`、`roce-probe.json`、`roce-producer.log` 和 `roce-consumer-*.log`。
其中 `probe_roce.py` 可复现本轮小包检查，不加载 GPU 模型。

## 排除 NVLink 管理设备

最初 `mlx5_z0` 探针在传输阶段报 `transport retry counter exceeded`，全部 Store put 返回 `-800`。
随后读取第一台 `mlx5_z0–z3` 及第二台 `mlx5_z0` 的 PCI VPD，确认含 `SMDL=SW_MNG`，并标注
`ConnectX7 mezz for Nvidia B300 NVL8 Umbriel System`。它们是 NVLink 管理桥，不能作为本轮跨机网卡。
`mlx5_10` 的 VPD 则标注 ConnectX8 网络设备。

这与 [NVIDIA Fabric Manager 文档](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/#nvidia-hgx-b200-b300-gpu-baseboard)
对 B200/B300 CX7 管理桥及 `SMDL=SW_MNG` 的说明一致。设备名本身不是判断依据。
VPD 记录见 `producer-vpd.json`，失败首轮见
[rdma_probe1](../../../outputs/dspark_two_node_20260916_rdma_probe1)。

## 补齐配置后重测

在平台侧核对第一台是否申请并挂载了与第二台相同类型的 RDMA 网络资源，
使对应的 RoCE 网口/IP、GID 和 verbs 设备在当前容器网络命名空间中一致可见。
不要仅安装 Python 包或设置 `MC_GID_INDEX`；本轮实际已经自动选中了 GID 3。
若平台需要重建容器，应等待当前训练完成后再操作，并重新确认 Ray 实际节点 IP。

两端可先只读检查：

```bash
ip -br addr
rdma link show
cat /sys/class/infiniband/mlx5_10/ports/1/gid_attrs/ndevs/3
cat /sys/class/infiniband/mlx5_10/ports/1/gid_attrs/types/3
```

恢复集群后，模型任务退出且两端 GPU 空闲时，先运行标准 Store 传输检查。
以下 IP 仅适用于本轮节点仍保持原地址的情况，输出目录必须不存在：

```bash
export RAY_HEAD_ADDRESS=172.20.1.195:26379
export PRODUCER_NODE=172.20.1.195
export CONSUMER_NODE=172.20.5.39

/tmp/deepspec_vllm_torchtitan_envs/bin/python \
  scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/check_rdma.py \
  --device mlx5_10 --gid-index 3 \
  --output outputs/rdma-network-recheck.json
```

`check_rdma.py` 通过 Ray 在两端只读检查，不申请 GPU、不初始化 Mooncake。
退出码 0 仅表示本轮 IPv4 RoCE 网络条件通过，退出码 2 表示检查失败；
JSON 中的 `cross_node_transfer_tested` 始终为 false。新平台分配若改变了设备或 GID index，
按实际映射调整参数。当前复查退出码为 2，第一台失败，第二台本地映射检查通过。

**仅在上述检查通过后**，继续跨机 Store 写入、SHA256 读取和删除：

```bash
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train_multinode.sh \
  --output outputs/dspark_two_node_rdma_transport_recheck \
  --context-length 4096 --pool-gib 4 \
  --protocol rdma --rdma-devices mlx5_10 --transport-only
```

通过写入、读取 SHA256 和删除后，再以 TCP20 相同的 80 条样本、128K、TP4/GAS4、
64 GiB CPU 池及完整校验运行 RDMA20 对比：

```bash
bash scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/train_multinode.sh \
  --output outputs/dspark_two_node_128k_rdma20 \
  --source outputs/dspark_torchtitan_orchestration_20260914/128k-source.jsonl \
  --context-length 131072 --pool-gib 64 --window 8 --receive-device cpu \
  --steps 20 --timeout-seconds 7200 \
  --protocol rdma --rdma-devices mlx5_10
```

对照需要核验新旧 `pipeline.json` 中 80 条样本的 `input_identity`、`sample_id`、
`position`、`length`、`epoch` 均一致，保留 TP4/DP1/GAS4 和完整 SHA256。
运行成功后沿用 `outputs/dspark_rdma_stability_20260916/verify_stability.py` 独立核验
320 次 rank 读取、20 次更新、源对象全部释放及 DCP optimizer step=20，
再比较第 2–20 步间隔及分项耗时。平台若改变宿主机，应在新环境补跑 TCP 基线再比较。

目前只准备了这条重测路径，没有将失败探针记为 RDMA 验收或报告 TCP/RDMA 加速比。
`mlx5_z0` 不提供本轮可读的硬件计数器，因此 TCP20 的该项遥测为空；不能据此计算网卡带宽。
