# DeepSpec 环境快照

- 来源：`/tmp/deepspec_vllm_torchtitan_envs`
- 文件：`env.deepspec_vllm_torchtitan_env.tar`（未压缩的 tar）
- 归档顶层目录：`deepspec_vllm_torchtitan_envs/`
- 包含环境的全部现有文件、隐藏的 `.cache/`、CUDA/Rust 工具和符号链接。
- 不包含环境目录外的模型权重或 vLLM 工作区源码与扩展。
- 创建时没有修改原环境，也没有替换路径或重建依赖。

## 校验与恢复

在归档所在目录执行：

```bash
sha256sum -c env.deepspec_vllm_torchtitan_env.tar.sha256
# 恢复时要求目标目录尚不存在，避免覆盖正在使用的环境。
test ! -e /tmp/deepspec_vllm_torchtitan_envs && \
  tar --no-same-owner -xf env.deepspec_vllm_torchtitan_env.tar -C /tmp
source /tmp/deepspec_vllm_torchtitan_envs/activate.sh
/tmp/deepspec_vllm_torchtitan_envs/bin/python \
  /mnt/afs_share/lezewei/DeepSpec_envs_h800/vllm/vllm_demo.py
```

## 当前保留的外部路径依赖

复用此原样快照，需要保留以下路径的可访问性：

- 激活脚本使用 `/mnt/afs_share/miniconda3/etc/profile.d/conda.sh`。
- vLLM 为 editable 安装，源码和预编译扩展来自
  `/mnt/afs_share/lezewei/DeepSpec_envs_h800/vllm/`。
- Triton 的部分 `__grp__*.json` 索引仍引用
  `/mnt/afs_share/miniconda3/envs/deepspec_vllm_torchtitan_envs/.cache/triton/`。
  打包前的 demo 跟踪确认从该 AFS 路径读取了 89 个 `.cubin`。
- 环境的激活配置和入口脚本包含 `/tmp/deepspec_vllm_torchtitan_envs` 绝对路径，
  所以应恢复到相同路径。仅解压到其他前缀不会自动完成路径迁移。

模型权重单独准备。当前 demo 优先使用
`/tmp/deepspec-model-cache/glm5-460f95f89e4c5af25439228e92ba425f019a25a2afdc54a0a0e477ff1ebde883/`，
并有 `/mnt/afs-agentpro/share/models/zai-org/GLM-5.3-Flash` 的回退路径；两者均未打进本包。

## 验证方式

创建后使用 GNU tar 的 `--compare` 对照源目录逐项检查归档内容，
并计算完整归档的 SHA-256。打包前该环境运行 `vllm_demo.py` 成功，
退出码为 0；对应记录为 `/tmp/vllm-cache-rerun.QYWn6BSD/demo.log`。
