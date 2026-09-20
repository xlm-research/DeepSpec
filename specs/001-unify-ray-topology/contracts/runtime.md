# 跨进程运行契约（待实现）

适用于 M0–M3，所有消息包含 `schema_version, run_id, plan_hash, sender_identity, event_id`。消息重放不重启角色、不重复预留额度。实际设备以 `(Ray node_id, GPU UUID)` 标识；GPU ordinal 只用于本进程可见设备映射。

## 组与服务边界

| 接口 | 请求 | 结果/约束 |
|---|---|---|
| `InferenceGroup.allocate/start` | 副本计划、外部已创建PG handles、deadline、gate handle | 只借用PG；非阻塞返回启动句柄，native core与worker由原生vLLM管理；错误传播到全运行 |
| `TrainingGroup.allocate/start` | 逐节点local GPU/CPU、node_rank、endpoint、环境、gate | 每节点Consumer只占本地卡；先上报allocation，再放行torchrun；禁用自动restart |
| `StoreService.start` | pool位置/bytes、owned或external master、TCP endpoints | 服务与全节点探针通过后返回ready；external只借用，不承担停止责任 |
| `group.ready/status` | run_id、期限 | 返回全部预期角色集合与身份/进度；单个actor存在不等于全组ready |
| `group.stop` | 原因、cleanup deadline | 幂等停止本组actor/子进程；返回每个资源released/unknown，不吞掉错误 |
| `NodeAgent.inspect/sample` | node预算/角色、计划hash、update_index/request_id | 现场采样的环境/设备/内存事实、boot_id/agent_epoch/sample_seq与预算组件；controller在最终准入时检查同钟年龄上界严格小于budget_snapshot |
| `NodeAgent.lease/cleanup` | fencing token、递增heartbeat序号、资源registry | 独立watchdog记录本地最后收到时间；仅清理登记的本次资源；过期token不重新续命 |

start 必须与 status/stop/heartbeat 解耦：训练 subprocess 或native engine初始化不能让控制RPC永远排队。所有Ray get/wait、subprocess退出、Store I/O与删除队列都有总deadline：受控接口显式传timeout，原生库未暴露timeout的阻塞调用由外层可终止actor/子进程及独立watchdog约束。取消Python future不能代替停止底层进程；重试不能重置总期限。

## 资源与初始化门禁

`report_allocation`：payload 包含 replica/node_rank、PG ID、bundle index、Ray actor ID、node ID、实际GPU IDs/UUID、PID/start-time和逻辑slot。来自全部推理worker与训练launcher的集合必须精确匹配计划，禁止重复、缺失、角色重叠或未授权设备。预期集合由不可变计划生成，不能由先到达的报告缩小。

协调CPU actor保存门禁状态；不运行模型。各组并发报告，因此vLLM engine constructor等待放行时driver仍可推进其他组。V2 worker shell Step6后、Step7 initialize_worker前等待此gate；训练launcher在启动torchrun前等待。失败/取消/超时使所有waiter收到同一运行终止原因。

`report_initialized`：每个vLLM DP/TP worker报告真实分布式身份；每个训练rank报告node/global/local/DP/TP身份、world、GAS、输入与模型hash；StoreService报告所有必要endpoint/探针。收齐且相符才允许ready，FeatureBuffer关闭原来的单一布尔开闸路径。保留TorchTitan全局barrier，但barrier不能代替拓扑校验。

## FeatureBuffer 与副本

| 操作 | 前置条件 | 可见结果 |
|---|---|---|
| `reserve_batch(position,count)` | 按计划连续位置；每个新更新边界预算新鲜；bytes/window足够 | 返回已准入前缀；源池reserved bytes增加，不等待未生产的整个batch |
| `begin_write(position,replica,node)` | 已reserved、writer=position%inference.dp、对应TP0节点匹配 | writing；重复或错writer拒绝 |
| `publish(position,descriptor,metrics)` | 所有写完成、key可见、身份/shape/dtype/字节一致 | ready；失败不能发布半写入对象 |
| `claim(position,reader)` | reader在该样本指定集合内、对象ready | descriptor；错reader/重复领取按既有规则拒绝 |
| `acknowledge(position,reader)` | 该reader已claim并完成完整校验、持有独立副本 | reader ACK；全部指定reader ACK后才可删除源对象 |
| `delete_complete(position)` | 删除成功且确认对象已不存在 | released；退源池额度，保留审计记录 |
| `reader_copy_retired(position,reader)` | 该rank不再有预取/前向/反向所需引用 | 更新节点副本观测，不改变训练提交游标 |
| `fail(reason)` | 任意非终态 | 关闭新准入、唤醒所有waiter；保留首因和后续诊断 |

删除失败保持字节占用；达到有限重试上限后全任务失败。异步writer/prefetch必须在分配新buffer前取得自己的额度。节点上界不能因为源对象释放而立即扣掉训练侧仍存活的独立副本。

每个新更新组按 [数据模型](../data-model.md#字节预算与释放规则) 分开核对静态/启动批准总量与运行期remaining_bound；只有可证明已计费并持续驻留的本运行内存能抵扣，不重复扣算已驻留池。所有节点快照须关联本轮请求；年龄上界由controller从请求发出到最终准入的同一单调时钟计算，`age >= budget_snapshot` 即拒绝新组。刷新、容量等待与重试共用transfer及剩余run总期限，heartbeat不刷新预算快照；已准入组不因下一组快照失效而阻塞在组内。

## 训练进度与保存

`rank_update_completed` 包含 rank、optimizer step、native微步cursor、global样本cursor、loss及计划身份；更新进度需要所有训练rank一致。`checkpoint_committed` 对应原生完整DCP和commit协议，包含计划/teacher身份与原生下一微步位置。

通用计数由冻结计划的U=steps、B=global_batch_size、D=training.dp推导：N=U*B、GAS=B/D、native cursor=N/D、global sample cursor=N、指定reader reads=N*training.tp；第k次更新的两个cursor为k*B/D和k*B。DP2的12样本是U=3的验收实例：8 ranks各3更新、native cursor=6、sample cursor=12、48 reads，每个DP组消费自己的6样本。U=5时为20样本、DP1/DP2原生cursor20/10、80 reads。verifier以计划预期值独立比对原生commit/DCP和事件，检查参数有限及预期更新，不硬编码验收常量或分片文件数。

## 运行证据前置

FR-018的实际事件采集及独立节点/进程/资源采样在首次真实验收前由T079/T080接通，并覆盖运行开始到清理结束。每次运行保留环境与落点、生产/读取/提交、预算/观测峰值、等待原因、生产token数及本节点计时、传输/训练耗时、checkpoint和cleanup；无测量用明确缺失原因，不能填0冒充测量，缺必需证据不能标passed。后续status展示可以回放已有记录，但不能补造过去的时长或峰值；live展示测试另用轻量可控run执行。

## 生命周期、取消与故障

状态序列遵守spec；`phase_detail`可以细分检查，但不新增含糊的成功状态。活动状态只能进入一个终态，首个失败原因保持；用户主动取消进入cancelled，之后清理错误附加报告。run总期限与cleanup期限分别计算，cleanup对所有资源共享总deadline，不能每个资源重新获得完整期限。

清理顺序：停止新准入并广播终止 → 取消/停止推理与全部训练 → 清理已发布源对象/关闭本任务Store clients → 释放本运行PG/actors → 停止自建master → 核查节点剩余进程与资源。正常收尾先保证消费与原生保存成功；模型/PG释放后完成CPU verifier，再完成最终清理和success确认。

在driver SIGKILL/断联时，Ray fate sharing回收PG；节点独立watchdog/subreaper继续清理已登记进程并记录节点报告。不能靠远程driver PID判断本机归属；父进程保护使用本机真实parent，远端driver生存依赖lease。进程归属需run marker与PID/start-time共同验证。权限不足、节点不可达、超时不能确认的资源记unknown，任务必须failed/cancelled且cleanup_complete=false。

status合并lease和节点报告：preview完成但未启动时显示preparing/preview_complete，无heartbeat不是故障；启动后才启用lease。driver失联后的旧token不能恢复执行，也不能把某个rank晚到的成功事件升级为全任务成功。cancel文件与消息按run_id/plan_hash校验，不能取消相邻目录的运行。

## 所有权与故障负例

- 用户已启动的Ray head/worker与外部Mooncake master只记录external引用；禁止全局ray stop、按进程名称pkill、按通用PG名称扫描回收。
- 未全部PG ready、native第2个core构造失败、allocation gate只收到部分角色、模型初始化超时，都回滚本次已获得资源。
- M3一个训练node/rank失效时另一个训练node不能继续更新；所有新生产也停止。
- 慢reader、删除失败、在更新边界跨batch的内存压力必须可观测并有界；不能通过跳样本清空账本。
- cleanup重复调用不误杀PID重用后的进程，不重复退别人的额度；并发旁路任务与外部服务持续健康。
