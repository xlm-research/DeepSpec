**大数据集 CPU 内存优化方案**

状态：复核后的设计方案，尚未修改训练实现。日志确认了数据加载进程的 OOM；文末的小规模实验验证了若干实现假设，但没有完成真实语料、AFS 随机读取或端到端训练的规模验证。

**1. 问题与目标**

本次任务使用 DSpark、vLLM、`train_with_decode=true`，最大序列长度 131072，训练集有 2,050,980 条样本，原始 JSONL 约 141.34 GiB。tokenization 进行到 85% 时，训练节点内存达到日志报告的 973.46/1024 GB；`AsyncTrainingController.load_dataset` 单进程占 912.64 GB，触发 Ray 的 95% 内存保护阈值。

当前有三处主要问题：

- [dataset.py](../torchspec/data/dataset.py) 同时保留全量 `raw_samples`、`work_items` 和累积的 `results`；每条结果还可能保留格式化文本与 token 数组。
- [training_controller.py](../torchspec/controller/training_controller.py) 将完整结果保存在 `_stored_dataset`，并通过 `add_dataset()` 为整份训练数据创建待推理条目。已消费样本仍被 `_stored_dataset` 引用。
- `.pt` 缓存通过整份 `torch.load()` 加载，现有预 tokenized 加载器也会返回完整列表。只修改输入文件格式或复用缓存，不能消除全量驻留。

这里不能把多个容器简单理解成多份完整数据：`work_items` 引用原始消息，controller 内的队列条目也引用原样本；`torch.from_numpy()` 与结果数组共享存储。真正需要测量的是同时存活的原始文本、格式化文本、token 数组、Python 对象和进程间传输临时对象。日志没有给出这些部分各自的占比，也不能据此证明丢弃文本后的纯 token 数据一定放不下。

目标是把数据主体的内存成本从“随全量文本和 token 数增长”改成“随有限的预处理窗口和训练批次增长”，允许保留随样本数增长的紧凑索引。保留当前 tokenizer、模板、128K 长度、过滤规则、在线 decode 和训练损失语义。

首期覆盖本次纯文本 JSONL、DSpark、train-with-decode 路径。普通 Parquet/Arrow、外部预 tokenized 输入、多模态和延迟 tokenization 作为后续接入范围；它们可以继续显式使用原内存模式，不能在请求磁盘模式时静默退回全量加载。共享 controller 的 eval、离线生成等调用必须做兼容性回归，但首期不承诺已解决这些流程全部的内存问题。

**2. 推荐的数据流**

```mermaid
flowchart LR
    A[逐条读取原始数据] --> B[有界并发 tokenize]
    B --> C[分批写入 Arrow 缓存]
    C --> D[完整缓存与紧凑索引]
    D --> E[按 seed 和 epoch 打乱索引]
    E --> F[controller 按请求读取少量样本]
    F --> G[vLLM 在线 decode]
    G --> H[现有 hidden-state 传输与训练]
```

首次运行先完成缓存构建，得到过滤后的准确样本数，再启动现有训练流程；再次运行直接打开缓存索引。第一版不引入边构建缓存边训练，以便保持全局 shuffle、总步数、学习率计划和恢复位置的现有语义。

缓存构建时间仍与数据量有关；优化目标是内存有界和后续缓存复用。

**3. 磁盘数据集模块**

新增 `torchspec/data/dataset_store.py`，以一个小接口封装分片、索引、读取和缓存生命周期：

| 接口 | 约定 |
|---|---|
| `len(store)` | 返回已验证、过滤后的准确样本数，不读取全部样本 |
| `store.read_rows(indices)` | 只读取指定样本，按请求顺序返回；保留原有 ID、tensor dtype/shape、mask 和元数据语义 |
| `store.read_lengths(indices)` | 从紧凑索引读取长度，用于长度分组，不解码文本和 token |
| `store.close()` | 关闭文件映射和读缓存；可重复调用 |

保留现有内存存储作为小数据和行为对照的 Adapter，新增 Arrow 磁盘 Adapter。controller 通过同一接口访问，两种模式显式选择。

Arrow IPC 是当前首选存储候选，原因是已有依赖且支持增量 record batch 与按索引读取；尚未验证它在当前 AFS 上优于其他分片格式。正式定型前应以真实序列长度分布比较随机读取延迟、I/O 放大和 CPU 开销，至少与分片 `.pt` 加 mmap 的方式对照。接口设计不依赖 Arrow。

存储模块管理有限的文件句柄、映射和按字节限制的读缓存。纯 token 路径直接取得选中行的数值缓冲，再构造该批次所需 tensor，不能把整个 record batch 的 token 先转成 Python 列表。必要时复制选中的小数组，使它有独立生命周期：实验中一条逻辑大小为 1 MiB 的 Arrow 切片仍引用 8 MiB record batch 的底层缓冲。这是引用范围，不等于每次切片都额外分配了 8 MiB 物理内存。跨 Ray 只传当前批次数据，不传存储对象或文件映射句柄。

缓存内容包括 `data_id`、`input_ids`、`packed_loss_mask`、`seq_len` 和必要元数据。当前纯文本、非延迟 tokenization 路径不持久化 `formatted_prompt`：推理端使用的是 `input_ids`。当前实现仅截断 token，返回的格式化文本仍可比截断后的内容长；缓存构建 worker 应在结果返回主进程前移除不需要的文本，避免传输和排队期间继续保留它。复用原有格式化和 tokenization 逻辑，并保留其他模式对文本的需求。第一版保留 token 的现有 int64 表示；int32 存储可作为后续优化，必须验证值域，并在批次读取时恢复所需类型。

紧凑索引保存分片位置、行位置和长度；字符串 ID 与大字段留在磁盘。2,050,980 条样本的一份 uint64 顺序数组约 15.65 MiB，此数字不包含其他索引、库开销和文件页缓存。

**4. 有界缓存构建**

新增 `torchspec/data/dataset_cache.py`，复用当前解析、规范化和 `_tokenize_single()` 的语义：

1. 按原始行序号读取和分配 ID，不再构造全量 `raw_samples` 与 `work_items`。
2. worker 数可配置，初始从 8 开始；待处理任务最多为 worker 数的两倍，同时限制排队原始数据、结果和写入缓冲的字节数。
3. 消费完成结果并增量写入 record batch。为保持顺序可保留有限的重排窗口；提交额度在结果写入后归还，避免一个慢任务前积压无限结果。
4. 写完即解除该批原始文本和结果的引用。不能仅把 `list(imap(...))` 换成迭代器，却仍在其他位置累积全量结果。
5. 每个完整分片独立提交；完整 manifest 原子发布后才允许训练使用。首期可以在中断后重新构建；按输入位置断点续建放在核心加载链路通过验收之后，单独验证偏移、过滤和 ID 不重不漏。

不能把 `Pool.imap()` 自身当成背压机制。复核实验使用两个 worker、40 个任务，并阻塞首个任务：其他 39 个结果全部进入重排缓冲；将显式提交窗口限制为 4 后，首个任务之前只能积压 3 个结果。提交窗口必须计入已完成但尚未按序写出的结果。

字节预算控制的是流水线中的数据；worker 自身和 tokenizer 临时对象还需单独测量。单条超大原始记录可能超过窗口预算，应单独处理并报告，超过单样本硬上限时明确失败，不能静默丢弃或改变原有截断语义。JSON 数组当前使用整体 `json.load()`，若纳入大数据模式，需要流式数组解析或先流式转换，不能沿用整体解析。

缓存用独立版本目录保存，例如 `tokenized_dataset_v2/<fingerprint>/`。fingerprint 覆盖数据源身份/修订、文件列表、tokenizer 与模板身份、渲染版本、长度和过滤参数，以及 decode/defer 模式。采用单写者构建与临时文件原子发布，只有完整 manifest 可用于训练；在实际 AFS 上验证锁和重命名行为。

Arrow 模式不自动加载旧的整份 `.pt` 缓存，避免通过兼容分支再次触发 OOM。旧缓存保留供原内存模式使用。

特别注意：decode prompt 的 mask 可以没有监督 token，当前原始对话路径会把它交给生成阶段处理；现有 `_load_pretokenized_dataset()` 则会拒绝没有监督 token 的外部预 tokenized 样本。因此内部 Arrow 缓存必须保存模式并走独立缓存读取路径，不能简单将其路径交给现有预 tokenized 检测逻辑，也不能放宽外部数据原有的验证规则。

**5. controller 改为按索引调度**

`_stored_dataset` 改为存储句柄。controller 保留有序的 epoch 索引段、每段游标和有限的待处理条目：

- `_prepare_dataset()` 对整数索引执行与当前相同的 `random.Random(seed + epoch).shuffle()`；长度分组通过长度索引实现，保持稳定排序和原有分组边界。
- `submit_training_dataset(epoch, skip)` 创建索引段，跳过位置仍基于原有 shuffle 和长度分组后的顺序。返回提交样本数，不为所有样本构造 `InferenceInput`。
- `reload_dataset()` 将下一轮索引段追加在尚未完成的旧索引段之后，复用磁盘缓存。现有方法是向队列追加整轮数据；直接重置单个游标会丢掉旧轮尾部，也会改变训练序列。
- 正常换 epoch 不清空旧轮 ready 条目、未完成读取或推理端缓存。只有明确的恢复重建或关闭才取消旧读取；使用读取 generation 与段 ID 区分过期结果，不能将正常追加下一轮误判为取消上一轮。
- eval 使用待处理索引范围表示其切片，并保持优先级和 DP 对齐规则。共享调用中的 `submit_eval_chunk(0, eval_size)` 也不能再次物化全量输入。

不能在 controller 的 Ray 方法中同步执行慢磁盘 I/O。当前 controller 是默认并发度为 1 的同步 actor；`get_prompts()` 中的阻塞读取会同时阻塞 `try_dispatch_batch()`、结果接收和状态查询。推理端的异步 refill 不会改变这一点。

首选在 controller 内增加一个专用 I/O 线程，由它管理存储读取和有数量、字节上限的 ready 缓冲。Ray 方法只操作索引状态和取出已完成的数据，不等待磁盘；锁内不能做 I/O，也不通过提高整个 actor 的并发度解决问题。读取索引先登记为在途，成功后移入 ready 状态，失败则记录致命错误，不能推进后直接丢失这些样本。

需要将取样返回值改为同时携带 `items`、`source_exhausted` 和 `pending_read` 的结果。读取错误必须传到 controller 的训练派发检查和推理 manager，不能仅让后台 refill task 退出而让训练持续等待。关闭时先停止新读取、等待或处理在途读取，再关闭映射。

需要同步修改 [inference_manager.py](../torchspec/controller/inference_manager.py) 和 [loop.py](../torchspec/controller/loop.py) 的数据耗尽判定。目前前者把空列表视为数据耗尽，后者会参考 `prompt_buffer_size == 0` 决定 reload。

新状态至少区分“尚未读取的样本数”“正在读取”“已读待分发”和“源已耗尽”。只要还有未读取或在途读取，空 ready 缓冲不能标记 EOF，推理 manager 也不能据此提前发送最终尾批。确认数据源已经不能产生更多训练结果时，还要确认推理端没有 buffered/refill/in-flight 请求，以及结果回传已完成。

该状态协议同时影响训练 loop 和 [offline/generate.py](../torchspec/offline/generate.py) 的 drained 检查。后者同样使用空 prompt 队列判断提前耗尽，不能只修改在线训练入口。正常按计划完成一个 epoch 时保留旧尾部并追加下一段；数据源因过滤提前排空时，沿用既有训练策略处理，再进行有条件 reload。

保持现有基于 completed step 的恢复方式，并明确它仍是 best-effort：异步预取尾部可能重放。此内存优化不额外承诺精确一次处理，也不更换 shuffle 算法。

**6. 避免其他路径重新全量物化**

- 词表裁剪：当前 [preprocessing.py](../torchspec/data/preprocessing.py) 的 `_count_token_frequencies()` 收集全部 ID 和 mask 后执行 `np.concatenate()`。通用磁盘后端接入该功能时，必须改为逐批、按监督 mask 累加词频，常驻开销与词表大小有关。当前 DSpark 配置未启用词表裁剪，可以放在首期之后；未实现前该组合须明确拒绝。
- 数据数量：使用已完成 manifest 的有效样本数，供 `auto_calculate_training_steps()` 使用。
- 缓存命中：直接打开索引，不能先 `torch.load()` 旧缓存，也不能 `list(store)`。
- 正常结束与异常：推理读取和在途请求停止后，关闭存储句柄、清理局部缓冲；不要在仍有读者时删除持久缓存。
- 内存观测：分别记录 controller、tokenize worker 合计、节点/cgroup 使用量、匿名内存与文件页缓存。mmap 的访问页可能反映在 RSS 中，不能只看 Python 对象大小或单个进程 RSS。
- eval 输入按需读取只解决输入数据；训练端仍有独立的 hidden-state eval cache，需要单独核算，不应将本方案描述为解决了整个 eval 的内存需求。

**7. 配置与资源预算**

下面是拟新增配置，不是当前可直接使用的参数：

```yaml
dataset:
  storage_backend: arrow
  num_proc: 8
  preprocess_buffer_mb: 2048
  read_cache_mb: 512
```

这些字段需要加入 `DatasetConfig` 并验证正值。MiB 为这里的缓冲计量单位；实现时配置帮助必须写清楚。待处理任务数可先由 worker 数派生，避免暴露过多调节参数。现有推理 batch、fetch 和 buffer 配置继续使用。

撤下原方案中的“预处理 64 GiB、controller 8 GiB”固定验收值：目前没有真实 tokenizer、长样本工作集和训练共存内存的测量依据。上面的 worker 数和缓冲值也只是初始实验设置，不能保证整个进程或节点的内存上限。

预算应先用真实数据测量，再按下面的组成核算：

```text
预处理工作集 = 主进程基线 + worker 基线合计
             + 有界输入/结果/写入缓冲
             + 序列化与各 worker 处理长样本时的临时峰值

训练数据工作集 = 紧凑索引 + 有界预读/在途批次
               + 活跃映射页与读缓存 + Ray 批次传输临时对象

节点预算 = 数据工作集 + 模型/训练/传输等其他进程开销 + 余量
```

需要测量有效 token 总数及长度分布，不能从 141.34 GiB JSONL 推断最终 token 大小。例如这份数据若平均每条有 16,384 个 token，仅 int64 token 就约 250.36 GiB；平均 32,768 个则约 500.73 GiB。这只是敏感性计算，不是对真实平均长度的估计。丢弃文本并消除全量原始样本后，本次语料能否继续使用内存模式，需据测量判断。

节点总使用量必须另行监测，为模型初始化、训练、传输缓冲及保存 checkpoint 留出空间，继续保留 Ray 的 95% 保护阈值。磁盘容量通过样本预估和构建期间的写入统计检查，不能将原始 JSONL 大小当作缓存大小。若有节点本地 SSD，可将热点缓存放在本地，减少 AFS 随机读取；应恢复请求顺序，不能通过修改全局 shuffle 换取 I/O 性能。

**8. 实施顺序和验收**

| 阶段 | 交付内容 | 完成条件 |
|---|---|---|
| 零：测量与小范围验证 | 真实样本长度/字段占用、tokenizer 峰值、去掉冗余文本后的开销、AFS 读取对照 | 有依据地设置 worker、缓冲与节点预算，并确认存储候选 |
| 一：存储与构建 | DatasetStore、分片缓存、有限并发构建、完整 manifest 与新配置 | 当前 JSONL/decode 数据新旧输出一致；缓存命中跳过 tokenize；半成品不会被训练读取 |
| 二：训练接入 | 保留尾部的索引段、独立 I/O、明确 EOF/错误协议、共享调用兼容、退出清理 | shuffle/长度分组/skip 一致；慢读取和慢推理不会提前换 epoch；当前 decode 训练可运行 |
| 三：规模验证 | 原始大数据集构建与训练观测 | 数据主体内存不随累计 token 数线性增加，满足实测后制定的节点预算，并完成训练验证 |
| 后续：扩展 | 缓存构建断点续建、额外输入格式、eval 磁盘模式与分批词频 | 每个组合独立完成内存与行为回归后再启用 |

具体检查：

1. 对同一批数据逐条比较 ID、input IDs、mask、长度、必要元数据和过滤结果，覆盖 user-only decode prompt、空对话、长文本截断、思考标签及预 tokenized 验证。
2. 比较多个 seed、epoch、length group 和恢复 skip 的完整样本顺序。测试总样本数不能整除批次大小、reload 时还有旧轮尾部和在途读取的情况。
3. 注入慢首条、慢磁盘和慢推理，验证任务队列、重排结果与读缓存达到上限后停止增长；ready 为空但读取在途时不得标记 EOF。检查 controller 派发与状态查询不会被磁盘读取阻塞，读取错误能终止所有等待方。
4. 验证缓存构建被中断、未完成 manifest、损坏分片、源数据变化和并发启动，防止复用半成品或错误数据。断点续建功能上线前另行验证输入偏移和 ID 一致性。
5. 用 10 万、50 万、205 万条数据观察预处理窗口、worker 内存、controller 匿名内存和节点总量。允许紧凑索引随样本数增长，不能允许文本或 token 主体线性常驻。
6. 使用新建缓存和缓存命中两种路径完成 DSpark train-with-decode 检查，覆盖正常训练、epoch 切换、checkpoint 恢复、最终保存和退出清理；记录推理供给速度与 GPU 等待，评估磁盘读取的吞吐影响。

阶段一、二需要一起接入才能达到“数据主体内存不随全量 token 增长”的长期目标。仅为跑通本次语料，减少加载峰值与冗余文本可能已经足够，但目前没有实测依据，不将它判定为必然成功或必然失败。只换 Arrow 文件而仍全量物化，不能达到长期目标。

**9. 当前已做的设计验证**

- 当前环境已有 `datasets` 和 PyArrow 25.0.1。若实现直接调用 PyArrow，应在项目中明确声明这一直接依赖。
- 使用临时文件验证了 Arrow IPC 的 record batch 增量写入、mmap 后选择性读取，以及按请求顺序恢复样本；临时文件已自动清理。
- 对当前源码的 `_prepare_dataset()` 做了 180 组对照，覆盖 shuffle 开关、seed、epoch、长度并列、尾部分组、skip 越界；索引方案生成的单轮顺序一致。此检查不等于完整异步训练顺序和 checkpoint 恢复已经验证。
- 直接执行从当前源码抽取的 `reload_dataset()`、`add_dataset()` 方法，确认旧尾部会保留在新一轮样本之前，并确认队列中的 `prompt` 与原样本是共享引用。
- 对当前 Python 的 `Pool.imap()` 做慢首条实验，观察到 39 个后续结果积压；显式窗口为 4 时只积压 3 个，支持提交窗口必须显式管理的要求。
- 检查当前 Ray 安装源码与常量，确认该同步 controller 默认方法并发度为 1；不能用推理端异步 refill 推断 controller 的同步磁盘读取不会阻塞派发。
- 使用约 128 MiB 的未压缩 Arrow 合成缓存，包含 128 条、每条 131072 个 token。两条读取路径均完整处理 16,777,216 个 token，checksum 一致。只保留两个独立数值数组时，处理到 32/64/128 MiB 时匿名内存约为 61.92/61.92/61.93 MiB；转成 Python 列表并全量保留时，约为 191.16/351.66/672.62 MiB。前者 RSS 仍从约 148.82 增至 244.76 MiB，说明映射页也须核算。这个对照验证有界数值读取与全量 Python 物化的区别，不是当前旧加载器的端到端性能对比，因为旧路径已使用 NumPy。
- 尚未测量真实语料的 tokenizer 峰值、AFS 随机读取吞吐、模型/传输共存内存或全量训练。合成实验不支持对实际任务承诺固定内存数字。
