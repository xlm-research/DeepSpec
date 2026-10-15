# Loop/mHC 单机图文 DSpark 训练规约

Status: ready-for-agent

Synthesized on 2026-10-15 from the recorded Q1–Q9 decisions and final overall approval. Implementation, GPU preflight, and the complete training pass were accepted on 2026-10-15. The triage label records specification readiness; the completed execution state and its evidence remain recorded below.

## Problem Statement

The user wants to train the existing Loop/mHC DSpark draft model on the entire agreed image-text dataset with minimal changes to the existing training system, and inspect local loss and acceptance-probability PNGs. At the start of this effort, the Loop recipe supported text training while the working full-data image-text recipe targeted another model. Neither recipe alone established that Loop vision input, historical reasoning, execution-layer features, and a complete data pass worked together.

Here, **source conversation** means **训练源会话**, counted by original source record even when message contents match; **assistant supervision segment** means **assistant 监督段**; **image-text context** means **图文上下文**; and **DSpark supervision window** means **DSpark 监督窗口**. The frozen **target model (目标模型)** supplies reference predictions and representations to the trainable **DSpark draft model (DSpark 草稿模型)**. **Estimated training acceptance probability** means **训练接受概率估计**, as defined in the project glossary.

The dataset contains long conversations, historical reasoning and tool calls, later system messages, and local, remote, and GIF images in user or tool messages. A successful run must preserve the agreed supervision, account for every original source and assistant segment, and provide precise metrics. A target prefill completion token or a prediction-accuracy scalar cannot establish actual speculative-decoding acceptance.

## Solution

Provide a Loop/mHC single-node image-text recipe that reuses the existing full-data workflow: prepare all 327 source conversations, validate actual image-text capture and training on representative windows, and train a newly initialized five-layer DSpark draft for one complete pass. Keep the target model, vision encoder, and shared target embedding and LM head frozen.

Prepare bounded supervision windows that retain all 28,872 existing assistant supervision segments, including reasoning and tool calls. Assign each segment to exactly one supervised window, preserve the agreed context and media semantics, and prove coverage after successful optimizer updates.

Deliver two local PNGs: weighted total loss with its CE, L1, and confidence components, and estimated training acceptance probability derived from the measured distribution L1. Preserve exact step metrics, logs, effective configuration, preparation and coverage records, and a complete final checkpoint. The original approved contract and execution history remain under Comments.

## User Stories

1. As a training user, I want a dedicated Loop/mHC image-text recipe, so that I can run the agreed workflow through a documented entry point.
2. As a training user, I want the existing Loop target checkpoint used, so that the draft learns from the agreed target model.
3. As a training user, I want the current five-layer DSpark draft architecture retained, so that the run stays compatible with the established Loop feature contract.
4. As a training user, I want the target model and vision encoder frozen, so that optimization changes only the draft parameters.
5. As a training user, I want shared target embeddings and the LM head frozen, so that the reference prediction space remains stable.
6. As a training user, I want fresh trainable draft parameters and optimizer state, so that earlier training does not influence the requested run.
7. As an operator, I want four target-inference GPUs and four draft-training GPUs on one node, so that the workflow fits the agreed eight-GPU setup.
8. As a training user, I want all 327 original source records included without content deduplication, so that matching conversations still count as separate source records.
9. As a dataset owner, I want the original JSONL and target checkpoint left intact, so that preparation and training preserve their source inputs.
10. As a reviewer, I want each prepared window traceable to source records, messages, and assistant segments, so that I can audit coverage independently.
11. As a training user, I want all 28,872 existing assistant supervision segments included, so that earlier answers contribute alongside later answers.
12. As a training user, I want historical reasoning content preserved and supervised, so that the draft learns the reasoning present in the source conversations.
13. As a training user, I want assistant tool-call content and top-level tool definitions preserved, so that tool-use supervision retains its meaning.
14. As a training user, I want user and tool messages retained as image-text context, so that they condition the assistant predictions.
15. As a training user, I want system, user, tool, visual, and generated verification-token positions excluded from answer loss, so that supervision applies only to the selected existing assistant segments.
16. As a training user, I want each original assistant segment supervised in exactly one window per pass, so that window overlap does not multiply its training credit.
17. As a training user, I want overlapping historical answers available as context without additional loss, so that windowing retains useful history.
18. As a training user, I want each window to exclude messages later than its selected answers, so that predictions cannot use future context.
19. As an operator, I want window lengths measured with the actual Loop tokenizer, template, processor, and image expansion, so that prepared requests fit the target's real input limit.
20. As an operator, I want one token reserved within the 128K request limit for prefill verification, so that the completion check can finish legally.
21. As a training user, I want removable history omitted and necessary user/tool text prefixes shortened before assistant content is affected, so that selected supervision segments stay complete.
22. As a reviewer, I want every context omission and text-shortening range recorded, so that I can reconstruct the prepared input.
23. As an operator, I want preparation to identify the source record when a complete assistant segment cannot fit, so that an unusable record cannot disappear silently.
24. As a training user, I want later system messages kept in their original role and order, so that earlier answers do not receive later instructions.
25. As a training user, I want images in both user and tool messages supported, so that all agreed image-text context can reach the target.
26. As a training user, I want retained image references paired with their markers in message order, so that each image remains associated with the correct context.
27. As an operator, I want relative image references resolved through the existing dataset media root, so that the original data needs no path rewrite.
28. As an operator, I want HTTPS images cached and actually decoded, so that preparation proves that their visual content is usable.
29. As a training user, I want GIFs interpreted as static first-frame images, so that the run follows the existing image semantics.
30. As a training user, I want existing image-orientation and transparency conversion behavior retained, so that image handling remains consistent.
31. As an operator, I want media failures to identify the original source record, so that I can diagnose failed preparation.
32. As an operator, I want incompatible source, template, tokenizer, processor, or preparation caches rejected, so that stale inputs cannot change the training contract.
33. As a training user, I want the Loop target to consume real visual input through V2 direct capture, so that image-text training uses visual representations.
34. As a training user, I want the agreed execution-layer features and separate post-norm final state preserved, so that draft conditioning and target predictions remain correctly defined.
35. As a reviewer, I want expanded tokens, supervision masks, auxiliary features, and final states aligned, so that loss cannot be applied to the wrong positions.
36. As a training user, I want one complete pass over all prepared windows, so that the run reaches the agreed coverage endpoint.
37. As an operator, I want optimizer and learning-rate horizons derived from the actual window count, so that the run includes the final partial batch.
38. As an operator, I want delayed inference results awaited and reconciled by window identity, so that asynchronous work does not trigger duplicate data loading.
39. As a reviewer, I want coverage credited only after a successful optimizer update, so that queued or failed work cannot count as trained data.
40. As a training user, I want zero-loss dummy entries to complete the distributed tail without adding supervision or coverage, so that every real tail window is trained once.
41. As a training user, I want the existing DSpark block, anchor-sampling, and loss settings retained, so that the adaptation does not change the training algorithm.
42. As an operator, I want a command preview that does not start training, so that I can inspect the selected workflow.
43. As an operator, I want preparation-only execution with an effective configuration, so that I can inspect the prepared contract before GPU work.
44. As an operator, I want preflight-only execution, so that I can validate the target and draft workflow independently of a full pass.
45. As a reviewer, I want representative preflight windows selected from actual media and preparation metadata, so that coverage includes real GIFs, cached remote images, and tool images.
46. As a reviewer, I want a two-step preflight with 17 distinct real windows and a padded tail, so that parameter updates and tail accounting are verified together.
47. As a maintainer, I want the existing Loop text-only path checked, so that image-text support preserves the established behavior.
48. As a training user, I want the full pass to start with fresh state after preflight, so that preflight optimization does not become part of the requested training run.
49. As an operator, I want target capture, training, and transfer-storage capacity checked against the real windows, so that the full workflow can retain its required data.
50. As a training user, I want a local loss PNG containing weighted total loss and all three components, so that I can inspect the optimization behavior.
51. As a training user, I want a separate local estimated-training-acceptance PNG, so that I can inspect distribution overlap throughout training.
52. As a reviewer, I want the acceptance plot labeled with its training-context, position-weighted, pre-update meaning, so that I can interpret the statistic correctly.
53. As a reviewer, I want the estimate derived from measured L1 before its loss coefficient is applied, so that the reported probability has the agreed definition.
54. As an operator, I want exact finite metrics for every successful optimizer step, so that the curves cover the entire completed run.
55. As an operator, I want missing, invalid, or out-of-range acceptance inputs reported as errors, so that a misleading substitute cannot appear in the plot.
56. As a training user, I want both PNGs reproducible from exact saved metrics, so that I can regenerate them after training.
57. As an operator, I want controller and actor logs, TensorBoard scalars, effective configuration, and input identities saved locally, so that I can inspect and reproduce the run.
58. As a reviewer, I want source, segment, and window coverage reconciled across the inference and training pipeline, so that completion is supported by evidence.
59. As a training user, I want a final checkpoint with model, optimizer, scheduler, RNG, and iteration state, so that the completed training state is preserved.
60. As an operator, I want failed-attempt diagnostics retained and retries written to separate outputs, so that incomplete runs remain distinguishable from successful runs.
61. As a maintainer, I want the existing image-text recipe checked after shared changes, so that adapting Loop does not break the earlier target.
62. As a maintainer, I want the existing full-data workflow reused with small, necessary extensions, so that the adaptation avoids duplicating the training system.
63. As a training user, I want launch modes and output locations documented, so that I can operate the recipe and locate its deliverables.
64. As a reviewer, I want full-window coverage and estimated training acceptance described with their actual limits, so that they do not imply exhaustive anchor sampling or measured speculative-decoding performance.

## Implementation Decisions

1. **Reuse the existing workflow.** Add the Loop example's configuration, thin launcher, and usage documentation. Reuse the shared image-text full-data driver, supervision-window preparation, full-data renderer, target prefill interface, complete-pass controller, DSpark trainer, Mooncake transport, and local metric recorder. Parameterize the base model configuration, V1/V2 selection, and legacy text baseline where needed. Keep shared fixes within the approved minimal-change scope.

2. **Retain the agreed models and deployment.** Use the Loop/mHC ckpt-225-bf16 target and the current five-layer DSpark draft: hidden width 4096, Q/KV heads 16/4, and head dimension 256. Freeze the target, vision encoder, and shared target embedding and LM head. Initialize trainable draft parameters from scratch without loading previous draft, optimizer, or scheduler state. Use one eight-GPU node with target TP=4, PP=1 and four FSDP training ranks.

3. **Preserve the Loop execution and feature contract.** Enable actual vision input with vLLM V2 direct capture. Retain shared physical weights and independent execution-layer inference state. Collect zero-based execution layers 3, 27, 51, 75, and 78, corresponding to auxiliary capture slots 4, 28, 52, 76, and 79. Final logical slot 80 separately carries the actual post-norm state entering the target LM head. Do not apply target final normalization again during training. Narrowly remove the model's text-only restriction while retaining its other constraints.

4. **Identify sources and supervision explicitly.** Process all 327 original source records without content deduplication. Preserve all 28,872 assistant supervision segments, their answer bodies, reasoning, tool calls, and top-level tools. Record source identity and mappings between original source rows, messages, segments, and prepared windows. Preserve the original dataset and checkpoint.

5. **Prepare windows with real expanded lengths.** The total request limit is 131072 tokens; the input limit is 131071 after reserving one verification token. Measure with the Loop tokenizer, adapted template, processor, and actual image expansion. Assign each original assistant segment to one supervised window. Overlapping historical assistant content is context only, and future messages are excluded. Omit removable history first, then apply the existing user/tool text-prefix shortening rules with recorded ranges. Keep each supervised assistant segment intact and fail with its source identity if it cannot fit.

6. **Preserve context and media semantics.** Keep later system messages in their original role and order. Resolve relative images using the established dataset media root; support images in user and tool messages, ordered with their retained markers. Cache HTTPS image bytes and validate actual decoding. Use GIF first frames and the existing orientation and transparency conversions. Report failed image preparation against its original source record.

7. **Adapt rendering and cache identity.** Preserve historical reasoning in the repository's template/renderer adaptation instead of changing checkpoint files. Recompute the supervision mask on actual expanded token IDs, excluding system/user/tool content, visual positions, overlapping context, and the extra verification token. Require position and length agreement between tokens, masks, five auxiliary features, and final state. Include source identity, preparation semantics, and relevant model/template/tokenizer/processor fingerprints in cache compatibility. Feature order, slot roles, and normalization remain compatibility constraints under ADR-0002.

8. **Complete and account for one data pass.** Let M be the number of successfully prepared real windows. Use a global batch of 16, consisting of four ranks, microbatch one, and four accumulation steps. Derive both optimizer and learning-rate horizons as ceil(M / 16). Credit real window, source, and segment coverage only after successful optimizer commits. Reconcile queued, submitted, received, dispatched, and committed identities; wait for in-flight work instead of reloading sources. Pad the tail with zero-loss dummy entries that contribute neither supervised denominators nor real-data coverage.

9. **Retain DSpark optimization semantics.** Keep seven-token blocks, 512 anchors, position-decay parameter 4.0, and CE/L1/confidence coefficients 0.1/0.9/1.0. A complete pass covers all windows and assigned assistant segments; anchor sampling does not guarantee that every supervised token is sampled during that pass. No DSpark algorithm or core trainer redesign is required.

10. **Record precise loss and acceptance metrics.** After each successful optimizer step, save its exact local metrics and update both PNGs. Weighted total loss is 0.1 × CE + 0.9 × L1 + confidence; save the coefficients with the metrics. Estimated training acceptance probability is A = 1 − L1 / 2, using the measured distribution L1 before multiplication by 0.9. Equivalently, A is the valid-position and position-decay-weighted mean of the target/draft probability overlap, with numerator and denominator aggregated across accumulation microbatches and training ranks in the existing manner.

11. **Keep metric timing and interpretation explicit.** Plot true optimizer steps on the horizontal axis and acceptance probability on a 0–100% vertical scale. The values come from that step's pre-update training forward pass and are recorded after the update succeeds. Label the training context and position weighting. Keep the L1 coefficient positive, reject missing, nonfinite, or materially invalid values, and retain exact records for offline replotting. Prediction accuracy and simulated acceptance length retain their own meanings. Finalize only after metrics cover every successful step from one through the final step.

12. **Use the recorded resource defaults.** Retain learning rate 1e-4, maximum gradient norm 0.5, zero warmup, seed 42, and the existing optimizer and scheduler types. Use FSDP FULL_SHARD, gradient checkpointing, and expandable CUDA segments. Run the target in BF16 eager mode with FP32 SSM state, 8192-token chunked prefill, one request at a time, and prefix caching disabled. Derive the image limit from prepared windows. Start with target GPU memory fraction 0.7, Mooncake global storage 256 GB, and local buffer 16 GB; verify capacity against real features, queued results, and in-flight capture. Record justified resource adjustments without changing supervision, model semantics, length limits, pass coverage, or metric definitions.

13. **Expose the existing operational stages.** The default launcher performs preparation, native image-text preflight, the legacy Loop text check, two-step draft-training preflight, and a separate fresh full pass. Provide command-preview, preparation-only, and preflight-only modes. Use independent cache, output, and log locations; save effective configuration and the data/model/template/feature identities. Check actual device and memory availability before execution.

14. **Preserve reviewable artifacts and failure state.** Deliver the prepared manifest, effective configuration, both PNGs, exact local metrics, TensorBoard scalars, raw controller and actor logs, coverage report, and complete final checkpoint. Reconcile model, optimizer, scheduler, RNG, and final iteration state. Preserve failed-attempt diagnostics and incomplete coverage; use separate retry outputs. The specification records successful execution separately from readiness for implementation.

These decisions retain ADR-0001's parameter/state ownership, ADR-0002's execution-layer and final-state contract, and ADR-0003's all-assistant image-text supervision. They do not reopen those architectural decisions. The numeric resource and optimizer defaults were adopted in the approved overall contract; they were not separate additional answers to Q1–Q9.

## Testing Decisions

- **Primary seam:** use the existing public full-data workflow, from original source conversations and the Loop configuration through preparation, GPU preflight, optimizer commits, and saved artifacts. Q8 and the recorded final overall approval already include this boundary. Retain that confirmed seam; no new production test interface is needed.

- **Good tests check observable behavior:** compare prepared conversations and supervision to original message identities and independent assistant-content boundaries; inspect returned tokens/features and saved coverage, metrics, and checkpoints. Expected masks must not be calculated solely by the renderer under test. Verify the acceptance formula against independently calculated overlap of small target/draft distributions. Avoid assertions about private call order or helper implementation.

- **Modules covered through existing interfaces:** supervision-window preparation and the full-data renderer, effective training configuration, Loop target capture and transfer, complete-pass scheduling and tail accounting, and the local metric recorder and replotting interface. Use focused regressions for edge cases while retaining the full workflow as the acceptance boundary.

- **Prior art:** the existing image-text full-data tests exercise preparation, expanded masks, later system messages, cached images/GIFs, effective configuration, and preflight selection. The full-data-pass tests exercise in-flight results, duplicate rejection, padded ranks, successful-commit coverage, and delayed metric flushing. Loop capture tests exercise execution-layer outputs and the actual LM-head input. Existing metric tests independently compare weighted probability overlap and verify replotting and invalid-input rejection. Native GPU prefill and distributed training preflight supply runtime acceptance evidence.

- **Representative preflight:** select 17 distinct real windows using actual image references and manifest mappings. Cover multi-image and tool-image inputs, real GIFs, cached remote images, later system messages, historical reasoning, context shortening, overlapping unsupervised history, and the maximum expanded input length. A textual mention of a GIF filename does not satisfy media coverage. Independently verify that the target consumes vision, returned features are finite and aligned, and the verification token is excluded from supervision.

- **Training acceptance:** the 17-window preflight requires two successful optimizer updates; its second batch contains one real window and 15 zero-loss dummy entries. Check actual draft parameter changes, finite losses, correct supervised denominators, and exact coverage. Start the full pass with separate fresh training state. Retain the original Loop text regression and appropriate tests for the reused image-text recipe. Static checks alone do not establish GPU acceptance.

| Acceptance ID | Observable passing condition |
| --- | --- |
| AC-01 Configuration | Command preview and effective configuration select the agreed Loop target, five-layer draft, V2 visual capture, 4+4 deployment, 128K limit, all-assistant supervision, and fresh training state. |
| AC-02 Preparation | Independent comparison to original messages reconciles all 327 source records and 28,872 segments exactly once; supervised content is intact, context changes are traceable, images decode, and expanded windows fit. |
| AC-03 Image-text preflight | Representative native captures consume actual images, return finite aligned features with correct auxiliary/final roles, preserve independently checked supervision, and exclude the generated verification token. |
| AC-04 Training preflight | Two optimizer commits consume all 17 distinct real windows with 15 tail dummy entries; draft parameters update, losses and denominators are valid, and formal training starts separately. |
| AC-05 Full pass | Exactly ceil(M / 16) successful updates cover all M windows, 327 source records, and 28,872 segments, with matching pipeline identities and no missing, duplicate, failed, or outstanding work. |
| AC-06 Metrics and artifacts | Every successful step has finite exact metrics; both PNGs agree with those records and can be regenerated; acceptance uses L1 before its loss coefficient; final checkpoint components reconcile at the final iteration. |
| AC-07 Regression | The original Loop text path and affected existing image-text behavior pass appropriate regressions, including template preservation, expanded masks, cache compatibility, V2 configuration, media selection, and metric semantics. |

## Out of Scope

- Training the target model, vision encoder, or shared target embedding and LM head.
- Editing the original source JSONL or target checkpoint, or deduplicating source records by conversation content.
- Replacing the DSpark algorithm, redesigning the core trainer, or changing the agreed feature contract.
- Loop speculative-decoding integration, actual generated-token acceptance rates, decoding speedup benchmarks, or an independent generation-evaluation dataset and schedule.
- Multi-epoch training, hyperparameter search, monotonic-loss guarantees, or exhaustive per-token anchor coverage.
- Video/time-series interpretation of GIFs or broader media-format expansion.
- A new checkpoint-resume acceptance exercise beyond preserving the complete final training state.
- A web dashboard or hosted reporting service; the requested reports are local PNGs, metrics, and logs.

## Further Notes

The scope comes from the local record of Q1–Q9 and the user's final “确认”. Q9 explicitly chose estimated training acceptance probability. The recorded approval already covers the preparation/preflight/full-pass test seam; no new seam or unresolved requirement is introduced by this synthesis. The original investigation, confirmations, implementation updates, failed-attempt notes, and final acceptance record are retained under Comments.

The completed Loop run prepared **595 windows** and performed **38 optimizer steps**, with **13 zero-loss tail entries**, from all **327 source records** and **28,872 assistant segments**. Its longest expanded input was 131071 tokens and its maximum images per window was 22. These are observed results of the accepted run; future compatible runs derive M, their horizon, and image limit from preparation rather than hard-coding these observations.

The existing independent acceptance report records complete pipeline reconciliation, finite metrics for all 38 steps, draft updates on all four training ranks, the agreed weighted-loss and acceptance identities, TensorBoard agreement, and consistent final checkpoint state. Weighted total loss moved from 3.718174922466278 to 2.7306162238121034; estimated training acceptance probability moved from 0.24548768997192383% to 1.5488207340240479%. These are training-context observations, not standalone answer-generation or speculative-decoding measurements.

The accepted evidence was inspected for this documentation synthesis; no new training or test run is claimed. The requested ready-for-agent triage label does not reset the recorded completed implementation or authorize a duplicate run.

| Reference | Purpose |
| --- | --- |
| [Glossary](../../GLOSSARY.md) | Shared domain vocabulary and source-record counting semantics. |
| [ADR-0001](../../docs/adr/0001-loop-physical-weights-execution-state.md), [ADR-0002](../../docs/adr/0002-loop-mhc-dspark-capture-contract.md), [ADR-0003](../../docs/adr/0003-multimodal-dspark-all-assistant-supervision.md) | Existing parameter/state, capture, and supervision decisions. |
| [Loop image-text example](../../examples/qwen3.5-loop-mhc-image-text-dspark-1-node/README.md) | Launch modes, agreed model/data locations, media preparation conventions, and artifact index. |
| [Source dataset](../../mydata/image-text-327.jsonl) | The agreed 327-record input. |
| [Image-text workflow tests](../../tests/test_image_text_full_data.py), [complete-pass tests](../../tests/test_full_data_pass.py), [Loop capture tests](../../tests/test_loop_capture.py) | Existing test interfaces and regression prior art. |
| [Preparation audit](../../outputs/loop-image-text-20261014/independent_preparation.json), [input identity audit](../../outputs/loop-image-text-20261014/input_identity_audit.json) | Original-message coverage and source/model identity checks. |
| [Preflight selection](../../outputs/loop-image-text-20261014/preflight/selection_audit.json), [native capture acceptance](../../outputs/loop-image-text-20261014/preflight/prefill_acceptance.json), [training preflight acceptance](../../outputs/loop-image-text-20261014/preflight/training/independent_acceptance.json) | Real-media coverage, capture alignment, and padded-tail training evidence. |
| [Full-pass coverage](../../outputs/loop-image-text-20261014/coverage.json), [independent acceptance](../../outputs/loop-image-text-20261014/independent_acceptance.json) | Completed-run coverage, updates, metrics, and checkpoint reconciliation. |
| [Curve regeneration audit](../../outputs/loop-image-text-20261014/curve_regeneration_audit.json), [loss PNG](../../outputs/loop-image-text-20261014/loss.png), [acceptance PNG](../../outputs/loop-image-text-20261014/acceptance.png) | Exact-record replotting and the requested local figures. |
| [Final checkpoint](../../outputs/loop-image-text-20261014/checkpoints/iter_0000038/) | Preserved final model, optimizer, scheduler, and RNG state. |

## Comments

### 2026-10-14–15 — Original scope and conversation record

The original record is preserved below. Statements about pending implementation, current restrictions, unverified paths, or temporary resource availability describe their recorded stage. The current completed state is summarized above and in the dated final acceptance comment.

记录日期：2026-10-14；最终验收日期：2026-10-15。用户已确认 Q1–Q9 及最终整体规约，适配、真实预检和一轮全量训练均已完成，AC-01 至 AC-07 验收通过。确认过程、执行记录及产物证据见文末。

#### 用户明确提出的目标

- 在现有代码基础上，以尽量少的代码改动适配图文训练。
- 示例入口位于 `examples/qwen3.5-loop-mhc-image-text-dspark-1-node/`。
- 数据源为 `mydata/image-text-327.jsonl`。
- 先讨论并确认需求规约，再进入实现；用户已完成最终确认。
- 用户在确认 Q4–Q8 时追加本地接受率 PNG，并通过 Q9 明确选择训练接受概率估计。

#### 需求合同

##### 模型与优化边界

沿用目标 checkpoint `/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16` 和 `configs/draft_models/qwen3_5_loop_mhc_dspark.json` 的当前五层 DSpark 结构：隐藏宽度 4096，Q/KV heads 为 16/4，head dimension 为 256。目标模型、视觉编码器和共享目标 embedding/LM head 保持冻结；草稿可训练参数从零初始化，不加载旧模型、optimizer 或 scheduler 状态。

采用单机八卡，四卡 TP=4、PP=1 运行目标模型，四卡 FSDP 训练草稿。目标继续使用 vLLM V2 与 direct capture；启用实际视觉输入并保留 Loop 的参数共享和独立执行层状态。辅助执行层保持 `[3, 27, 51, 75, 78]`，对应采集槽 `[4, 28, 52, 76, 79]`；独立 final 槽为 80，内容为实际进入目标 LM head 的 post-norm 状态，训练侧不得再次应用目标 final norm。

##### 数据、窗口和监督

1. 使用原始 `mydata/image-text-327.jsonl` 全部 327 条源记录，不额外按内容去重。原始 JSONL 不修改，准备产物使用独立缓存；记录源文件身份与源行、消息、监督段和窗口的对应关系。
2. 全部 28,872 个已有 assistant 监督段参与监督，完整保留正文、reasoning 和 tool calls。修正 Loop 模板省略历史 reasoning 的行为，并保留顶层 tools。system/user/tool 内容、视觉位置和额外生成的一个验证 token 不承担回答 loss。
3. 总请求上限为 131072 tokens，实际图片展开后的输入最多 131071 tokens。用 Loop 的实际 tokenizer、适配后的模板和 processor 测量窗口，不能复用另一模型的窗口计数或占位符长度。
4. 每个原 assistant 段在一轮中恰好分配到一个受监督窗口。重叠历史只作上下文；窗口不能包含目标回答之后的未来消息。先省略可移除的历史，仍超限时沿用现有规则缩短 user/tool 文本前缀并记录范围；被监督的 assistant 段保持完整。无法容纳完整监督段时明确失败并报告源记录，不静默跳过。
5. 保留中途 system 的原角色和顺序。图片按保留消息顺序与标记配对，支持 user 和 tool 中的图片。相对图片路径沿用 `/mnt/afs_toolcall/juxiaolong1/Data/draftData` 根目录；HTTPS 图片先缓存再实际解码，GIF 按静态首帧处理，沿用现有方向和透明背景转换规则。图片失败需定位到源记录。
6. 在真实视觉展开后的 token IDs 上重算监督 mask，并校验 token、mask、五份辅助特征和最终状态的位置及长度一致。模板和窗口语义变化须使不兼容缓存失效。

##### 完整训练一轮

全量准备成功后得到 M 个真实监督窗口，完整训练一轮。窗口仅在对应 optimizer step 成功后计入覆盖；提交、接收和训练消费按窗口身份核对，禁止靠重复加载填补异步推理缺口。尾批沿用零 loss dummy 补齐，dummy 不贡献监督分母、源记录、监督段或窗口覆盖。

沿用现有 DSpark 的七 token block、512 anchors、位置衰减参数 4.0，以及 CE/L1/confidence 权重 0.1/0.9/1.0。一轮覆盖指所有窗口和监督段都进入训练，不表示每个监督 token 都被 anchor 抽样选中。

##### 本地曲线和指标

每个成功 optimizer step 保存精确本地指标并更新两张独立 PNG，结束时核对指标完整覆盖 step 1 到最终步。

- `loss.png`：加权总 loss 及 CE、L1、confidence 分项；总 loss 为 `0.1 × CE + 0.9 × L1 + confidence`，权重同时保存到运行记录。
- `acceptance.png`：**训练接受概率估计**，使用 `A = 1 - train/l1_loss / 2`，横轴为真实 optimizer step，纵轴为 0–100%。标题或图注明确其来自训练上下文、按位置衰减权重平均。

这里的 L1 是尚未乘以损失系数 0.9 的分布 L1 指标。设 p/q 是同一训练位置的目标/草稿分布，w 是有效监督与既有位置衰减权重，则 A 等于 `sum(w * sum(min(p, q))) / sum(w)`；按现有方式先跨累积 microbatch 和训练 rank 汇总分子分母。数据来自该步更新前的训练前向，在更新成功后记录，不是更新后固定验证集的重测。

本次保持 L1 系数为正。缺失、非有限或明显超出合法范围的指标应明确报错，不以常数或其他准确率替代。保留足以离线重绘两张图的精确记录；`avg_acc` 和模拟接受长度不改名为真实接受率。

##### 实施默认值

以下是为最小化适配而选择的技术默认值，纳入整体规约核对；它们不是用户逐项回答的新增要求。

| 项目 | 默认值及推导规则 |
| --- | --- |
| 全局 batch | 复用图文全量流程的 16：四个训练 rank × microbatch 1 × 累积 4 |
| 训练终点 | `ceil(M / 16)` 个 optimizer steps；学习率调度总步数同步推导，不保留旧 smoke 的固定终点 |
| 优化参数 | 学习率 `1e-4`、最大梯度范数 `0.5`、warmup 比例 0、seed 42，保留已有 optimizer/scheduler 类型 |
| 训练内存 | 沿用 FSDP FULL_SHARD、gradient checkpointing 和 `expandable_segments` |
| 目标运行 | BF16、eager、FP32 SSM、8192-token 分块 prefill、每次一个请求、关闭 prefix caching |
| 图片上限 | 根据本次成功准备的窗口推导；不沿用三图 smoke 上限，也不固定旧模型的二十图结果 |
| 目标显存比例 | 初始沿用图文全量流程的 0.7，为视觉计算与特征暂存留空间；根据真实预检可调 |
| 传输存储 | 初始沿用全量流程的 Mooncake 256GB、local buffer 16GB；按 Loop 实际特征大小、排队和在途捕获核查容量 |
| 运行目录 | 新示例使用独立 cache/output/log 目录，保存解析后的配置和数据/模型/模板/特征契约身份 |

不改变监督、模型、长度上限、完整遍历和指标定义的资源参数可依据真实预检调整并记录；不能以漏数据、裁掉监督内容或复用不兼容状态解决资源问题。运行前检查实际设备及内存可用性。

#### 最小实现范围

- 新目录提供配置、薄启动入口和使用说明，默认完成准备、预检及一轮完整训练；提供命令预览、仅准备数据和仅预检的入口。
- 复用已有图文窗口准备、renderer、完整遍历调度、尾批处理、DSpark trainer、Mooncake 和本地指标记录。仅参数化影响复用的模型配置、V1/V2 和文本回归基线等固定项，避免复制整套训练流程。
- 调整 Loop 的纯文本限制以允许本规约的图文前向，保留其余既有模型约束；真实验证视觉权重加载与 V2 图文采集。
- 在仓库的模板/renderer 适配层完整保留历史 reasoning，原始 checkpoint 文件不修改；增加相应缓存版本和内容核验。
- 扩展已有本地绘图与重绘能力，输出 Q9 确认的训练接受概率估计。无需接入 Loop 推测解码或独立生成评估。

目前没有证据需要修改 DSpark 算法或主体训练器。如真实执行揭示必要缺口，仍遵循 Q3 的小范围修复边界，不把尚未运行的路径标为已通过。

#### 验收合同

| 编号 | 可观察的通过条件 |
| --- | --- |
| AC-01 配置 | 启动预览和解析配置明确显示正确 Loop 目标、草稿结构、V2、视觉启用、4+4 布局、128K、全部 assistant 监督及新训练状态 |
| AC-02 数据 | 独立对照原始消息确认 327 条源记录、28,872 个 assistant 段各分配一次；内容完整、上下文裁剪可审计、图片可解码且展开后窗口均合法 |
| AC-03 图文预检 | 代表性真实窗口覆盖多图、tool 图片、GIF、远程图片缓存、中途 system、历史 reasoning、上下文缩短及最大展开长度；目标实际消费视觉输入，取回的特征有限且角色/行数正确，独立核对原消息监督范围，额外验证 token 不进入监督 |
| AC-04 训练预检 | 复用全量流程的 17 个不同真实窗口验证两步更新，第二步包含一个真实窗口及十五个零 loss dummy；真实参数更新、loss 有限、尾批覆盖和分母正确。正式训练另起新状态 |
| AC-05 全量完成 | `ceil(M / 16)` 次成功更新后，全部 M 个窗口、327 条源记录、28,872 个监督段对账完整，无缺失、重复监督或未完成在途结果 |
| AC-06 指标与产物 | 每一步的精确指标完整且有限，两张 PNG 与记录一致、可重绘；接受概率使用正确的未加权 L1 分项并检查取值；完整 checkpoint 的模型、optimizer、scheduler、RNG 和最终步数一致 |
| AC-07 回归 | Loop 原纯文本路径及复用改动影响到的图文流程通过适当回归；模板、mask、缓存、V2 配置和指标的针对性检查通过 |

自动化验证沿用现有公共工作流接口；监督期望以原消息内容和独立边界确定，接受概率可用独立的小分布重叠计算核验，不以实现函数自算的值作为唯一预期。真实 GPU 预检与训练承担运行验收，静态检查不替代它们。

交付包括新示例、README、准备 manifest、有效配置、`loss.png`、`acceptance.png`、精确 `metrics.jsonl`、本地 TensorBoard 指标、训练及 actor 原始日志、覆盖报告和完整最终 checkpoint。失败保留诊断及不完整覆盖，重试使用独立输出，不能覆盖旧产物或宣称完整验收。

#### 范围边界

本次不训练目标模型或视觉编码器，不修改原始 checkpoint/JSONL，不接入真实推测解码接受率或加速比评测，不增加多轮训练或超参数搜索，也不承诺 loss 单调下降或每个监督 token 在单轮都被抽中。最终 checkpoint 需完整保存，但本次不新增独立的恢复训练验收。

#### 确认前的调查依据

本节保留确认前的代码与数据调查；实施后的验证结果见文末执行记录。

##### 可复用的两条基线

确认前，指定的新示例目录存在但为空。可复用的基线分别是：

- [Loop/mHC 单机 DSpark](../../examples/qwen3.5-loop-mhc-dspark/README.md)：目标模型四卡 TP=4，草稿四卡 FSDP；4096 隐藏宽度、五层草稿，attention 为 16/4/256。当前结构的纯文本训练已在 [2026-10-12 验收记录](../../docs/qwen3_5_loop_mhc_8gpu_revalidation_plan.md)中验证。
- [既有单机图文训练](../../examples/qwen3.8-27b-image-text-1node/README.md)：已实现图片准备、监督窗口、完整数据遍历、尾批处理及本地指标。其[独立验收产物](../../outputs/image-text-full-20261014/independent_acceptance.json)记录了另一目标模型的 327 条源记录、610 个窗口、28,872 个 assistant 监督段和 39 次更新。

两条链路均优化 DSpark 草稿，目标模型保持冻结。Loop 的特征契约为执行层 `[3, 27, 51, 75, 78]`、独立 final 槽 80、post-norm；不能沿用另一图文模型的特征层、宽度或 pre-norm final 槽 64。具体约定见 [ADR-0002](../../docs/adr/0002-loop-mhc-dspark-capture-contract.md)。

旧图文验收不能证明 Loop 图文训练已经可用。Loop 的窗口数、训练步数及资源用量需要依据实际 tokenizer、模板和图片展开重新核定。

##### 数据形态

只读解析确认原文件有 327 条有效 JSONL，每条包含 `messages`、`tools` 等字段。数据是含图片的多轮会话，共 65,379 条消息、28,872 个 assistant 段；全部源记录含推理内容，326 条含工具调用。

全部源记录含图，共 1,452 个图片引用，单条最多 48 图；图片可出现在 user 或 tool 消息中。1,437 次本地引用对应 746 个唯一文件，按既有脚本的 `/mnt/afs_toolcall/juxiaolong1/Data/draftData` 根目录解析后均存在且权限可读。本次没有实际解码。另有 15 个 HTTPS 图片引用未在本次调查中联网验证；本地图片包含 GIF。

327 条源记录中，按完整 `messages` 内容比较有 318 种不同内容。用户通过 Q6 确认保留全部源记录，不额外按消息内容去重。

现有图文全量流程采用 128K 总请求上限，为预填充验证保留一个生成 token；必要时准备多个监督窗口并缩短 user/tool 上下文。用户通过 Q5 确认沿用该边界：被监督的 assistant 段保持完整，允许必要时缩短文章或工具返回文本，并记录裁剪范围。

##### 已确定的适配缺口

1. [Loop 配置校验](../../vllm/vllm/model_executor/models/config.py)明确要求 `language_model_only=True`，开启图文会被拒绝。只改新示例的 YAML 不能完成适配。
2. Loop 要求 vLLM V2；旧图文 [full_train.py](../../examples/qwen3.8-27b-image-text-1node/full_train.py)强制 V1，并绑定旧示例配置和 batch=16。复用需要处理这些固定项。
3. Loop checkpoint 的原生模板不识别 `preserve_thinking`，省略最后 user 之前的 assistant 推理内容。既有 renderer 只传入该开关，无法自动保留历史推理。用户已通过 Q4 确认沿用 [ADR-0003](../../docs/adr/0003-multimodal-dspark-all-assistant-supervision.md)的全部 assistant 监督约定，因此需要明确的模板适配与原数据内容核验。

Loop checkpoint 已包含视觉配置及视觉权重索引，模型类继承已有视觉外层，V2 runner 也有图片编码和 embedding 输入通道。现有目标请求与采集接口已支持图片展开后的 token/mask 校验；但 Loop 的图文权重加载、位置编码、特征采集与训练联合运行仍未验证。当前没有证据要求改写 DSpark 算法或主体训练器。

#### 第一轮决策：已确认

用户回答：`Q1:是，Q2:是的 Q3:是的`。

| 问题 | 决定 | 状态 |
| --- | --- | --- |
| Q1：训练对象 | 冻结目标模型与视觉编码器，只优化 DSpark 草稿，学习已有 assistant 回答 | 已确认 |
| Q2：数据覆盖 | 全部 327 条源记录进入训练并核对覆盖；代表性小样本只用于前置验证 | 已确认 |
| Q3：最少改动的边界 | 优先新示例配置和薄入口，复用现有流程；允许必要的小范围公共代码修复，包括已定位的 vLLM 限制，并保持现有示例行为 | 已确认 |

#### 第二轮决策：已确认

用户回答：`本地接受率的 png，Q4-Q8都是 yes`。Q4–Q8 的建议已确认；追加接受率 PNG，其统计口径随后在 Q9 明确。

| 问题 | 决定 | 状态 |
| --- | --- | --- |
| Q4：监督范围 | 监督所有已有 assistant 正文、reasoning 和 tool calls；user/tool、视觉位置及额外生成的验证 token 不参与回答 loss | 已确认 |
| Q5：长上下文与文章完整性 | 沿用 128K 总请求上限，输入最多 131071 tokens；分窗保留被监督的完整 assistant 段，重叠历史只作上下文。必要时缩短 user/tool 文本前缀并记录；这可能缩短原文章内容。无法完整容纳监督段时报错，不静默丢弃 | 已确认 |
| Q6：原始数据与图片语义 | 保留全部 327 条源记录，不额外按消息内容去重；中途 system 保持原角色及顺序；GIF 取首帧；HTTPS 图片缓存后实际解码，失败报源行；原始 JSONL 保持不变 | 已确认 |
| Q7：训练起点与终点 | 沿用现有 Loop ckpt-225-bf16 目标和当前五层草稿结构，可训练草稿参数从零初始化；不恢复旧训练状态；先完整遍历准备后的窗口一轮，更新步数依实际窗口数计算 | 已确认 |
| Q8：运行及验收交付 | 单机八卡，四卡 TP=4 目标推理、四卡 FSDP 草稿训练；先真实图文预检与相关文本回归，再全量训练。核验有限 loss、真实参数更新、源记录/监督段覆盖和完整 checkpoint，保存本地 loss PNG、精确指标及原始日志；追加接受率 PNG，其口径见 Q9 | 已确认 |

Q7 的“一轮”指实际准备后的训练窗口，不预设 Loop 仍为旧模型的 610 窗口或 39 次更新。Q8 已确定单机 4+4 的资源布局，未在本轮检查 GPU 是否空闲。

#### 接受率口径的调查依据

用户明确要求交付本地接受率 PNG，保留 Q8 已确认的 loss PNG。Q9 已选择训练接受概率估计；以下事实解释其统计含义与最小改动范围。

只读核查得到：

- [DSpark 模型](../../torchspec/models/dspark.py)在原数据上下文中计算目标和草稿的概率分布。Markov 项使用真实前一个 token，按有效监督位置和既有位置衰减权重统计 L1；当前配置的 L1 权重为正，故可从精确 `train/l1_loss` 得到加权分布重叠率 `1 - L1 / 2`。它是训练上下文中的单 token 接受概率估计，不是生成轨迹中实际接受的 token 比例。
- `train/avg_acc` 和各位置 `train/acc_i` 比较草稿 argmax 与原数据 token；`train/simulated_acc_len` 累加各位置准确率的前缀连乘。它们分别是预测准确率和模拟接受长度，不能用真实解码接受率标题代替。
- [本地指标记录器](../../torchspec/utils/dspark_metrics.py)已保存每个 optimizer metric step 的精确分项损失和标量指标，目前仅生成 `loss.png`。训练接受概率估计可复用这些记录生成独立 PNG，无需改变训练目标或启动解码评估。
- [已有解码示例](../../examples/infer_dspark.py)以 vLLM counters 计算真实 `accepted_tokens / proposed_tokens`，但入口只处理文本，且 [Loop 校验器](../../vllm/vllm/model_executor/models/config.py)明确拒绝 speculative decoding。当前 prefill 训练只运行目标模型并额外生成一个验证 token，没有真实草稿提出/接受计数。
- 现有缓存特征 eval 仍是无梯度训练前向；未找到可直接用于 Loop 图文的独立离线生成接受率评估入口。真实生成接受率需要额外适配推测解码，并确定评估数据、采样和生成长度。

##### 第三轮 Q9：已确认

用户回答：`Q9: 训练接受概率估计`。

本次在 `acceptance.png` 中绘制每个 optimizer step 的**训练接受概率估计**，使用 `1 - train/l1_loss / 2`，按 0–100% 展示，并明确标注训练上下文和位置衰减加权。沿用精确本地指标，可重绘曲线；不将 `avg_acc` 或模拟接受长度改名为真实接受率。

未选择实际生成接受率，因此不展开独立解码评估的数据集、生成长度、采样设置和 checkpoint 评估频率分支。训练接受概率估计已写入 [GLOSSARY.md](../../GLOSSARY.md)；这是可复用的领域用语，具体公式与实现条件留在本规约。

#### 决策树与整体确认

Q1–Q9 均已确认：训练对象、全量覆盖、允许的小范围修改、监督内容、上下文裁剪、媒体规则、新训练一轮、资源布局和交付物、训练接受概率估计。未选择的真实解码评估分支已关闭；具体实施默认值和验收要求已在上文列出。

用户对最终整体规约回答“确认”。需求访谈已完成，按上文合同实施并执行预检和全量训练，无需重复请求已授权工作的许可。

#### 确认前的调查状态

确认前完成了只读文件/数据调查，记录全部九项需求决定，整理实施默认值和验收合同，并在术语表中补充训练接受概率估计、明确源记录的计数单位；当时尚未修改训练代码或启动训练。特征采集和全部 assistant 监督继续沿用 ADR-0002/0003，本次不新增重复的架构 ADR。确认后的实施与运行状态记录如下。

### 2026-10-14 — 整体确认并开始实施

用户已确认最终规约。正在复用全量驱动增加 Loop 配置和 V2 选择，适配历史 reasoning、图文配置与接受概率 PNG；尚未宣称真实图文训练通过。开始时八张 H800 均空闲，可用主机内存约 1.9 TiB，`/dev/shm` 可用约 512 GiB。

### 2026-10-14 — 适配与针对性检查

新增示例的配置、薄入口和 README；公共全量驱动可接收 Loop 配置及其文本回归基线，按 direct capture 选择 V2。Loop 校验器仅移除纯文本限制，已有其他限制保留。全量 renderer 适配原生模板以保留历史 reasoning，缓存版本更新并记录 tokenizer/template/processor 文件指纹。

本地指标新增 `estimated_training_acceptance` 和独立 `acceptance.png`，沿用未乘损失系数的精确 L1；支持离线重绘，拒绝失效的分布 L1。原 loss 路径保留。

Loop checkpoint 下的数据/完整遍历/指标检查 29 项通过，原目标的图文检查 16 项通过；renderer、目标引擎、Loop 采集和监督回归 143 项通过；vLLM 的 Loop 配置检查 20 项通过。三个改动模块通过 MyPy，主仓库相关 pre-commit hooks 通过。vLLM hooks 和真实运行仍在进行。

全量准备目录为 `cache/qwen3.5-loop-mhc-image-text-dspark-1-node/full-data/`，运行目录为 `outputs/loop-image-text-20261014/`。同时增加原数据零起始行 114 的早期图文预填充检查，以提前暴露 V2 视觉接入问题；它不替代后续十七窗口预检或全量验收。

### 2026-10-14 — 早期图文采集通过

vLLM 全部适用 pre-commit hooks 通过。早期真实图文预检正常退出：3 张图片、11065 个输入 token、1 个额外验证 token、3044 个监督 token；辅助特征形状 `[11065, 20480]`，独立最终状态形状 `[11065, 4096]`，数值全部有限，独立监督核对通过。采集槽为 `[4,28,52,76,79]`，最终状态保持 post-norm。原始日志位于运行目录 `early_prefill/prefill.log`。全量准备仍在进行，此结果不是完整训练验收。

### 2026-10-14 — 全量准备与原文审计通过

327 条源记录生成 595 个窗口，共 68934575 个展开输入 token；最短 11065，最长 131071，单窗口最多 22 张图片。正式训练终点为 38 步，尾批补 13 个零 loss dummy。324 个窗口存在已记录的上下文省略或文本前缀缩短。

独立审计对照原始数据确认全部 28872 个 assistant 段各分配一次，缺失和重复均为零。按 manifest 的裁剪和媒体映射重建每条保留消息后，与准备结果完全一致；同时核对了渲染后的非空 reasoning 内容及工具调用标记。准备 manifest 和独立审计结果已留存于运行目录 `preparation_manifest.json`、`independent_preparation.json`。十七窗口 GPU 预检及训练流程已启动，尚未完成训练验收。

### 2026-10-14 — 预检选择审计发现并修正 GIF 假阳性

旧选择逻辑在消息全文搜索 `.gif`，会把正文里的文件名讨论误判为实际 GIF 输入。独立检查发现第一批十七窗口没有实际 GIF，因此在采集和优化器更新前停止该次启动，日志与原因保留在 `preflight_selection_attempt1/`。选择逻辑改为检查真实图片 URI，并通过 manifest 的远程媒体映射识别缓存图片；新增回归用例通过，相关 pre-commit hooks 通过。

重新选择的十七个窗口通过 `preflight/selection_audit.json` 核对：最大 131071 tokens、最多 22 图，真实 GIF、远程缓存图片、tool 图片、中途 system、历史 reasoning、缩短文本前缀及仅作上下文的 assistant 均覆盖。全量数据准备无需重建，真实 GPU 预检已重新启动。

### 2026-10-14 — 图文、文本与两步训练预检通过

十七窗口全部完成真实 GPU 采集，特征有限且形状、独立监督范围正确，验证 token 均排除在监督之外；见 `preflight/prefill_acceptance.json`。Loop 原纯文本请求通过，87 个输入 token、38 个监督 token，原 mask 一致。

两步训练完整提交 17 个真实窗口、691 个监督段，尾批恰好补 15 个 dummy。四个训练 rank 的两步真实更新、精确 loss、TensorBoard 一致性及接受概率公式均通过独立核查。总 loss 为 3.72427249、3.08709301；训练接受概率估计分别约 0.2330%、1.4324%。这些是预检结果，不用于宣称全量训练表现。

预检 checkpoint `preflight/training/checkpoints/iter_0000002` 的模型、optimizer、scheduler 各四个 shard 完整；62 个参数的 optimizer step 均为 2，scheduler 与 RNG 核对通过，冻结的共享 embedding/head 不在 optimizer master 参数中。报告为 `preflight/training/independent_acceptance.json`。队列首次传输曾耗时约 217 秒，采样确认是依赖导入的冷启动，随后恢复，无需改动训练主体。全量 595 窗口、38 步训练已以全新状态启动。

### 2026-10-15 — 全量训练完成，最终验收通过

全量运行于北京时间 02:49:37 报告完成，主进程退出码为 0。实际完成 38 次 optimizer 更新，595 个真实窗口、327 条源记录、28872 个 assistant 段全部提交；尾批恰好补 13 个零 loss dummy。提交、接收、分发和训练提交的窗口身份集合完全一致，缺失、重复和未完成在途结果均为零。

四个训练 rank 的 38 次真实参数更新均通过日志核验。38 步精确指标全部有限，加权总 loss 公式、未乘 0.9 的 L1 接受概率公式及 TensorBoard 一致性全部通过。首步与末步总 loss 分别为 3.718174922466278、2.7306162238121034；训练接受概率估计分别为 0.24548768997192383%、1.5488207340240479%。这些指标来自各步更新前的训练上下文，保持 Q9 确认的统计口径。

最终 checkpoint 为 `outputs/loop-image-text-20261014/checkpoints/iter_0000038/`。模型、optimizer、scheduler 各四个 shard 的元数据和文件范围完整；62 个参数的 optimizer step 均为 38，scheduler 的 last_epoch/total_steps、RNG 和 checkpoint 最终步数核对通过。冻结的共享 embedding/head 未进入 optimizer master 参数，Loop 五层草稿、辅助执行层及 post-norm final 契约保持一致。运行结束后八张 GPU 均已释放。

原始 JSONL SHA256 保持 `8cb95a3270ff3c5515c3a1457cf6ff7bd86331b727b6b79f09482465deb6967e`，配置、模板、tokenizer 和 processor 文件指纹与准备 manifest 一致。两张最终 PNG 已目视检查，并从完整 `metrics.jsonl` 经命令行离线重绘；重绘文件与训练产物逐字节一致。Loss PNG 为 1540×980，训练接受概率 PNG 为 1400×700。

| 验收项 | 结果与证据 |
| --- | --- |
| AC-01 配置 | 通过：[实际运行配置](../../outputs/loop-image-text-20261014/config.yaml)，正确 Loop/V2/视觉/4+4/128K/新训练状态；四个训练 actor 的 allocator 环境已留存 |
| AC-02 数据 | 通过：[准备审计](../../outputs/loop-image-text-20261014/independent_preparation.json)、[输入身份核验](../../outputs/loop-image-text-20261014/input_identity_audit.json) |
| AC-03 图文预检 | 通过：[十七窗口选择审计](../../outputs/loop-image-text-20261014/preflight/selection_audit.json)、[真实采集验收](../../outputs/loop-image-text-20261014/preflight/prefill_acceptance.json) |
| AC-04 训练预检 | 通过：[两步预检独立验收](../../outputs/loop-image-text-20261014/preflight/training/independent_acceptance.json) |
| AC-05 全量完成 | 通过：[覆盖报告](../../outputs/loop-image-text-20261014/coverage.json)、[全量独立验收](../../outputs/loop-image-text-20261014/independent_acceptance.json) |
| AC-06 指标与产物 | 通过：[全量独立验收](../../outputs/loop-image-text-20261014/independent_acceptance.json)、[曲线重绘核验](../../outputs/loop-image-text-20261014/curve_regeneration_audit.json)、[最终 checkpoint](../../outputs/loop-image-text-20261014/checkpoints/iter_0000038/) |
| AC-07 回归 | 通过：[Loop 原文本预检](../../outputs/loop-image-text-20261014/preflight/legacy-text/summary.json)及上文记录的针对性测试、MyPy、主仓库与 vLLM hooks；真实 GIF 选择回归另有一项通过 |

运行入口和产物索引已更新到[示例 README](../../examples/qwen3.5-loop-mhc-image-text-dspark-1-node/README.md)。本次未改动 DSpark 算法或主体训练器，按已确认范围完成适配和验收。

### 2026-10-15 — to-spec synthesis

Reorganized the approved Loop/mHC image-text DSpark scope into the requested specification template and recorded Status: ready-for-agent. Added comprehensive user stories, explicit implementation decisions, the already-confirmed workflow test seam, regression prior art, acceptance criteria, and scope boundaries. Preserved the original scope and Q1–Q9 confirmations alongside all execution comments and evidence links. Existing reports still record the completed 595-window, 38-step pass; this synthesis did not modify training code, rerun training, or claim new runtime validation.

### 2026-10-15 — Implementation verification and commit

The implement workflow finalized the existing feature changes against starting commit
`9545d3e180bf7cc1d7e65dbe1fe9db9d2ea54996`. The accepted 595-window, 38-step GPU
run remains the runtime evidence; its reports, coverage, PNGs, and final checkpoint
were checked again and remain present and consistent. No new training run was needed.

- The Loop workflow passed 30 tests, the original target's image-text workflow passed
  17 tests, and the vLLM Loop configuration checks passed 20 tests. The launcher command
  preview and shell syntax checks passed.
- MyPy passed for the three changed library modules. The expanded check of both example
  scripts retains one existing OmegaConf merge assignment diagnostic; checking their
  versions from the starting commit reproduces the same diagnostic. The vLLM MyPy hook
  and all applicable hooks in both repositories passed.
- The completed standard full-suite run reported **1,110 passed, 74 failed, 24 skipped,
  and 125 passing subtests**. All 74 failing test IDs exactly match the recorded baseline;
  there are no newly failing IDs. This is not a claim that the full suite is green.
- The placement-group tests also passed separately: 12 tests. Two incomplete runs with
  that module isolated are retained in the verification record: the first encountered
  an invalid relative interpreter path in Ray workers; the second was interrupted after
  86 passing tests because fresh Ray-worker startup made the variant slow. The standard
  full-suite invocation supplied the completed, directly comparable result above.
- Independent code-review axes found **0 Standards findings** and **0 Spec findings**.
  Reviews covered the staged feature changes and the vLLM configuration/test changes.
- The vLLM change is committed as `60bd0552dc99e587777b2b3370ca10608a51b987`, and the
  parent feature commit records that revision. Unrelated existing working-tree edits
  are outside this feature commit.

The [verification report](../../outputs/loop-image-text-implementation-20261015/verification.json)
records the commands, baseline comparison, review results, and preserved attempt logs.
The [complete suite log](../../outputs/loop-image-text-implementation-20261015/full-suite-standard.log)
and [JUnit report](../../outputs/loop-image-text-implementation-20261015/full-suite-standard.xml)
retain the detailed results.
