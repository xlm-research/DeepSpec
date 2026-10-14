# 单机图文 DSpark 全量训练与本地 loss 曲线

Status: ready-for-agent

## Problem Statement

The existing single-node image-text example proves that target-model prefill, hidden-state transfer, and DSpark optimization work for three source conversations repeated into 160 smoke records. It does not train on all 327 distinct source conversations, prove a complete data pass, or provide the requested local loss curves for a fresh full-data run.

Here, **source conversation** means **训练源会话**, **assistant supervision segment** means **assistant 监督段**, **image-text context** means **图文上下文**, and **DSpark supervision window** means **DSpark 监督窗口**, as defined in the project glossary. The **target model (目标模型)** supplies reference predictions and representations; only the **DSpark draft model (DSpark 草稿模型)** is optimized.

Several source conversations exceed the chosen 128K limit before image expansion. The smoke configuration also limits each request to three images, excludes remote-image preparation, fixes training at ten steps, and calculates epoch length by rounding down. Reloading data while inference is still in flight can produce repeated exposure without proving coverage. Existing rounded console loss messages cannot recover a precise full-run curve.

## Solution

Provide a full-data mode of the existing single-node image-text workflow that prepares all 327 source conversations, trains a newly initialized DSpark draft model, and delivers local PNG loss curves, exact local metrics, raw training logs, a coverage report, and the final checkpoint.

Prepare bounded DSpark supervision windows that preserve each selected assistant supervision segment in full, including existing reasoning and tool calls. Supervise each segment in exactly one window per data pass; reuse overlapping historical answers only as context. Validate the actual image-expanded input length against a 131072-token total limit, reserving one token for the existing prefill completion check.

The concrete proposed implementation uses one complete pass over the prepared windows, preserves later system messages in their original order and role, and shortens necessary user/tool context with an auditable record. These choices come from the earlier recommended plan; they must not be reported as additional user confirmations. Their relationship to the outstanding overall run confirmation is recorded in Further Notes.

## User Stories

1. As a training user, I want all 327 distinct source conversations included, so that the run uses the full requested dataset.
2. As a training user, I want the full-data input to contain genuine supervision windows, so that repeated smoke records do not inflate apparent data coverage.
3. As a training user, I want the original source data left intact, so that I can reproduce preparation and inspect the source conversations.
4. As a training user, I want a newly initialized DSpark draft model, so that the full-data run starts from the agreed training state.
5. As a training user, I want the target model and its vision encoder frozen, so that the run only learns the requested draft parameters.
6. As a training user, I want shared target embeddings, head, and norm frozen, so that optimization preserves the existing target contract.
7. As a training user, I want every existing assistant supervision segment available to training, so that earlier answers contribute alongside the final answer.
8. As a training user, I want complete assistant answer bodies preserved, so that window preparation does not change the learning targets.
9. As a training user, I want historical reasoning preserved and supervised, so that the draft learns the reasoning present in the source data.
10. As a training user, I want complete tool-call names and arguments preserved and supervised, so that tool-use targets remain faithful to the source.
11. As a training user, I want user and tool messages used as image-text context, so that their content conditions predictions without becoming answer targets.
12. As a training user, I want visual tokens excluded from answer loss, so that image expansion does not introduce unintended prediction targets.
13. As a training user, I want the newly generated prefill-check token excluded from supervision, so that training uses existing answers.
14. As a training user, I want long source conversations converted into 128K supervision windows, so that the selected context limit can support the full-data workflow.
15. As a training user, I want complete selected assistant segments kept within individual windows, so that window boundaries do not split reasoning or tool calls.
16. As a training user, I want each assistant segment assigned to exactly one supervised window per pass, so that overlapping history does not receive duplicate loss.
17. As a training user, I want necessary context reductions reported, so that I can understand which user/tool information was shortened.
18. As a training user, I want unrepresentable supervision reported with source identifiers, so that a failed preparation cannot silently reduce the requested dataset.
19. As a training user, I want window limits checked after real image expansion, so that an apparently short placeholder sequence cannot exceed the inference limit.
20. As a training user, I want image payloads kept aligned with their placeholders during windowing, so that the model receives the intended visual context.
21. As a training user, I want images from both user and tool messages handled, so that preparation covers the actual multimodal source format.
22. As a training user, I want the image limit derived from prepared windows, so that requests with more than three images are supported.
23. As a training user, I want relative local media paths resolved consistently, so that the full run reads the same images as preparation.
24. As a training user, I want HTTPS images cached and actually decoded before training, so that a successful HEAD request cannot hide an unusable training input.
25. As a training user, I want GIFs treated as first-frame still images, so that the run follows the agreed image semantics.
26. As a training user, I want existing orientation and transparency conversion behavior retained, so that preparation and inference interpret images consistently.
27. As a training user, I want later system messages kept in their original role and order, so that later instructions do not affect answers that preceded them.
28. As a training user, I want top-level tools and formatting options preserved, so that window rendering maintains the source conversation contract.
29. As a training user, I want actual expanded-token masks aligned with captured hidden states, so that loss applies to the intended assistant segments.
30. As a training user, I want changed windowing and mask semantics to invalidate incompatible caches, so that stale preprocessing cannot change training behavior.
31. As a training user, I want one complete pass over all prepared windows, so that the proposed endpoint has a verifiable coverage meaning.
32. As a training user, I want the optimizer-step count and learning-rate schedule derived from the window count, so that smoke-run settings cannot stop the full run early.
33. As a training user, I want the final incomplete batch trained, so that rounding down does not omit real supervision.
34. As a training user, I want tail padding to contribute zero loss and zero supervision count, so that distributed batching does not duplicate or dilute real targets.
35. As a training user, I want in-flight inference distinguished from exhaustion, so that slow results cannot trigger duplicate data submission.
36. As a training user, I want missing, failed, or duplicate windows reported explicitly, so that the run cannot claim full coverage from optimizer steps alone.
37. As a training user, I want the established four-GPU training and four-GPU TP=4 inference layout retained, so that the workflow uses the existing single-node setup.
38. As a training user, I want representative real-GPU preflight checks before the full run, so that windowing, masking, and tail handling have direct execution evidence.
39. As a training user, I want exact per-step loss components saved locally, so that I can inspect and regenerate the curve without rounded console values.
40. As a training user, I want total loss and CE, L1, and confidence components plotted against actual optimizer metric steps, so that the PNG reflects the optimized objective.
41. As a training user, I want raw training logs retained, so that I can investigate runtime failures and compare them with coverage and metrics.
42. As a training user, I want a final checkpoint tied to the completed coverage report, so that I can identify the resulting draft model.
43. As a maintainer, I want the existing text-only workflow covered by regression checks, so that full-data image-text support preserves established text behavior.
44. As a training user, I want completion evidence separated from model-quality claims, so that a successful data pass is not mistaken for verified decoding quality.

## Implementation Decisions

- **Extend the existing image-text example.** Add full-data preparation and configuration within the existing single-node image-text workflow. Reuse the renderer registry, dataset loader, target prefill adapter, Mooncake transfer, DSpark trainer, optimizer, checkpointing, and local metric writer. Preserve the smoke recipe and existing text-only defaults.
- **Use the recorded source and model contract.** Process all 327 distinct source conversations from the recorded source dataset using the established Sensenova Flash Lite v42 target checkpoint and DSpark draft configuration. Derive training inputs without modifying the source. Retain the existing optimizer settings, loss weights, and anchor-sampling strategy unless the full-data contract requires a documented correction.
- **Start fresh.** The full-data mode must resolve to a newly initialized draft training state with no old smoke model, optimizer, or scheduler state loaded. Use a separate run output and record the resolved configuration and source identity.
- **Make provenance explicit.** Reuse the existing sample identifier and metadata interfaces to carry a stable source identifier, original assistant-segment identifiers, a unique window identifier, selected supervision segments, context-only segments, media references, and context-reduction records. Preparation must emit a manifest that maps every source conversation and supervision segment to its windows.
- **Prepare complete, bounded supervision windows.** A window may supervise multiple complete assistant segments. Assign every original segment to exactly one supervised window per pass; historical segments reused in later windows are context-only. Keep retained messages in original order and exclude future messages from a window targeting earlier answers.
- **Treat 128K as an expanded-input limit.** The total target request limit remains 131072 tokens, with at most 131071 input tokens and one reserved generated token. Validate with the actual tokenizer and processor, including system/tools formatting and expanded visual tokens. A larger model configuration limit does not authorize a larger window.
- **Record context shortening.** When bounded history selection is insufficient, shorten necessary user/tool context using a deterministic policy and record the original message indices and removed ranges. Never cut a selected assistant answer, its reasoning, or its tool-call arguments. Preserve the association between retained image markers and payloads, and keep retained roles and order intact. An impossible window is a preparation failure with a source/segment diagnostic; it cannot be silently skipped.
- **Preserve later system messages.** The proposed full-data formatting extension preserves non-initial system messages at their original positions and with their original roles. It must not hoist later instructions into the initial prefix or change the original text-only formatter. Preserve top-level tools and generation-formatting options, historical thinking, and the existing complete-conversation prefill behavior.
- **Prepare media before training.** Resolve relative media against the recorded media root, cache remote image bytes locally, and fully decode required images. Use the agreed first GIF frame and existing EXIF/orientation and white-background conversion behavior. Keep media order aligned with retained content blocks. Set the inference image limit from the maximum actually prepared window; retain static-image semantics.
- **Apply supervision selection after image expansion.** Reuse native assistant/visual masking on the actual expanded input IDs, then constrain it to the selected complete supervision segments. System/user/tool content, visual positions, overlapping context-only assistant segments, and the generated check token have zero answer loss. The same effective supervision weights constrain CE, L1, and confidence losses. Do not reuse offsets or packed masks from unexpanded placeholders.
- **Keep the target feature contract.** Reuse the five auxiliary layers 3, 15, 31, 47, and 59, their capture slots 4, 16, 32, 48, and 60, and independent final slot 64 with the existing pre-norm semantics. Do not apply the separate Loop/mHC post-norm contract to this target. Keep the target model, vision encoder, and shared embedding/head/norm frozen.
- **Version preparation semantics.** Invalidate caches whose renderer, supervision selection, context-reduction policy, media handling, or source identity is incompatible with the full-data inputs. Record enough preparation metadata to reproduce the selected windows and avoid accidental reuse of smoke records.
- **Derive the complete-pass endpoint.** Let M be the number of valid real supervision windows after successful full-source preparation. Under the existing global batch of 16, the proposed single pass requires ceil(M / 16) optimizer steps. Derive epoch steps, the explicit training endpoint, checkpoint timing, and learning-rate horizon together. Remove the ten-step endpoint and eleven-step smoke horizon from the full-data configuration.
- **Include the tail without repeated supervision.** Keep every real tail window and complete the distributed dispatch/accumulation shape using distinguishable zero-loss dummy entries. Dummies carry no real source or supervision-segment credit. They contribute zero loss numerator, zero supervised denominator, and no sample/coverage credit; ranks with only padding still participate safely in collectives. Normalize the real tail objective consistently with the existing supervised-token pooling, rather than diluting it by padding.
- **Reconcile completion by identity.** Track expected, submitted, in-flight, received, failed, and committed real window IDs for the pass. Count a window as consumed only after its optimizer step succeeds. An empty prompt buffer or temporarily small result pool does not prove exhaustion. Do not reload a pass to fill a shortage; wait for known in-flight work or fail with missing/failed IDs. Duplicate results must not create duplicate training commits.
- **Use the existing execution layout and metric path.** Retain one eight-H800 node, four FSDP training GPUs, and four target-inference GPUs with TP=4 and PP=1. Use the existing prefill extraction route. Recheck resource availability when executing; the earlier idle-GPU observation is historical.
- **Deliver exact local loss curves.** Enable exact local optimizer metrics through the existing TensorBoard/local writer and preserve raw launch and actor logs. Keep train/avg_loss identified as the existing CE summary. Plot the token-pooled train/ce_loss, train/l1_loss, and train/confidence_loss components, and reconstruct total loss as 0.1 × CE + 0.9 × L1 + 1.0 × confidence. Record coefficients from the resolved run configuration. Use the actual optimizer metric-step IDs, including the final drained step; do not substitute inference steps or rounded progress text.
- **Tie artifacts to completion.** Refresh local PNG plots during the run and produce the final plot, logs, metrics, coverage report, and checkpoint at completion. The report must reconcile all 327 source IDs, all original supervision-segment assignments, and all M real committed windows, with no missing or duplicate supervision. Preserve diagnostics and mark incomplete runs accurately if preparation or execution fails.

## Testing Decisions

- **Primary seam: the existing single-node image-text workflow.** Exercise preparation, loading, target prefill, training, and artifact consumption through their existing public interfaces. Assert the externally visible manifest, effective supervision, committed IDs, metric steps, and resulting checkpoint. Prefer this workflow boundary over new test-only interfaces or separate seams for every helper. This was proposed in the testing-boundary check; absent a preference change, it is the spec's default.
- **A good test compares behavior with independent source expectations.** Verify complete answer/reasoning/tool-call content, source-to-window assignments, selected supervision, and coverage counts using original message identities and independent spans. Do not derive the expected mask or coverage set by calling the implementation being checked. Avoid asserting private helper structure, retry-loop counts, or rank scheduling.
- **Preparation cases.** Cover a short unchanged conversation, multiple complete assistant segments, overlapping context-only history, oversized history, a user/tool message that exceeds the input budget, later system messages, and an assistant segment that cannot fit intact. Check deterministic provenance and explicit failures. Include source rows 111, 236, and 296 for extreme lengths, and rows 232/233 for the previously observed oversized user-message case when running real-source preflight.
- **Media and expanded-length cases.** Cover user and tool images, more than three images, the largest prepared image count, remote images, GIF first-frame behavior, orientation/transparency conversion, and decoding failures. Check exact-limit success and one-token overflow after real expansion. Keep marker/payload counts and ordering consistent with recorded context reductions.
- **Supervision through the real processor.** Use the established real-checkpoint tokenizer/processor fixture with independent rendered character spans and tokenizer offsets. Verify all selected assistant bodies, full reasoning and tool arguments, excluded context-only historical answers, later system roles, and zero visual/context/generated-token weights. Check input IDs, masks, auxiliary features, and final states remain position-aligned through actual Mooncake retrieval.
- **Complete-pass and asynchronous cases.** Use controlled inference results at the existing training-loop boundary to cover delayed and out-of-order results, a drained prompt buffer with work still in flight, a duplicate result, a failed or missing result, and failure before optimizer commit. Compare observable committed-window/segment sets and completion reports; a shortage must not be repaired with repeated source submission.
- **Tail and endpoint cases.** Cover M below 16, exactly 16, above 16 with a partial tail, and a divisible multi-step pass. Check ceil(M / 16) successful optimizer steps, all real windows committed once, zero dummy coverage, finite loss/gradients, safe participation of padding-only ranks, and the real tail objective's numerator/denominator. Validate that smoke step/scheduler overrides cannot silently replace the full-pass endpoint.
- **Loss artifacts.** Feed known unequal-denominator component metrics and actual optimizer-step IDs into the artifact path. Check pooled component values, the weighted total identity, ordered unique plot steps, and inclusion of the final optimizer step. Verify the local PNG is readable and regenerable from exact metrics; do not compare pixels or require a monotonically decreasing loss.
- **Freeze and checkpoint behavior.** Reuse public optimizer update checks to verify finite updates to trainable draft parameters and no updates to frozen target/shared parameters. Confirm the final checkpoint records the same completed optimizer endpoint and target feature contract as the final metrics and coverage report. No old smoke training state may be loaded for the new run.
- **Prior art.** Extend existing renderer-registry real-template tests, training-loop cleanup/prefetch/final-checkpoint tests, inference-manager delayed/failure handling tests, Mooncake data-fetcher incomplete-batch and zero-mask tests, DSpark weighted-loss/all-masked/frozen-embedding tests, and optimizer update-probe tests. Use these established patterns for focused regressions instead of building a new testing framework.
- **Real-GPU acceptance.** Before the full run, execute representative windows for long-context handling, multiple images, tool images, later system messages, overlap masking, and a real distributed tail step. Retrieve and inspect target features and effective masks. Then run the complete pass and reconcile coverage and artifacts. Record CPU/simulated results separately from real-GPU evidence.
- **Compatibility and practical limits.** Run the existing legacy text-only GPU regression and focused affected tests. ADR-0003 remains the all-assistant-supervision contract. Record unrelated baseline failures separately. A complete data pass validates execution and eligible supervision; existing anchor sampling does not guarantee every target token is sampled or establish draft decoding quality.

## Out of Scope

- Increasing the selected context limit to the model's 262144-token configuration maximum.
- Cutting assistant supervision segments or silently omitting source conversations to make the data fit.
- Reusing or restoring the previous smoke checkpoint for this fresh run.
- Training the target model, vision encoder, or shared embedding/head/norm.
- Converting GIFs into temporal video inputs.
- Replacing the DSpark architecture, loss objective, feature-capture contract, or anchor-sampling strategy.
- Changing the default text-only supervision policy.
- Adding a website, hosted dashboard, CSV deliverable, or external metric upload.
- Multiple full-data epochs, hyperparameter searches, complete-answer generation evaluation, or speculative-decoding quality benchmarks.
- Claiming every assistant token is sampled during one pass or guaranteeing a particular loss trajectory.

## Further Notes

The source of this spec is the existing local scope record and conversation history preserved below, plus the glossary, ADR-0003, and the inspected image-text implementation and tests. No new feature interview was conducted.

The recorded user confirmations are: all 327 distinct source conversations; local PNG curves and raw logs; fresh draft training; GIF first-frame semantics; and a 128K window limit. One complete pass, later system messages preserved in order/role, and necessary user/tool context reduction are the concrete proposed defaults from the earlier plan. The earlier record still calls for an overall semantic confirmation before starting the new full training run. The ready-for-agent status describes a reviewable implementation specification; it does not turn those proposals into user confirmations or record a completed run.

The 2026-10-14 investigation reported 1452 image references, at most 48 images per source conversation, 30 conversations containing GIFs, 15 distinct remote URLs across four conversations, and 26 conversations with later system messages. Remote HEAD checks and idle GPUs were observations at that time; actual decode validation, complete expanded-window preparation, the final M and step count, and full-source training coverage remain execution work.

The earlier smoke acceptance validated three distinct source conversations, ten optimizer steps, an additional restored step, and text-only compatibility. It did not validate this full-data contract. Publishing this specification does not claim implementation or GPU acceptance.

Useful local evidence: [domain glossary](../../GLOSSARY.md), [all-assistant supervision ADR](../../docs/adr/0003-multimodal-dspark-all-assistant-supervision.md), [image-text workflow](../../examples/qwen3.8-27b-image-text-1node/README.md), and [prior smoke acceptance](../../docs/qwen3_8_27b_multimodal_results.md).

## Comments

### 2026-10-14 — Original scope and conversation record

The original record is retained below. Its pending-confirmation and execution-state statements describe the state before this template synthesis.

#### 已确认范围

- 入口：`examples/qwen3.8-27b-image-text-1node`。
- 使用 `mydata/image-text-327.jsonl` 中全部 327 条不同训练源会话，不沿用三条源会话重复成 160 条记录的 smoke 数据。
- 沿用全部已有 assistant 监督段，包括 reasoning 和 tool calls；目标模型、视觉编码器及共享 embedding/head/norm 保持冻结，仅优化 DSpark 草稿模型。
- 用户要求本地 PNG loss 曲线和原始训练日志；不要求网页或 CSV 交付。
- 用户确认从零开始新的 DSpark 草稿训练，不恢复旧 smoke checkpoint。
- GIF 沿用现有静态图片语义，读取首帧，不以视频时序输入处理。
- 用户选择保持 128K 窗口上限。按第三轮给出的 128K 选项制定必要的 user/tool 上下文缩短方案；所有 assistant 监督段完整保留。该细节纳入最终整体确认。
- 用户指定 `grill-with-docs`：先调查并确认语义，再启动训练。

#### 已核实事实（2026-10-14）

- 八张 H800 当前空闲；现有单机布局为四卡 FSDP 训练、四卡 TP=4 目标推理，global batch 为 16。
- 原始数据包含 1452 个图片引用；106 条会话超过 smoke 配置每请求三图的限制，最多 48 图。30 条会话含 GIF，四条会话引用 HTTPS 图片。
- 真实 tokenizer 的 CPU 检查显示，零起始源行 111、236、296 在图片展开前分别为 520537、771971、564506 tokens，已经超过现有 prompt 上限 131071；不是全部会话均能保持完整历史后直接运行现有配置。
- 目标和草稿模型的配置上限均为 262144 tokens。源行 232/233 的单个 user 消息渲染后为 131086 tokens，加实际 system/tools 前缀为 172238；纯 128K 完整消息分窗仍无法容纳它们。
- 26 条源会话在首条之后出现 system 消息；原生模板要求 system 位于开头，需显式确定规范化语义。
- 15 个远程图片地址的 HEAD 检查均返回 200 与 PNG 类型；尚不能代替完整下载和解码验证。
- 现有 `num_train_steps=10` 优先于 epoch 配置；`lr_total_steps=11` 仅服务旧 smoke/恢复验收。
- epoch 步数采用向下取整，327 条/global batch 16 只能计入 320 次训练曝光。此前 smoke 日志还存在推理在途时误判数据耗尽、提前重载的问题。全量验收需要核对实际参与训练的源会话/监督片段覆盖。
- `train/avg_loss` 实际为 CE；总 DSpark 目标可由 `0.1 * ce_loss + 0.9 * l1_loss + confidence_loss` 重建。需按真实 optimizer metric step 绘图，展示总损失及分项。
- 现有 TensorBoard 路径可保留每步精确指标；旧十步没有启用该路径，恢复运行仅有一个第 11 步的点。

#### 尚未确认

- 中途 system 消息的规范化语义。
- 训练终点暂按 Q3 推荐的一轮完整数据遍历制定；需纳入最终共同确认。分段后的具体 optimizer 步数由预检确定。
- 确认共同理解：128K 分窗与必要上下文缩短、每段 assistant 监督仅覆盖一次、一轮完整遍历、中途 system 按原顺序保留、本地 PNG 与日志及最终 checkpoint。

#### 最终方案建议

- 原始 327 条源会话全部纳入，保留所有已有 assistant 正文、reasoning、tool calls；DSpark 继续使用原有 anchor 抽样策略，不承诺每个目标 token 在单轮都被实际抽中。
- 将长会话准备成 128K DSpark 监督窗口。重叠历史作为上下文，不重复计 loss；必要时缩短 user/tool 上下文，记录移除的范围。保留图片输入与其对应标记的关系，超限或无法生成有效监督窗口时明确报出源行，不能静默跳过。
- 后续 system 按原顺序和原角色保留，只在独立全量图文路径扩展格式化支持，避免把后出现的指令移到早期回答之前。
- 远程图片本地缓存后做实际解码验证；GIF 沿用首帧与现有 EXIF/白底转换语义。
- 从零开始训练，先完成一个 epoch；M 个有效监督窗口对应 ceil(M/16) 个 optimizer steps，动态制定学习率调度。尾步通过零 loss dummy 补齐，不重复真实监督。
- 修复提前重载问题，以预期、收到及训练提交的窗口 ID 对账，证明一轮覆盖；严格失败并报告缺失窗口，不能以重复重载冒充覆盖。
- 在八张 H800 上沿用四卡训练、四卡 TP=4 推理。先做针对窗口/mask/尾步的真实 GPU 预检，再运行全量训练。
- 保存最终 checkpoint；按真实 optimizer metric step 输出更新的本地 loss PNG 和日志，图中包含加权总 loss 与三个分项。

#### 对话记录

- 第一轮 Q1：用户确认全部 327 条不同训练源会话。
- 第一轮 Q2：用户要求本地 PNG 曲线和日志即可。
- 第二轮 Q3：用户确认从零开始训练；一轮完整数据遍历是本轮提出的建议终点。
- 第二轮 Q4：用户确认 GIF 沿用现有首帧行为。
- 第三轮 Q5：用户选择 128K 上限。

#### 验收与执行状态

仅完成只读调查并记录范围，尚未启动新的训练。待最终共同确认后实施并运行。

### 2026-10-14 — Implementation and verification in progress

The user invoked `implement` for this specification. Implementation follows its concrete
one-pass, ordered-system, 128K-window defaults. The earlier pending-confirmation statements
above remain the historical scope record.

The full-data path now prepares and identifies complete supervision windows, preserves
window metadata through native formatting and expanded-token mask recomputation, accounts
for optimizer commits by window ID, pads the distributed tail without real supervision,
and saves exact local component metrics and PNG loss curves.

For a user/tool context that cannot fit even one complete assistant segment, preparation
packs the largest feasible group of intact supervision segments and shortens user/tool
text prefixes proportionally across the retained context. Each removed range is recorded.
This avoids recreating an almost-full 128K context separately for every answer in a large
source conversation. The real oversized source rows 232 and 233 prepare as four and three
windows, respectively, preserving all 128 and 96 assistant segments.

Verification so far: 253 focused regressions passed; the final workflow checks, including
delayed metric draining, passed. The full repository suite finished with 1101 passed,
24 skipped, 125 passing subtests and 74 failures. Its failed-test identifiers match the
previous recorded baseline exactly, with no new failures. Five new typed modules pass
MyPy; the broader eight-module check reports the same 16 diagnostics as the starting commit.

Full-source preparation and real-GPU acceptance are still in progress. No completed
full-data training run is claimed by this record.

### 2026-10-14 — Review findings resolved

The Standards review found no documented-rule violations. Its duplicated window-search
heuristic was resolved by sharing the search across the unchanged context-shortening
phases. The Spec review found one GPU verification gap: the expected supervision mask
used the same renderer implementation as the engine. GPU preflight now checks retrieved
tokens using independent assistant-body character spans, tokenizer offsets, and original
message identities. A regression rejects supervision leaking into overlapping history.
Both reviewers confirmed their findings were resolved; all 22 focused workflow tests,
the affected typecheck, and applicable pre-commit hooks passed.

### 2026-10-14 — Preparation accepted; GPU startup race fixed

The complete source prepares as 610 unique supervision windows. An independent audit
against original messages confirms all 327 sources and 28,872 assistant segments are
assigned exactly once, with intact assistant/system content and no future messages in
earlier windows. Inputs range from 11,296 to 131,071 expanded tokens; the largest window
contains 20 images. This requires 39 optimizer steps and 14 zero-loss tail entries.
The audit is saved in `cache/qwen3.8-27b-image-text-1node/full-data/independent_audit.json`.

The first GPU preflight exposed an existing Mooncake startup race: the master returned
after a fixed two-second delay before its gRPC port accepted connections. Startup now
waits for both gRPC and metadata TCP readiness with a bounded timeout and cleanup.
A delayed-service regression reproduced the failure before the fix; all 18 Mooncake
tests and the affected typecheck now pass. Both reviewers confirmed no remaining
findings. First-attempt diagnostics are preserved under the run's `preflight_attempt1/`;
the retry has reached target-model initialization. Full GPU acceptance is still pending.

### 2026-10-14 — Native GPU checks passed; capture staging headroom adjusted

All 17 selected image-text captures passed with finite aligned features, exactly one
prefill-check token, and independently verified supervision masks. They cover 1,486,624
expanded input tokens, including the 131,071-token limit and 20-image case. The legacy
text-only GPU regression also passed (100 input tokens, 39 supervised tokens).

The first training preflight failed before an optimizer commit: concurrent long captures
exhausted target GPU memory during Mooncake's `tensor.contiguous()` staging. Its incomplete
coverage and logs are preserved under `preflight/training_attempt1/` and
`preflight/training_attempt1.log`. The full-data recipe now uses one target request at a
time and a 0.7 target memory fraction to leave capture staging space. The four-GPU TP=4
inference/four-GPU FSDP layout and global training batch of 16 are unchanged. All 22
focused workflow tests, typechecking, hooks, and both review axes pass. A fresh training
preflight is running; no full-data optimizer completion is claimed yet.
