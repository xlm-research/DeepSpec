# DSpark native tensor parallel acceptance

Ticket 09 is complete. Real eight-GPU FP32/BF16 updates, checkpoint restart,
save-failure handling and affected smaller-topology regressions pass. These runs
use actual DSpark classes at small dimensions and do not establish 128K support.

## Storage and computation

Native `spmd_types` defines every parameter placement. Q/K/V and gate/up weights
are sharded on projection output; attention output/down weights are sharded on
projection input. Context projection, residual norms, Markov/confidence heads,
embedding and the frozen output head are replicated across TP. FSDP shards each
owned TP parameter across DP. A DTensor storage bridge supplies the pinned
n-dimensional FSDP/DCP API with global shapes; computation uses local tensors
and native SPMD collectives. It does not use the legacy partial-DTensor backend.

TP peers consume the same planned DP example and RNG stream. The DSpark loss
normalizes each logical microbatch over DP only, then averages GAS. TP peers do
not add samples to the loss denominator. The basic path materializes complete
vocabulary logits; vocabulary-parallel loss is a separate ticket.

## Floating-point reduction contract

The first real 24Q/4KV TP4 attempt exposed rounding differences that the earlier
4Q/2KV TP2 fixture did not expose. FP32 loss/gradients met the original tolerance,
but Adam amplified tiny gradients near epsilon into three out-of-tolerance
parameter entries. BF16 forward outputs before the first update were bitwise
identical; a few backward GEMM differences changed later updates. The original
reference and tolerances were preserved.

Actual DSpark layer inputs and gradients reproduced the operator difference.
A sum of smaller GEMMs can round differently from the retained GEMM even when
partial sums use FP32. The corrected column backward gathers the output gradient
and changes weight partition axis with all-to-all, so each rank computes its
share of the input gradient with the complete contraction dimension. Weight
and optimizer ownership remain sharded. Row backward uses the retained BF16
GEMM directly. The FP32 path additionally changes partition axes for row forward
and column weight gradients, preserving their contraction dimension and input
layout; shared Q/K norm weight gradients retain the full head reduction order.
This is a correctness choice with measurable communication cost, not a speedup
claim.

The final FP32 discrepancy came from Inductor's device-local autotuning of
FlexAttention backward's `(output * grad_output).sum(-1)` reduction. Captured
layer inputs, outputs and incoming gradients were identical. Only TP rank two
selected a different reduction tree, changing Q/K gradients and one Adam master
entry near epsilon. Replaying the actual cached kernels reproduced that rank's
delta difference; the other three ranks had none. Complete projection VJPs and
same-device global/head-sharded attention replay matched exactly.

The native DSpark FP32 head-sharded attention wrapper uses the pinned compiler's
`batch_invariant=True` and `triton.max_tiles=1` options. This fixes the reduction
tree across head partitions and devices. It does not modify installed libraries,
compiler cache files, the retained baseline, unsharded/BF16 attention or the outer model's
compile setting. `gqa-fp32-delta-policies.log` records exact agreement with the
captured original reduction, while generic deterministic compilation alone did
not reproduce it. Full phase-entry validation remains the acceptance criterion.

## Evidence

All artifacts are under `output/dspark_torchtitan_orchestration_20260914`.

- `gqa-reference/`: additional immutable, SHA256-recorded 24Q/4KV real DSpark
  reference from the retained two-rank trainer, two updates and GAS two in both
  FP32/BF16. Its independently stated objective also passes. Original baseline
  fixtures are untouched.
- `gqa-native-tp1-test.log`: native FSDP2 TP1 matches that reference in both dtypes
  (316.794 seconds, no skips).
- `gqa-tp4-operator-rounding.log`, `gqa-tp4-backward-rounding.log` and
  `gqa-tp4-fp32-reduction.log`: actual activation/gradient diagnostics. These are
  diagnosis artifacts, not substitutes for the real phase-entry acceptance.
- `gqa-native-tp4-bf16-v2-test.log`: DeepSpec entry launches all eight GPUs as
  DP2 x TP4, performs two updates and exits; every rank passes all original
  trajectory comparisons at rtol 1e-4, atol 1e-6 (247.593 seconds, no skips).
- `gqa-native-tp4-fp32-stable-test.log`: the same eight-GPU acceptance now passes
  FP32 at the original tolerance (253.502 seconds, no skips), including all
  losses, gradients, clip norms, parameters, master/Adam state, scheduler and RNG
  across two updates with unequal microbatch denominators and GAS two.

The feature-reader tests independently cover full and two/three head-tail
producer shards with bitwise reconstruction. Changing draft TP does not alter
teacher identity, source-built vLLM configuration or producer cache ownership.

`gqa-tp4-checkpoint-bf16-test.log` passed in 392.895 seconds without skips.
Eight native workers commit full DCP after update one and exit; eight new
workers restore it after deliberately perturbed initialization, complete
update two, commit and exit. Combined phase observations are bitwise identical
to the already accepted uninterrupted BF16 trajectory, including all eight
rank RNG states, optimizer state and consumption cursors.

`tp2-complete-contraction-test.log` passed the updated TP implementation against
the original immutable 4Q/2KV fixture in both FP32/BF16, DP2 x TP2 on four GPUs
(334.047 seconds, no skips). The complete-contraction changes preserve that
earlier acceptance.

`gqa-tp4-checkpoint-fp32-stable-test.log` passed in 372.673 seconds without
skips. The eight-rank FP32 first/resumed phases are bitwise identical to the
accepted uninterrupted trajectory. Optional synchronized timing also passes:
launch + native critical span + exit equals parent wall time, and each rank's
initialization, restore, training, save and close intervals do not overlap.
`gqa-tp4-save-failure-test.log` passed in 161.178 seconds: real DCP tensor shards
and metadata were written, final commit rename failed, no successful phase was
published, the previous commit remained unchanged and all eight GPUs were freed.

`stable-fp32-tp2-test.log` passed the affected four-GPU regression in 134.196
seconds. Applying the new reduction tree to unsharded attention had changed one
entry in the older single-GPU reference, so its scope is restricted to FP32 TP
head partitions. `stable-fp32-single-scoped-test.log` then passed the original
single-GPU trajectory in 135.977 seconds. No tolerance was relaxed.
`gqa-tp4-host-rng-recovery.log` additionally reads actual committed DCP state on
CPU and verifies Python/NumPy RNG preservation for all eight ranks across restart.
