# Reliable L2L retraining on conservative ViCo397

> Superseded for execution: all L2L training stopped pending paper-grounded protocol audit on2026-11-07. Historical design only; do not use this plan to auto-resume. See experiments/l2l_literature_protocol_audit_20261106/PAPER_GROUNDED_REPRODUCTION_AUDIT_20261107.md.

## Authorization and outcome

The user requested a reliable restart after two failed local training attempts.
They explicitly authorized temporary training use of node11 physicalGPU7
(172.29.189.119, UUID GPU-4d5ee808-e829-c824-c506-cad80ff133f5), to return to
inference when finished. Do not touch the other GPUs/jobs. Earlier pause markers
remain historical evidence and do not apply to this newly authorized run.

Deliver a validated locally retrained motion baseline, test43 outputs/scores and
review-page integration, preserving all previous attempts. Do not claim exact
reproduction of DIM's unpublished EMOCA baseline or fabricate video for L2L.

## Approach

Use the upstream L2L architecture, losses, optimizer factories/ScheduledOptim,
defaultbatch32 and100000-step warmup in a new adapter/run directory. Restore
training conformance before trying alternative learning rates or normalization.
Using official pretrained person models is an already completed transfer
baseline; extending the failed fixed-lr run cannot establish a reliable restart.

Use one primary seed137 followed by replication seeds271 and811. Never choose a
seed/checkpoint using test scores. Report all three and aggregate metrics rather
than choosing the best. Primary seed137 supplies the main per-case display.

## Frozen data and transparent adaptations

Reuse the verified397 feature pairs in
experiments/common_method_comparison_20261106/l2l_retrained/features.
Reuse the approved split: train351, validation46, quarantine1 shared speaker
interval; official test43 andOOD42 stay outside training/statistics/selection.
Freeze sample/window origins and feature digests.64-frame windows: train stride32,
validation stride64, append final complete window, no short-clip padding.
Report actual coverage326train/41validation and3637/336windows.

Apply upstream bilateral_filter to each64-frame window before upstream
mean_std_swap, matching upstream load_data order (the old adapter smoothed whole
clips instead). Fit all speaker/listener/audio statistics only on train windows.
Preserve the unusual upstream normalization formula, document scales and compare
to mean baselines; do not silently replace it with global standard deviation.
The cached rawDECA and alignedmel features need no new GPU extraction.

VQ uses both32-frame halves of listener windows. Predictor uses the unchanged
upstream create_data_vq (past32, speaker40, audio160, next4VQ tokens). Retain the
previously approved source-only4t audio alignment at inference.

Upstream training permutes full batch indices without replacement within each
epoch. Follow that scheme, record omitted training tail counts; validation uses
all samples with sample-weighted aggregation. Full data splits override upstream
random window train/val splitting to avoid leakage.

## Training conformance and logging

Call setup_vq_transformer/setup_model and KEEP the returned ScheduledOptim:
VQ AdamW, baseLR2.0,d_model256; predictor Adam, baseLR0.01,d_model200;
warmup100000,beta(.9,.98),eps1e-9. No replacement optimizer, no undocumented
clipping, no AMP/TF32. Record gradients and reject nonfinite loss/gradients.
Preflight compares parameter/optimizer updates with an independent call to
upstream training functions on identical real batches and RNG state.

Log VQ total, expression/head/jaw reconstruction, quantization loss, code usage,
reconstruction-vs-train-mean baseline and temporal variance. Log predictorCE,
perplexity and LR. Validation must not perturb training RNG or BatchNorm stats.
For deterministic predictor validation use explicit masks0..4, weighted
1/21,1/21,1/21,1/21,17/21: the expectation of upstream randommask0..20,
where4..20 expose all4historytokens. Also report fully maskedCE separately.
This deterministic validation adaptation is explicit, not claimed as upstream's
exact random validation trajectory.

max_steps200000 per stage; warmup must complete before plateau stopping;
minimum150000 updates, validation every1000 (plus initialization and diagnostic
early points), plateau patience20 validations. Do not shorten warmup because a
convenient wall-clock budget expired. Record max-budget termination distinctly
from convergence. Estimate duration from actual throughput, not prior20k runs.

## Checkpoints and resume

Atomically save best and last, last at every1000 updates, validation and phase
end; periodically keep versioned recovery checkpoints. Preserve model buffers,
optimizer tensors/param_groups, ScheduledOptim parameters/n_steps, globalstep,
stage, sampler epoch/permutation/cursor/RandomState, Python/NumPyglobal/TorchCPU/
CUDA RNG, every model-local NumPy RandomState (FACTModel.rng included), selection
history/best scores, and data/code/config/statistics fingerprints.

Store the initial model separately. End-of-stage last must be written BEFORE
loading best; last must contain the actual final-step model. Verify checkpoint
hashes and internal steps independently. Existing run code must be snapshotted;
resume refuses mismatched data/code/config. SIGTERM/STOP sets a flag, saves at the
next safe optimizer boundary and exits without starting another stage.

Test real model uninterrupted N+M updates vs N/save/new model/load/M: exact model
and Adam state, scheduler/sampler/RNG and next loss. Test bothVQ and predictor,
including model-local mask RNG and validation boundary effects. A mismatch blocks
formal training. Smaller smoke-only controls live outside formal outputs.

## Learning checks, not execution-only acceptance

First perform diagnostic overfit on8train windows, no test/val selection; label
any diagnostic shortened warmup/optimizer explicitly and NEVER reuse diagnostic
weights as a formal baseline. Require meaningful reconstruction improvement and
nonzero parameter changes. Validate model/optimizer parity and resume tests.

VQ checkpoint selection: minimum validation totalloss (same upstream objective),
also retain best-reconstruction diagnostic checkpoint. A formal seed can proceed
only if selected step>0, validation objective improves relative to initialization,
reconstruction beats the train-mean baseline by at least5%, and validation code
usage>=2. Report variance ratios and per-group errors; no test-driven thresholds.
Fully freeze these checks before test. Failure stops the pipeline, leaves last
and diagnostics, and is not published as a successful baseline.

Predictor selection: deterministic weightedvalidationCE minimum; require
improvement over initialization and at least1%relative improvement over a
training-token-unigram baseline. Compute counts from the effective training rows
(the fullbatch prefix, excluding the permanent dropped training tail), apply
Laplacealpha1 over all200codes, and score all validation target tokens. This
prevents unseen validation codes from producing infinite baselineCE.

Source-only validation rollout uses the original46clips, original full durations,
and removes only the32-frame artificial history and computed tail padding. Require
finite outputs and mean-per-clip temporal-variance ratios(pred/GT) in[0.01,10]
for expression and globalhead, with rawFP64 coefficients. A GT aggregate variance
<=1e-12 gives N/A rather than an arbitrary division. Report jaw variance without a
gate. Sampling seed is fixed per training seed; no repeated-draw selection.
Report all failures rather than quietly dropping clips. Do not use a random or
GT-selected checkpoint to fill a table.

Initialization includes a real step0 validation checkpoint; neither step0 nor
step1 may masquerade as successful learning. Training and test inference preserve
the approved output fields and no-GT-input checks.

## Test and publication

Only after all seeds pass validation checks, freeze all checkpoint hashes and
issue a test release manifest. One seed-matched sample per clip, no GT selection,
32standardizedzero history, source-only input, full original duration43clips.
DECA raw50+head3+jaw3, existing metric definitions and43/42/41coverage. Save every
seed's NPZ/per-clip scores; mean/std summarize scores, not averaged motion.

The primary seed137 can be linked as learning2listen/retrained in existing pages;
add links/table for all seeds and mean/std. Publish only with learning, execution,
provenance and browser checks passed. Retain older4official person-transfer rows
and distinguish local training from them. RestoreGPU7 for inference and update
the root/evaluation/compare indexes and handoff with final state and limitations.

## Implementation calls left to the builder

Module boundaries, JSON layout, immutable code snapshot plumbing and small
preflight step counts are engineering choices. Tests must use real model steps,
not only serialization round trips or mirror-implementation arithmetic tests.
No external author communication or model uploads are authorized.
