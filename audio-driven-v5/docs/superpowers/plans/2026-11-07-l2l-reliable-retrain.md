# L2L reliable restart implementation plan

> Superseded for execution: all L2L training stopped pending paper-grounded protocol audit on2026-11-07. Historical design only; do not use this plan to auto-resume. See experiments/l2l_literature_protocol_audit_20261106/PAPER_GROUNDED_REPRODUCTION_AUDIT_20261107.md.

User authorized reliable retraining and temporary node11GPU7 use. The design is
`docs/superpowers/specs/2026-11-07-l2l-reliable-retrain-design.md`.

1. Read/freeze upstream optimizer, model, normalization and masking behavior.
   Independently review design and resolve acceptance criteria before test.
2. Create a new adapter and directory, reusing raw397pair features and approved
   split. Match upstream window smoothing and train-only statistics. Save arrays,
   source/code hashes and actual window/tail coverage.
3. Keep factory ScheduledOptim and epoch batch shuffling. Add atomic best/last,
   optimizer/scheduler/globalandmodel-localRNG/sampler recovery, safe stop,
   decomposed validation losses and train-mean/unigram baselines.
4. OnGPU7, verify real upstream training-step parity, nontrivial smallbatch
   learning, and uninterrupted-vs-save/reload resume parity for both stages.
   Correct code and rerun failing preflights before freezing formal code.
5. Snapshot code/configs, run primary137 then replications271/811 with official
   warmup, preregistered gates and heldout validation. Do not test on failure.
6. Freeze eligible checkpoints; run single-candidate source-onlytest43 per seed,
   score and report individual plus mean/std. Publish primary per-case data and
   replication tables with provenance, audit and browser checks.
7. Record final state in indexes/handoff and releaseGPU7. If a learning gate fails,
   retain evidence and report a failed experiment instead of a completed baseline.

The separately referenced writing-plans skill is absent from the local installed
skill catalog; this task uses this explicit plan and the available design review.
This absence does not prevent implementing the user-authorized repair.
