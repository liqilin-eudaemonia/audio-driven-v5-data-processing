# RealTalk boundary-v2 pilot

User-authorized correction of the reviewed v1 boundary issues, evaluated on the complete -1wCGAxqT_4 source, not a subset or the approved-reference bypass.

Source time is normalized to 25 fps from zero and actual decoded frames are counted. Black classification is per normalized frame, including EOF. Real decoded audio determines the last complete supported frame; no zero padding pretends to be real audio. Every frame remains in the ledger, including black, overlap, missing audio, and unknown roles.

Known-role labels are evidence, not semantic ground truth. Median-smoothed observed per-person activity may backdate sustained takeover candidates; short opposite turns returning to an active speaker remain backchannel candidates. Missing scores never prove silence. Formal takeover candidates require 0.8s evidence; 0.8s is an explicit pilot heuristic, not a universal linguistic definition. Sustained overlap around takeover is retained separately, and the before/after candidate intervals remain available. All role corrections have visible provenance and remain reviewable; training_admitted=false.

Only stable geometry changes with image-change support are extra hard cuts. Geometry cut estimates within two frames of an existing scene cut snap to that cut; distinct visual scene cuts and black intervals are never blindly merged. Native crop uses per-pair identity boxes and horizontal or vertical partition, with missing/low-coverage crops flagged.

Fresh Whisper large-v3 word estimates replace coarse segment-end ASR for length partition. Prefer sentence ends, pauses in word gaps, word ends, and only finally frame fallbacks. Scene/takeover boundaries override sentence completeness; maximum remains 11s and short tails are retained. ASR words still require review, especially overlaps and chunk edges.

The manifest cache fingerprint includes code, policy, source metadata/stat, annotation, prepared video/audio, and fresh word outputs. The legacy directory and its 692 manifests remain untouched.

Review: all candidate pairs plus the complete non-candidate ledger. Original mixed audio is explicitly labeled as boundary-review audio, not isolated listener audio. A single source player and its synchronized listener canvas show boxes, interval boundaries, and before/after context. No full-dataset replacement or final SyncNet/clean-audio admission occurs in this pilot.
