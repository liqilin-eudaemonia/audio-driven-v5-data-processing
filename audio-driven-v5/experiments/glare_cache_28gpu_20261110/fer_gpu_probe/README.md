# GPU FER execution probe

Three original selected pair clips, 283 frames. Same FER and MTCNN weights, preprocessing and 32-frame detection batches. CPU: 10.117 s; GPU FP32 batch: 1.685 s. Face counts, integer boxes and emitted emotion probabilities all matched on these frames. GPU TF32 paths did not match (maximum emotion delta 0.29) and are not used. This is stage timing, not full-pipeline speedup.

The opt-in cluster `execution.gpu_fer=true` keeps native MediaPipe and reaction assembly, and runs MTCNN on the same physical GPU as its LIA worker. Both TF32 switches are disabled. TensorFlow remains CPU for the native TFLite model. Original production cache files and default config remain unchanged pending end-to-end comparison.

End-to-end test: `../gpu_fer_640/config.json`; same 640 pair / 4,130 prosody chunk cohort as the completed CPU runs. GPU test outputs are isolated. Compare saved reaction arrays and masks before adopting.
