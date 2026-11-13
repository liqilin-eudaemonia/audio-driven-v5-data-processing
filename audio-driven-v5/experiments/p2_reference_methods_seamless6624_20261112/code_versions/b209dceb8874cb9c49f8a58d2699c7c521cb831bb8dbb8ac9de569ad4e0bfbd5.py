"""Original CPU-prefetch pipeline with a pinned evaluator package and original data root."""
from runtime import *
activate_readers()
import importlib.util
p=WORK/'pinned_metrics/p2_pipelined_video_metrics.py'
spec=importlib.util.spec_from_file_location('pinned_original_metric_pipeline',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# Preserve module code; its original ROOT-relative paths target the shared project sources.
sys.path[:0]=[str(ROOT/'src')]
module.main()
