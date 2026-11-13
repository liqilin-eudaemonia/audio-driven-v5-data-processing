"""Original pinned evaluator; optional process-resident original model instances."""
from runtime import *
import importlib.util
from contextlib import contextmanager
_pipeline=None
_instances={}
_counts={}

@contextmanager
def original_metric_precision():
    """Match the original standalone scorer, rather than generator device flags."""
    import torch
    previous=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=True):
            yield
    finally:torch.backends.cuda.matmul.allow_tf32=previous

def install_backend_cache():
    from evaluation.video.paired_metrics import backends
    if getattr(backends,'_reference_resident',False):return
    def cached_factory(label,original):
        def construct(*args,**kwargs):
            key=label+json.dumps([args,kwargs],sort_keys=True,default=str)
            if key not in _instances:
                _instances[key]=original(*args,**kwargs);_counts[label]=_counts.get(label,0)+1
                save(WORK/'SCORER_RESIDENCY.json',dict(pid=os.getpid(),constructor_loads=_counts,instances=len(_instances),at=time.time()))
            return _instances[key]
        return construct
    backends.Networks=cached_factory('Networks',backends.Networks)
    backends.Faces=cached_factory('Faces',backends.Faces)
    backends._reference_resident=True

def run_pipeline(args,resident=False):
    global _pipeline
    activate_readers()
    if resident:install_backend_cache()
    if _pipeline is None:
        p=WORK/'pinned_metrics/p2_pipelined_video_metrics.py'
        spec=importlib.util.spec_from_file_location('pinned_original_metric_pipeline',p)
        _pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(_pipeline)
        sys.path.insert(0,str(ROOT/'src'))
    previous=sys.argv;sys.argv=[str(WORK/'metric_launcher.py'),*args]
    try:
        try:
            with original_metric_precision():_pipeline.main()
        except SystemExit as e:return e.code or 0
        return 0
    finally:sys.argv=previous

if __name__=='__main__':raise SystemExit(run_pipeline(sys.argv[1:]))
