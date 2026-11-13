"""Fresh official ViCo image scoring of verified AF videos."""
import argparse,sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'compare/deployment'),str(ROOT/'scripts')]
from vico_more_common import read,save,sha,gpu_guard
import p2_vico_official_metrics as official
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=OUT/'image_eval');p.add_argument('--child');p.add_argument('--index',type=int);a=p.parse_args()
official.ARM='avatarforcing';official.__file__=__file__
if a.child:
    if a.child == 'fid':
        audit=read(OUT/'FID_CACHE_PARITY.json')
        assert audit['passed'] and audit['source_sha256']==sha(OUT/'fid_cache.py')
        from fid_cache import install
        install()
    if a.child in ['fid','csim']:
        gpu_guard(16)
        import torch
        torch.cuda.set_per_process_memory_fraction(.10,0)
        native=runpy.run_path
        def bounded(path,*args,**kwargs):
            if '--batch_size' in sys.argv:sys.argv[sys.argv.index('--batch_size')+1]='64'
            return native(path,*args,**kwargs)
        runpy.run_path=bounded
    official.child(a.work,a.child,a.index)
else:
    official.run(a.work)
    r=read(a.work/'RESULTS.json');r.update(model='Avatar Forcing official weights, audio-only adaptation',generation=dict(width=512,height=512,fps=25),image_inference_batch=64)
    save(a.work/'RESULTS.json',r);save(a.work/'COMPLETE.json',dict(passed=True,results_sha256=sha(a.work/'RESULTS.json')))
