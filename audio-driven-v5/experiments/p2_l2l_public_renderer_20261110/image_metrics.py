"""Original ViCo image metrics for a named L2L+GPAvatar output directory."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
import p2_l2l_gpavatar as gp
import p2_vico_official_metrics as official

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--child');parser.add_argument('--index',type=int)
    args=parser.parse_args()
    protocol=gp.read(args.work/'protocol.json')
    official.ARM=protocol['method'];official.__file__=__file__
    if args.child:
        if args.child in ['fid','csim']:
            gp.guard()
            # Same networks/functions; smaller batch bounds allocator memory.
            native=runpy.run_path
            def bounded(path,*a,**kwargs):
                if '--batch_size' in sys.argv:
                    sys.argv[sys.argv.index('--batch_size')+1]='64'
                return native(path,*a,**kwargs)
            runpy.run_path=bounded
        official.child(args.work,args.child,args.index)
    else:
        official.run(args.work)
        result=gp.read(args.work/'RESULTS.json')
        result.update(model=protocol['label'],generation=dict(width=512,height=512,fps=30),
            image_inference_batch=64,protocol_sha256=gp.sha(args.work/'protocol.json'),
            official_metric_script_sha256=gp.sha(ROOT/'scripts/p2_vico_official_metrics.py'))
        gp.save(args.work/'RESULTS.json',result)
        gp.save(args.work/'COMPLETE.json',dict(passed=True,pairs=43,results_sha256=gp.sha(args.work/'RESULTS.json')))
