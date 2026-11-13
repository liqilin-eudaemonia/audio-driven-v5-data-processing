"""Run frozen official ViCo metrics under the additional method's directory."""
import argparse
import sys
from pathlib import Path
from vico_more_common import ROOT, read, save, sha, gpu_guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--child');p.add_argument('--index',type=int);a=p.parse_args()
    sys.path.insert(0,str(ROOT/'scripts'))
    import p2_vico_official_metrics as official
    spec=read(a.work/'protocol.json')
    official.ARM=spec['method']
    # Its child invocations return here, so method selection survives subprocesses.
    official.__file__=__file__
    if a.child:
        official.child(a.work,a.child,a.index)
    else:
        gpu_guard(12)
        official.run(a.work)
        result=read(a.work/'RESULTS.json')
        result.update(model=spec['method']+' local ViCo retraining + PIRender EMA',
            generation=dict(width=256,height=256,fps=30),
            protocol_sha256=sha(a.work/'protocol.json'),
            original_metric_source_sha256=sha(ROOT/'scripts/p2_vico_official_metrics.py'))
        save(a.work/'RESULTS.json',result)
        save(a.work/'IMAGE_COMPLETE.json',dict(success=True,pairs=43,
            results_sha256=sha(a.work/'RESULTS.json'),original_metric_source_unchanged=True))


if __name__=='__main__':main()
