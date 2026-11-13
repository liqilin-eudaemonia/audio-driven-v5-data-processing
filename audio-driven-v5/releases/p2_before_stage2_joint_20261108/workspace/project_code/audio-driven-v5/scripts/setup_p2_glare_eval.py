#!/usr/bin/env python3
"""Create isolated CPU detectors and prefetch pinned public evaluation assets.
Run with the existing P2 Python; its installed packages are never changed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_glare_eval.common import read,sha,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--python310',default='/usr/bin/python3.10')
    parser.add_argument('--uv',default=shutil.which('uv') or '/private/software/conda/bin/uv')
    args=parser.parse_args()
    detector_python=ROOT/'.venv-p2-glare-detector/bin/python'
    lock=read(ROOT/'configs/p2_glare_upstream.lock.json')
    if not args.check_only:
        for name,commit,url in [('glare',lock['glare_commit'],'https://github.com/lzk901372/glare.git'),
            ('visual_reaction_annotation',lock['detector_commit'],'https://github.com/lzk901372/visual_reaction_annotation.git')]:
            destination=ROOT/'related'/name
            if not destination.exists():
                subprocess.run(['git','clone',url,str(destination)],check=True)
                subprocess.run(['git','-C',str(destination),'checkout','--detach',commit],check=True)
        if not detector_python.exists():
            subprocess.run([args.uv,'venv','--python',args.python310,str(detector_python.parent.parent)],check=True)
        subprocess.run([args.uv,'pip','install','--python',str(detector_python),'-r',str(ROOT/'configs/p2_glare_detector_environment.freeze.txt'),
            '--extra-index-url','https://download.pytorch.org/whl/cpu','--index-strategy','unsafe-best-match'],check=True)
        fid_package=ROOT/'.eval_deps/pytorch-fid-0.3.0'
        if not (fid_package/'pytorch_fid/__init__.py').is_file():
            subprocess.run([args.uv,'pip','install','--python',sys.executable,'--no-deps','--target',str(fid_package),'pytorch-fid==0.3.0'],check=True)
        asset=read(ROOT/'configs/p2_glare_assets.lock.json')
        from p2_glare_eval.scoring import FID_WEIGHT
        if not FID_WEIGHT.exists():
            FID_WEIGHT.parent.mkdir(parents=True,exist_ok=True)
            partial=FID_WEIGHT.with_suffix('.download')
            try:
                urllib.request.urlretrieve(asset['fid_url'],partial)
                if sha(partial)!=asset['fid_inception_sha256']:raise ValueError('Downloaded FID weight hash differs')
                os.replace(partial,FID_WEIGHT)
            finally:
                partial.unlink(missing_ok=True)
    from p2_glare_eval.upstream import verify_sources
    from p2_glare_eval.scoring import fid_identity
    from p2_glare_eval.pipeline import detector_env,DEFAULT_RUNTIME
    upstream=verify_sources();fid=fid_identity()
    for executable in ('ffmpeg','ffprobe'):
        if not shutil.which(executable):raise RuntimeError(f'{executable} is required')
    runtime=json.loads(subprocess.check_output([str(detector_python),str(ROOT/'scripts/p2_glare_eval.py'),'detector-runtime'],
        text=True,env=detector_env()))
    atomic_json(DEFAULT_RUNTIME,runtime,immutable=True)
    print(json.dumps(dict(status='ready',upstream_commits={k:v for k,v in upstream.items() if k!='files'},
        detector_runtime=str(DEFAULT_RUNTIME),fid=fid,training_environment_modified=False),indent=2))


if __name__=='__main__':main()
