#!/usr/bin/env python3
"""One CPU-only corrected-crop GLARE measurement; caller holds the output lock."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import contextlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--row', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--resources', type=Path)
    p.add_argument('--serve', action='store_true')
    args = p.parse_args()
    import cv2
    cv2.setNumThreads(1)
    from listen_benchmark.j_annotation import read, write
    from listen_j_annotation_nod import annotate
    from run_glare_nodding_pilot import module, FACE, NOD, MODEL, PROFILES, intervals
    with contextlib.redirect_stdout(sys.stderr):
        face = module('corrected_crop_face', FACE); nod = module('corrected_crop_nod', NOD)
    def execute(row_path, output, resources, decoded_dir=None):
        row = read(row_path)
        if row['source']['crop_proposal']['revision'] != 'quality-bounded-crop-v1-20261030':
            raise ValueError('Expected reviewed corrected crop')
        with contextlib.redirect_stdout(sys.stderr):
            result = annotate(row, face, nod, PROFILES['recall'], MODEL, intervals,
                              Path(resources), fast_crop=True,decoded_dir=decoded_dir)
        if Path(output).exists():
            raise ValueError('Refusing to replace an existing nod measurement')
        write(output, result)
    if args.serve:
        # EOF closes the service when its owning CPU preparer exits. Model
        # imports are reused, while each clip still gets its own face detector.
        for line in sys.stdin:
            try:
                request=json.loads(line)
                execute(request['row'],request['output'],request['resources'],request.get('decoded_dir'))
                reply=dict(ok=True,output=request['output'])
            except Exception as exc:reply=dict(ok=False,error=repr(exc))
            print(json.dumps(reply),flush=True)
    else:
        if not all((args.row,args.output,args.resources)):p.error('row/output/resources are required')
        execute(args.row,args.output,args.resources)


if __name__ == '__main__':
    main()
