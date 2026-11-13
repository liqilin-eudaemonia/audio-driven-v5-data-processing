#!/usr/bin/env python3
"""Resident batch interface for official GLARE model inference."""
import argparse,json,time
from pathlib import Path
from glare_resident_inference import ResidentGLARE

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--jobs',type=Path,required=True,help='JSONL rows with reference, audio, output and optional seed/nfe')
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--decode-batch',type=int,default=8)
    parser.add_argument('--serial-export',action='store_true')
    a=parser.parse_args()
    if a.decode_batch<1:raise ValueError('decode-batch must be positive')
    jobs=[json.loads(line) for line in a.jobs.read_text().splitlines() if line.strip()]
    started=time.perf_counter()
    engine=ResidentGLARE(a.checkpoint,a.cache_dir)
    results=engine.run_jobs(jobs,a.decode_batch,not a.serial_export)
    result=dict(model_load_seconds=engine.load_seconds,total_seconds=time.perf_counter()-started,jobs=results)
    a.jobs.with_suffix('.result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
