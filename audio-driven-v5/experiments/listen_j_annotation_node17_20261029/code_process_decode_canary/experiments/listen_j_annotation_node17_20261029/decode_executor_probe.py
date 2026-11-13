"""Matched CPU-only executor probe; no production outputs/configuration mutations."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import sys,json,time,hashlib,multiprocessing
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'src'))

def decode(row):
    from listen_benchmark.visual_native8 import prepare_media
    from listen_benchmark.frame_crop import crop_original_frame
    started=time.monotonic();media=prepare_media(row['source'],frame_crop=crop_original_frame)[8]
    signature=hashlib.sha256(json.dumps(media['meta'],sort_keys=True).encode()).hexdigest()
    return {'sample_id':row['sample_id'],'seconds':time.monotonic()-started,'signature':signature,'frames':len(media['images'])}

def main():
    rows=json.loads((WORK/'jobs/naturalistic_V03_S0130_I00000372.json').read_text())['rows'][100:116]
    destination=WORK/'monitor_snapshots/decode_executor_probe.json'
    if destination.exists():raise SystemExit('Report already exists')
    report={'started_at':time.time(),'workers':8,'samples':len(rows),'scope':'CPU-only same original input, same crop/frame/JPEG code. Thread/process ABBA; production remains running.','arms':[]}
    for name in ['thread','process','process','thread']:
        t=time.monotonic();pool=(ThreadPoolExecutor(max_workers=8) if name=='thread' else ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')))
        with pool:results=list(pool.map(decode,rows))
        arm={'executor':name,'wall_s':time.monotonic()-t,'results':results};report['arms'].append(arm)
        expected=report['arms'][0]['results'];assert [(r['sample_id'],r['signature'],r['frames']) for r in results]==[(r['sample_id'],r['signature'],r['frames']) for r in expected]
        report['all_signatures_equal']=True;destination.write_text(json.dumps(report,indent=2))
        print(json.dumps({'executor':name,'wall_s':arm['wall_s'],'signatures_equal':True}),flush=True)
    report.update(complete=True,finished_at=time.time());destination.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
