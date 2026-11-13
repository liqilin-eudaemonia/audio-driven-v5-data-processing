"""Equal-input CPU preparation benchmark, including process IPC and payload hashes."""
import os,sys,time,json,hashlib,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write

def init():
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def prepare_task(row,old_work):
 from listen_benchmark.j_omni import prepare_input
 started=time.monotonic();out=prepare_input(row,old_work,W,W/'resources_node4');out['_prepare_seconds']=time.monotonic()-started;return out

def main():
 cfg=read(W/'config.json');checks=read(W/'validation/production_v2_24gpu_audit.json')['checks'];items=[]
 for e in checks:
  r=read(e['file']);s=r['source'];origin=read(W/'job_origins'/f"{s['pair']}.json")
  items.append((dict(sample_id=r['sample_id'],source=s),origin['old_work'],r['input_evidence']))
 # Same interleaved source order for every backend; no changed media or prompt.
 reports=[]
 for kind,workers in [('thread',2),('process',4),('process',8)]:
  pool=ThreadPoolExecutor(max_workers=workers) if kind=='thread' else ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=init)
  with pool:
   for repeat in range(2):
    started=time.monotonic();futures={pool.submit(prepare_task,row,old):(row,evidence) for row,old,evidence in items};records=[]
    for f in as_completed(futures):
     row,ev=futures.pop(f);out=f.result();assert out['evidence']['video_frames_sha256']==ev['video_frames_sha256'] and out['evidence']['audio_encoder_sha256']==ev['audio_encoder_sha256']
     assert hashlib.sha256(out['frames'].tobytes()).hexdigest()==ev['video_frames_sha256']
     assert hashlib.sha256(out['audio'].tobytes()).hexdigest()==ev['audio_encoder_sha256']
     records.append(dict(sample_id=row['sample_id'],prepare_seconds=out['_prepare_seconds'],received_at_s=time.monotonic()-started));del out
    elapsed=time.monotonic()-started;r=dict(backend=kind,workers=workers,repeat=repeat,seconds=elapsed,clips=len(items),clips_per_second=len(items)/elapsed,all_hashes_equal=True,records=records)
    reports.append(r);write(W/'performance_check/preparation_benchmark.json',dict(at=time.time(),reports=reports,complete=False));print(json.dumps({k:v for k,v in r.items() if k!='records'}),flush=True)
 write(W/'performance_check/preparation_benchmark.json',dict(at=time.time(),reports=reports,complete=True))
if __name__=='__main__':main()
