import json,sys,os,time,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write
from listen_benchmark.j_visual_gate import initialize_cpu,MODEL,EVIDENCE
from listen_visual_quality.measurement import measure_row
from listen_visual_quality.decision import decide,evidence_index
W=ROOT/'experiments/listen_j_omni_20261030';DEST=W/'performance_check/queued4_quality_20261031'
def measure(task):
 cfg=read(W/'visual_quality/config.json');record=measure_row(task['row'],cfg['contract']['policy'],str(ROOT),str(MODEL));decision=decide(record,cfg['contract']['policy'],evidence_index(read(EVIDENCE)))
 write(DEST/'quality'/f"{task['row']['sample_id']}.json",dict(record=record,decision=decision));return decision
if __name__=='__main__':
 initialize_cpu();source=W/'performance_check/queued4_single_sequence_canary';tasks=read(source/'tasks.json')
 with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_cpu) as pool:ds=list(pool.map(measure,tasks))
 kept=[t for t,d in zip(tasks,ds) if d['decision']=='retain'];refs=read(source/'reference_captions.json')
 write(DEST/'tasks.json',kept);write(DEST/'reference_captions.json',{t['row']['sample_id']:refs[t['row']['sample_id']] for t in kept})
 result=dict(at=time.time(),retained=len(kept),decisions=ds,engine_max_num_seqs=1,requests_per_call=[1,4,4,1],production_changed=False)
 write(DEST/'quality_review.json',result);print(json.dumps(dict(retained=len(kept),total=len(tasks))))
