"""Aggregate durable raw output counters and three-node resource telemetry."""
import argparse,json,time,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--preview',required=True);a=p.parse_args()
    run=Path(a.run).resolve();preview=Path(a.preview).resolve();preview.mkdir(parents=True,exist_ok=True)
    config=read(run/'config.json')
    history=[]
    while not (run/'STOP_MONITOR').exists():
        now=time.time();index=read(run/'queue/index.json');branches={}
        for branch in ['visual','audio','asr']:
            records=[read(p) for p in (run/'raw'/branch/'state').glob('*.json')]
            counts=Counter()
            for r in records:counts.update(r.get('counts',{}))
            branches[branch]=dict(counts=counts,requests=sum(counts.values()),shards_done=sum(bool(r.get('done')) for r in records),
                shards_started=len(records),total_shards=len(index.get('shards',{})))
        history.append((now,{k:v['requests'] for k,v in branches.items()}));history=[x for x in history if now-x[0]<900]
        base=next((x for x in history if now-x[0]<=300),history[0]);elapsed=now-base[0]
        for k,v in branches.items():
            v['recent_requests_per_s']=(v['requests']-base[1][k])/elapsed if elapsed>20 else None
        nodes={p.parent.name:read(p) for p in (run/'nodes').glob('*/latest.json')}
        workers={p.stem:read(p) for p in (run/'workers').glob('*.json')}
        repairs=read(run/'raw/asr_alignment_repairs/status.json')
        branches['asr']['alignment_repairs_complete']=repairs.get('complete',0)
        notes=[]
        for n,r in nodes.items():
            if now-r.get('utc',0)>90:notes.append(n+': telemetry stale')
        for name,w in workers.items():
            if w.get('status') in ('running','inference'):
                wait=w.get('cumulative_input_wait_s',0);infer=w.get('cumulative_inference_s',0)
                if infer>60 and wait/(infer+wait)>.10:notes.append(name+': input preparation wait above 10%; inspect decode and prefetch')
        # A finished fast branch releases its GPUs to remaining visual work.
        # The owning supervisor changes only its own verified worker processes.
        handoffs=read(run/'handoffs.json')
        visual_left=branches['visual']['total_shards']-branches['visual']['shards_done']
        if visual_left>sum(w.get('branch')=='visual' for w in workers.values()):
            for name,w in workers.items():
                if w.get('branch') not in ('audio','asr') or w.get('status')!='finished_available_shards':continue
                if name in handoffs:continue
                control=read(run/'control.json');control[name]=dict(batch=64,memory=.85,cuda_graph=True)
                atomic_json(run/'control.json',control)
                assignments=read(run/'allocations.json');assignments[name]='visual'
                atomic_json(run/'allocations.json',assignments)
                handoffs[name]=dict(utc=now,from_branch=w['branch'],to_branch='visual',mode='supervisor_owned_process_handoff')
                atomic_json(run/'handoffs.json',handoffs)
        summary=dict(run_id=run.name,updated_at=now,index=index.get('counts',{}),branches=branches,nodes=nodes,workers=workers,
                     optimization_notes=notes,handoffs=handoffs,alignment_repairs=repairs,raw_only=True,event_merge=False,human_ground_truth=False,training_admitted=False)
        atomic_json(run/'status.json',summary);atomic_json(preview/'status.json',summary)
        with (run/'throughput_history.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now,branches=branches))+'\n')
        time.sleep(15)

if __name__=='__main__':main()
