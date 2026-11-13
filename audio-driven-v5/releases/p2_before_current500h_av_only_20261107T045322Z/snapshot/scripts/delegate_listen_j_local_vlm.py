#!/usr/bin/env python3
"""Delegate the stopped local VLM queue; keep local CPU nod ownership intact."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,locked,acquire,result_path
from listen_benchmark.j_quality_pipeline import live

BASE=ROOT/'experiments/listen_j_annotation_three_nodes_20261030'
TX=BASE/'local_vlm_delegation_20261030'
SOURCE=ROOT/'experiments/listen_j_annotation_node12_20261030'
QUALITY=ROOT/'data_corpora/seamless_listen_j_visual_quality_v1_20261030'


def entries(raw):
    rows={}
    for line in raw.splitlines():
        r=json.loads(line)
        if r['job_id'] in rows and rows[r['job_id']]!=r['count']:raise ValueError('Conflicting count')
        rows[r['job_id']]=r['count']
    return rows


def assert_stopped():
    s=read(SOURCE/'supervisor.json')
    if not (SOURCE/'STOP_SUPERVISION').exists() or live(s) or live(read(SOURCE/'watchdog.json')):
        raise RuntimeError('Local annotation GPU supervisor must remain stopped')
    if any(live(c) for n,c in s['children'].items() if n.startswith('gpu-')):
        raise RuntimeError('Local GPU annotation worker still live')


def delegate_nod(work,job):
    p=work/'nod/completed'/f'{job}.json'
    record={'job_id':job,'clips':0,'status':'delegated_branch_not_measurement',
        'branch':'nod','owner_work':str(SOURCE),'reason':'Only visual inference moved; original CPU nod continues'}
    if p.exists() and read(p)!=record:raise ValueError('Unexpected destination nod receipt')
    write(p,record)


def main():
    TX.mkdir(exist_ok=True)
    with locked(TX/'transaction.lock'),locked(SOURCE/'queues/vlm/state.lock'):
        assert_stopped();scope=read(BASE/'nodes.json');raw=(SOURCE/'ready.jsonl').read_bytes();state=read(SOURCE/'queues/vlm/state.json')
        if not (TX/'plan.json').exists():
            write(TX/'scope_before.json',scope);write(TX/'source_vlm_state_before.json',state)
            (TX/'source_ready_before.jsonl').write_bytes(raw)
            all_rows=entries(raw);done={p.stem for p in (SOURCE/'vlm/completed').iterdir() if p.suffix=='.json'}
            if not done<=all_rows.keys():raise ValueError('Completion outside source scope')
            if sum(all_rows[j] for j in done)!=state['completed_clips']:raise ValueError('Completion counter mismatch')
            pending={j:n for j,n in all_rows.items() if j not in done}
            remaining={n:scope['nodes'][n]['scope_clips']-read(Path(scope['nodes'][n]['work'])/'queues/vlm/state.json')['completed_clips'] for n in ('node17','node8')}
            assigned={n:{} for n in remaining}
            for job,count in sorted(pending.items(),key=lambda x:(-x[1],x[0])):
                node=min(remaining,key=remaining.get);assigned[node][job]=count;remaining[node]+=count
            write(TX/'plan.json',{'at':time.time(),'assigned':assigned,'inflight_before':sorted(set(state['inflight'])|set(state['failed'])),
                'completed_sources':len(done),'completed_clips':state['completed_clips'],
                'delegated_sources':len(pending),'delegated_source_clips':sum(pending.values())})
        plan=read(TX/'plan.json');before=read(TX/'scope_before.json')
        for job in state['inflight']:
            handle=acquire(SOURCE/'queues/vlm/claims'/f'{job}.lock',False)
            if handle is None:raise RuntimeError('Local source still claimed: '+job)
            handle.close()
        reuse=read(TX/'reused_results.json') if (TX/'reused_results.json').exists() else {}
        for node,jobs in plan['assigned'].items():
            work=Path(before['nodes'][node]['work'])
            for job in jobs:
                if not (work/'jobs'/f'{job}.json').exists():raise ValueError('Job catalog missing source')
                delegate_nod(work,job)
                # Only inflight sources can contain results without a source receipt.
                if job in plan['inflight_before']:
                    for row in read(SOURCE/'jobs'/f'{job}.json')['rows']:
                        sid=row['sample_id'];old=result_path(SOURCE,'vlm',sid)
                        if not old.exists():continue
                        label=read(old)
                        if label['source']!=row['source']:raise ValueError('Existing label source differs')
                        target=result_path(work,'vlm',sid);target.parent.mkdir(parents=True,exist_ok=True)
                        if target.exists() or target.is_symlink():
                            if target.resolve()!=old.resolve():raise ValueError('Existing destination result differs')
                        else:target.symlink_to(old.resolve())
                        reuse[sid]={'original':str(old),'link':str(target),'sha256':hashlib.sha256(old.read_bytes()).hexdigest()}
        write(TX/'reused_results.json',reuse)
        # Retire only local GPU claims. Nod state/journal and all existing results stay intact.
        state.update(offset=len(raw),inflight={},failed={},delegation_transaction=str(TX))
        write(SOURCE/'queues/vlm/state.json',state)
        for node,jobs in plan['assigned'].items():
            work=Path(before['nodes'][node]['work'])
            with locked(work/'ingest.lock'),locked(work/'queues/vlm/state.lock'),locked(work/'queues/nod/state.lock'):
                current=(work/'ready.jsonl').read_bytes();present=entries(current)
                overlap=set(jobs)&set(present)
                receipt=TX/(node+'_append.json')
                if overlap:
                    if overlap!=set(jobs) or not receipt.exists():raise ValueError('Partial or unowned queue append')
                else:
                    (TX/(node+'_ready_before.jsonl')).write_bytes(current)
                    write(receipt,{'state':'prepared','prefix_sha256':hashlib.sha256(current).hexdigest(),'prefix_bytes':len(current)})
                    extra=b''.join((json.dumps({'job_id':j,'count':count})+'\n').encode() for j,count in jobs.items())
                    temp=work/'ready.delegate.tmp';temp.write_bytes(current+extra);temp.replace(work/'ready.jsonl')
                r=read(receipt);r['state']='committed';write(receipt,r)
        scope=before;scope['schema']='listen-j-branch-delegated-scope-v2'
        overrides=scope.setdefault('branch_ownership_overrides',{})
        for node,jobs in plan['assigned'].items():
            for job in jobs:overrides[job]={'vlm':scope['nodes'][node]['work'],'nod':str(SOURCE)}
        scope['total_gpus']=16;scope['nodes']['node12']['gpu_ids']=[]
        scope['gpu_delegation_transaction']=str(TX)
        for node,cfg in scope['nodes'].items():
            cfg['branch_scope_clips']={'vlm':cfg['scope_clips'],'nod':cfg['scope_clips']}
            if node=='node12':cfg['branch_scope_clips']['vlm']=plan['completed_clips']
            else:cfg['branch_scope_clips']['vlm']+=sum(plan['assigned'][node].values())
        assert sum(c['branch_scope_clips']['vlm'] for c in scope['nodes'].values())==scope['total_clips']
        for p in (BASE/'nodes.json',QUALITY/'annotation_nodes.json'):write(p,scope)
        report={'at':time.time(),'state':'committed','delegated_sources':plan['delegated_sources'],
            'delegated_source_clips':plan['delegated_source_clips'],'preserved_partial_results':len(reuse),
            'remaining_clips_to_infer':plan['delegated_source_clips']-len(reuse),
            'nodes':{n:{'sources':len(j),'clips':sum(j.values())} for n,j in plan['assigned'].items()},
            'local_nod_not_changed':True,'local_gpu_not_restarted':True,'gpu_models_not_restarted':True}
        write(TX/'committed.json',report);print(json.dumps(report))


if __name__=='__main__':main()
