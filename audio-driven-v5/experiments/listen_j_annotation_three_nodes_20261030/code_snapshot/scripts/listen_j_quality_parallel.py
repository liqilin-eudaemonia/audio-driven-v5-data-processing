#!/usr/bin/env python3
"""Reduced CPU quality load alongside original annotation; never launches a second annotation queue."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_quality_pipeline import (
    live,identity,spawn,read,write,compact_write,export_node,QUALITY_PYTHON,ENTRY)
from listen_benchmark.j_annotation import JobQueue,acquire,result_path


def annotation_drained(work):
    work=Path(work)
    size=(work/'ready.jsonl').stat().st_size
    return all((s:=read(work/'queues'/branch/'state.json'))['offset']>=size and not s['inflight']
               for branch in ('vlm','nod'))


def annotation_owner_index(scope):
    owners={}
    for node,item in scope['nodes'].items():
        with (Path(item['work'])/'ready.jsonl').open() as f:
            for line in f:
                pair=json.loads(line)['job_id']
                if pair in owners and owners[pair]!=item['work']:
                    raise ValueError('Annotation source assigned to two nodes')
                owners[pair]=item['work']
    return owners


def export_annotation_index(work,scope):
    work=Path(work);owners=annotation_owner_index(scope)
    counts={'retained':0,'both_labels_present':0,'missing_vlm':0,'missing_nod':0,'crop_differs_from_annotation':0}
    temp=work/'retained_annotation_index.jsonl.tmp'
    with temp.open('w') as out,(work/'retained_rows.jsonl').open() as rows:
        for line in rows:
            row=json.loads(line);s=row['source'];owner=Path(owners[s['pair']]);labels={}
            for branch in ('vlm','nod'):
                path=result_path(owner,branch,row['sample_id'])
                labels[branch]=str(path) if path.exists() else None
                counts['missing_'+branch]+=int(labels[branch] is None)
            changed=s['quality_screening']['original_crop']!=s['video_transform']['crop']
            counts['retained']+=1;counts['both_labels_present']+=int(all(labels.values()))
            counts['crop_differs_from_annotation']+=int(changed)
            out.write(json.dumps({'sample_id':row['sample_id'],'row':row,'annotations':labels,
                'annotation_work':str(owner),'annotation_crop':s['quality_screening']['original_crop'],
                'crop_differs_from_annotation':changed,'training_admitted':False,
                'status':'labels_present_pending_semantic_review' if all(labels.values()) else 'annotation_missing'})+'\n')
    temp.replace(work/'retained_annotation_index.jsonl')
    write(work/'parallel_complete.json',{'at':time.time(),'counts':counts,
        'note':'File presence is not semantic annotation accuracy or training admission.'})


def controller(work):
    work=Path(work);owner=acquire(work/'controller.lock',False)
    if owner is None:return
    cfg=read(work/'pipeline_config.json');state=read(work/'controller.json')
    state.update(pid=os.getpid(),start_ticks=identity(os.getpid()),mode='parallel_original_annotation')
    state.pop('controller_ended_at',None);queue=JobQueue(work,'quality');stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    draining=state.setdefault('parallel_draining',{})
    try:
        while not stop:
            now=time.time();state['heartbeat']=now
            control=read(work/'worker_control.json');target=control['workers'];nice=control.get('nice',10)
            state['target_workers']=target
            for key,child in state['children'].items():
                if not live(child):continue
                keep=key.isdigit() and int(key)<target
                if keep:
                    try:os.setpriority(os.PRIO_PROCESS,child['pid'],nice)
                    except ProcessLookupError:pass
                elif key not in draining:
                    try:os.kill(child['pid'],signal.SIGTERM)
                    except ProcessLookupError:pass
                    draining[key]={'requested_at':now,'child':child}
                elif now-draining[key]['requested_at']>180:
                    # Only recorded quality workers; atomic per-clip outputs survive and the
                    # source lock becomes recoverable. Never touch annotation processes here.
                    if os.getpgid(child['pid'])==child['pid']:
                        os.killpg(child['pid'],signal.SIGKILL)
                        draining[key]['forced_after_source_grace']=now
            active=sum(live(c) for c in state['children'].values());state['active_workers']=active
            if not (work/'quality_complete.json').exists():
                q=read(queue.path);remaining=cfg['scope_sources']-q['completed']-len(q['failed'])
                state.update(phase='cpu_quality_parallel_annotation',completed_sources=q['completed'],
                    completed_clips=q.get('completed_clips',0),failed_sources=len(q['failed']),remaining_sources=remaining)
                if remaining==0 and active==0:
                    state['phase']='exporting_quality_filter_only';write(work/'controller.json',state)
                    export_node(work,cfg)
                    write(work/'parallel_export_policy.json',{'annotation_new_queue_must_not_start':True,
                        'original_annotation_continues':True,'at':time.time()})
                else:
                    for slot in range(target):
                        key=str(slot)
                        if not live(state['children'].get(key,{})) and active<min(target,remaining):
                            child=spawn(work,f'quality-{slot}',[QUALITY_PYTHON,str(ENTRY),'worker',
                                '--work',str(work),'--slot',str(slot)])
                            os.setpriority(os.PRIO_PROCESS,child['pid'],nice)
                            state['children'][key]=child;draining.pop(key,None);active+=1
                    state['active_workers']=active
            else:
                state['phase']='quality_complete_waiting_original_annotations'
                scope=read(work.parent/'annotation_nodes.json')
                waiting=[node for node,item in scope['nodes'].items() if not annotation_drained(item['work'])]
                state['waiting_annotation_nodes']=waiting
                if not waiting:
                    export_annotation_index(work,scope)
                    state['phase']='parallel_filter_join_complete';write(work/'controller.json',state);break
            compact_write(work/'controller.json',state)
            with (work/'telemetry.jsonl').open('a') as f:
                f.write(json.dumps({k:v for k,v in state.items() if k not in ('children','parallel_draining')})+'\n')
            time.sleep(10)
    finally:
        state['controller_ended_at']=time.time();write(work/'controller.json',state);owner.close()


def watchdog(work):
    work=Path(work);owner=acquire(work/'parallel_watchdog.lock',False)
    if owner is None:return
    state={'pid':os.getpid(),'start_ticks':identity(os.getpid()),'started_at':time.time()};stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping)
    while not stop and not (work/'STOP_PARALLEL').exists():
        c=read(work/'controller.json')
        if c.get('phase')=='parallel_filter_join_complete':break
        if not live(c):state['controller']=spawn(work,'parallel-controller',
            [QUALITY_PYTHON,str(Path(__file__).resolve()),'controller','--work',str(work)])
        elif c.get('mode')!='parallel_original_annotation':raise RuntimeError('Old quality controller still owns this run')
        state['heartbeat']=time.time();write(work/'parallel_watchdog.json',state);time.sleep(15)
    state['ended_at']=time.time();write(work/'parallel_watchdog.json',state);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('controller','watchdog'))
    p.add_argument('--work',type=Path,required=True);a=p.parse_args();globals()[a.command](a.work.resolve())
