"""Read-only runtime comparison on node17; no worker or tuning mutations."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,result_path


def process_live(record):
    try:fields=Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,KeyError):return False
    return fields[0]!='Z' and fields[19]==str(record.get('start_ticks'))


def snapshot(work):
    supervisor=read(work/'supervisor.json');workers={}
    for name,child in supervisor['children'].items():
        if not name.startswith(('gpu-','nod-')):continue
        h=read(work/'workers'/f'{name}.json')
        workers[name]={'pid':child['pid'],'start_ticks':child['start_ticks'],
            'live':process_live(child),'heartbeat_matches':h.get('pid')==child['pid'],
            'completed':h.get('completed_clips',0),'failed':h.get('failed_clips',0),
            'revision':h.get('revision'),'stage':h.get('stage'),
            'execution':{k:h.get('execution',{}).get(k) for k in
                ['batch_size','tokens','frames','memory','clip_batch_size','eager',
                 'renderer_workers','prefix_caching','bounded_source_groups']},
            'rolling_inference':h.get('execution',{}).get('rolling_inference',False),
            'rolling_metrics':{k:v for k,v in h.items() if k.startswith('rolling_')},
            'decode_workers':h.get('decode_workers_active'),
            'decode_slots':h.get('decode_slots_active')}
    tuning=read(work/'resource_tuning.json');selected=read(work/'selected.json')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    gpus=[dict(zip(('index','utilization','memory_used_mib'),map(float,row)))
          for row in csv.reader(io.StringIO(raw))]
    cpu=dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
    mem=dict(line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())
    queue=read(work/'queues/vlm/state.json')
    return {'at':time.time(),'workers':workers,'gpus':gpus,
        'vlm_queue_range':{'offset':queue['offset'],'inflight':list(queue['inflight'])},
        'expected_nod_workers':read(work/'config.json')['nod_workers'],
        'guards_live':process_live(supervisor) and process_live(read(work/'watchdog.json')),
        'tuning':{k:tuning[k] for k in ['decode_workers','decode_slots']},
        'nod_decode_slots':selected['decode_slots'],
        'rolling':bool(supervisor.get('rolling_restart')),'warming':supervisor.get('warming_gpus',[]),
        'cpu_usage_usec':int(cpu['usage_usec']),'oom_kill':int(mem['oom_kill']),
        'queue_progress':supervisor['progress']}


def summarize(samples,revision,revision_overrides=None):
    first,last=samples[0],samples[-1];elapsed=last['at']-first['at'];reasons=[]
    expected=first['workers'];gpu_names=[f'gpu-{i}' for i in range(8)]
    revision_overrides=revision_overrides or {}
    if set(revision_overrides)-set(gpu_names):raise ValueError('Revision override names unknown GPU')
    for row in samples:
        if row['rolling'] or row['warming']:reasons.append('worker rollout or warmup')
        if not row['guards_live']:reasons.append('supervision process missing')
        if sum(k.startswith('nod-') for k in row['workers'])!=row['expected_nod_workers']:
            reasons.append('nod worker count mismatch')
        if row['tuning']!=first['tuning'] or row['nod_decode_slots']!=first['nod_decode_slots']:
            reasons.append('runtime tuning changed')
        if row['nod_decode_slots']!=row['tuning']['decode_slots']:reasons.append('branch slot mismatch')
        if set(row['workers'])!=set(expected):reasons.append('worker membership changed')
        for name,old in expected.items():
            current=row['workers'].get(name,{})
            if (current.get('pid'),current.get('start_ticks'))!=(old['pid'],old['start_ticks']):
                reasons.append('worker identity changed')
            if not current.get('live') or not current.get('heartbeat_matches'):
                reasons.append('missing live worker or stale heartbeat')
            if current.get('completed',0)<old['completed']:reasons.append('worker count regressed')
            if current.get('execution')!=old['execution']:reasons.append('model execution changed')
            if current.get('rolling_inference',False)!=old.get('rolling_inference',False):
                reasons.append('rolling inference path changed')
        for name in gpu_names:
            current=row['workers'].get(name,{})
            if current.get('revision')!=revision_overrides.get(name,revision):reasons.append('GPU revision mismatch')
            if current.get('completed',0)<=0:reasons.append('GPU has no initial output')
            if current.get('decode_workers')!=row['tuning']['decode_workers'] or current.get('decode_slots')!=row['tuning']['decode_slots']:
                reasons.append('GPU has not adopted tuning')
        if row['oom_kill']>first['oom_kill']:reasons.append('OOM kill')
    if elapsed<=0:reasons.append('nonpositive interval')
    valid=not reasons
    totals={branch:sum(last['workers'].get(k,{}).get('completed',0)-v['completed']
                      for k,v in expected.items() if k.startswith(prefix))
            for branch,prefix in [('vlm','gpu-'),('nod','nod-')]}
    return {'valid_for_throughput_comparison':valid,'invalid_reasons':sorted(set(reasons)),
        'elapsed_s':elapsed,'tuning':first['tuning'],'new_clips':totals,
        'clips_per_s':{k:v/elapsed for k,v in totals.items()} if valid else None,
        'per_gpu_clips_per_s':{k:(last['workers'].get(k,{}).get('completed',0)-expected[k]['completed'])/elapsed
            for k in gpu_names} if valid else None,
        'expected_gpu_revisions':{k:revision_overrides.get(k,revision) for k in gpu_names},
        'mean_sampled_gpu_utilization':sum(g['utilization'] for r in samples for g in r['gpus'])/
            max(1,sum(len(r['gpus']) for r in samples)),
        'used_cpu_cores':(last['cpu_usage_usec']-first['cpu_usage_usec'])/1e6/elapsed if elapsed>0 else None,
        'scope':'Continuous production with different source clips; fixed worker identities and input settings required. Not matched-sample speedup or labeling accuracy.'}


def workload_summary(work,first,last):
    """Read only sources claimed during this window and initial live claims."""
    start,end=first['vlm_queue_range']['offset'],last['vlm_queue_range']['offset']
    if end<start:raise ValueError('Queue journal offset regressed')
    jobs=set(first['vlm_queue_range']['inflight'])
    with (work/'ready.jsonl').open('rb') as f:
        f.seek(start)
        for line in f.read(end-start).splitlines():jobs.add(json.loads(line)['job_id'])
    variants={};seen=set();out={'source_jobs':len(jobs),'candidate_clips':0,'read_results':0,
        'clips':0,'video_seconds':0.,'requests':0,'valid_requests':0,'truncated_requests':0,
        'prompt_tokens':0,'completion_tokens':0,'source_pixels_sum':0,
        'scope':'Result completion timestamps within [window start, window end); bounded by queue journal range, no result-directory scan.'}
    for job in sorted(jobs):
        for row in read(work/'jobs'/f'{job}.json')['rows']:
            sid=row['sample_id']
            if sid in seen:continue
            seen.add(sid);out['candidate_clips']+=1;path=result_path(work,'vlm',sid)
            if not path.exists():continue
            d=read(path);out['read_results']+=1
            if not first['at']<=d['completed_at']<last['at']:continue
            if d['sample_id']!=sid or d['source']!=row['source']:
                raise ValueError(f'Result provenance mismatch: {sid}')
            out['clips']+=1;out['video_seconds']+=d['duration_s']
            width,height=d['source']['video_transform']['source_size'];out['source_pixels_sum']+=width*height
            for record in d['records']:
                out['requests']+=1;out['valid_requests']+=record['status']=='valid'
                out['truncated_requests']+=record['finish_reason']=='length'
                out['prompt_tokens']+=record['prompt_tokens'];out['completion_tokens']+=record['completion_tokens']
            variant='rolling_groups' if d.get('execution',{}).get('rolling_inference',False) else 'bounded_batches'
            counts=variants.setdefault(variant,{'clips':0,'video_seconds':0.,'requests':0,
                'valid_requests':0,'truncated_requests':0,'prompt_tokens':0,'completion_tokens':0,'source_pixels_sum':0})
            counts['clips']+=1;counts['video_seconds']+=d['duration_s'];counts['source_pixels_sum']+=width*height
            for record in d['records']:
                counts['requests']+=1;counts['valid_requests']+=record['status']=='valid'
                counts['truncated_requests']+=record['finish_reason']=='length'
                counts['prompt_tokens']+=record['prompt_tokens'];counts['completion_tokens']+=record['completion_tokens']
    elapsed=last['at']-first['at'];out['completed_video_seconds_per_s']=out['video_seconds']/elapsed
    out['mean_clip_seconds']=out['video_seconds']/out['clips'] if out['clips'] else None
    out['mean_original_frame_pixels']=out['source_pixels_sum']/out['clips'] if out['clips'] else None
    out['by_execution_variant']=variants
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--tag',required=True);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--revision',type=int,default=7)
    p.add_argument('--worker-revision',action='append',default=[],metavar='GPU_NAME=REVISION')
    a=p.parse_args();overrides={}
    for entry in a.worker_revision:
        try:name,number=entry.split('=');number=int(number)
        except ValueError:p.error('Expected --worker-revision gpu-7=8')
        if name not in {f'gpu-{i}' for i in range(8)} or name in overrides or number<1:
            p.error('Invalid or duplicate worker revision override')
        overrides[name]=number
    if Path(a.tag).name!=a.tag or a.tag in ('.','..') or not 30<=a.seconds<=3600:p.error('Invalid tag/duration')
    destination=a.work/'monitor_snapshots'/f'{a.tag}.json'
    if destination.exists():p.error('Report already exists; choose a new tag')
    samples=[snapshot(a.work)];start_check=summarize(samples,a.revision,overrides)
    reasons=[x for x in start_check['invalid_reasons'] if x!='nonpositive interval']
    if reasons:raise SystemExit('Not ready: '+', '.join(reasons))
    deadline=time.monotonic()+a.seconds
    while time.monotonic()<deadline:
        time.sleep(min(30,max(0,deadline-time.monotonic())))
        samples.append(snapshot(a.work));summary=summarize(samples,a.revision,overrides)
        write(destination,{'complete':False,'summary':summary,'samples':samples})
        print(json.dumps({'tag':a.tag,'elapsed':round(summary['elapsed_s']),
              'new_clips':summary['new_clips'],'valid':summary['valid_for_throughput_comparison']}),flush=True)
        if not summary['valid_for_throughput_comparison']:break
    report={'complete':True,'summary':summarize(samples,a.revision,overrides),'samples':samples}
    try:report['workload']=workload_summary(a.work,samples[0],samples[-1])
    except Exception as exc:report['workload_error']=repr(exc)
    write(destination,report)
    print(str(destination),flush=True)


if __name__=='__main__':main()
