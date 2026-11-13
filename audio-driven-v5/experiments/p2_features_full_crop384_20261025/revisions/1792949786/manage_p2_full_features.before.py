"""Three-node follow-on scheduler, durable watchdogs and 20-minute feedback."""
import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_fusion import read,atomic,lock,file_sha,jsonl_writer
from listen_benchmark.feature_pipeline import build_source_index,publish_shard


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(')')[-1].split()[0]!='Z'
    except FileNotFoundError:return False


def owned(pid,run,script):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
        return str(run) in cmd and script in cmd and os.getpgid(pid)==pid
    except (FileNotFoundError,ProcessLookupError):return False


def launch(run,cfg,role,worker=None,rank=0,gpu=None,node=None,watchdog=False):
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if role=='features':env.update(OMP_NUM_THREADS=str(cfg.get('feature_cpu_threads',4)),MKL_NUM_THREADS=str(cfg.get('feature_cpu_threads',4)))
    if role=='features' and cfg.get('expandable_cuda_segments'):env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    if worker:
        cmd=[cfg['python'],str(ROOT/'scripts/run_p2_full_features.py'),'--run',str(run),'--role',role,'--worker',worker,'--rank',str(rank)]
    else:
        cmd=[cfg['manager_python'],str(Path(__file__).resolve()),'--run',str(run),'--role',role]
        if node:cmd+=['--node',node]
        if watchdog:cmd+=['--watchdog']
    logpath=run/f'logs/{worker or role+("-"+node if node else "")}-{time.time_ns()}.log'
    with logpath.open('a') as log:child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return dict(pid=child.pid,role=role,worker=worker,gpu=gpu,command=cmd,log=str(logpath),launched_at=time.time())


def telemetry(previous=None):
    value=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=20)
    gpus={int(z[0]):dict(memory_used_mib=float(z[1]),memory_total_mib=float(z[2]),gpu_util=float(z[3]),power_w=float(z[4])) for line in value.splitlines() if (z:=line.split(','))}
    ticks=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]));total=sum(ticks);idle=ticks[3]+ticks[4]
    prev=(previous or {}).get('cpu_ticks');cpu=None;iowait=None
    if prev and total>sum(prev):
        dt=total-sum(prev);cpu=100*(1-(idle-prev[3]-prev[4])/dt);iowait=100*(ticks[4]-prev[4])/dt
    memory={k:int(v.split()[0]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line for k,v in [line.split(':',1)]}
    return dict(utc=time.time(),gpus=gpus,cpu_ticks=ticks,cpu_util=cpu,iowait_percent=iowait,load_average=os.getloadavg(),logical_cpus=os.cpu_count(),memory_available_gib=memory['MemAvailable']/2**20)


def ensure_nodes(run,cfg):
    for node,value in cfg['nodes'].items():
        if time.time()-read(run/f'watchdogs/node-{node}.json').get('heartbeat',0)<90:continue
        cmd=[cfg['manager_python'],str(Path(__file__).resolve()),'--run',str(run),'--role','node','--node',node,'--watchdog']
        code='import subprocess; f=open('+repr(str(run/f'logs/bootstrap-{node}.log'))+',"a"); p=subprocess.Popen('+repr(cmd)+',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)'
        try:
            out=subprocess.run(['ssh','-i',cfg['ssh_key'],'-o','BatchMode=yes','-o','ConnectTimeout=8','root@'+value['host'],shlex.join([cfg['manager_python'],'-c',code])],text=True,capture_output=True,timeout=20)
            if out.returncode:raise RuntimeError(out.stderr[-1000:])
            atomic(run/f'watchdogs/bootstrap-{node}.json',dict(pid=out.stdout.strip(),utc=time.time()))
            alert=run/f'alerts/bootstrap-{node}.json'
            if alert.exists():atomic(alert,dict(read(alert),resolved_at=time.time()))
        except Exception as e:atomic(run/f'alerts/bootstrap-{node}.json',dict(error=repr(e),utc=time.time()))


def node_loop(run,node):
    cfg=read(run/'config.json');folder=run/'nodes'/node;folder.mkdir(parents=True,exist_ok=True);children={};retry={};terminating={};previous=None
    with lock(folder/'supervisor.lock') as acquired:
        if not acquired:raise RuntimeError('Node feature supervisor already active')
        for p in folder.glob('child-*.json'):
            record=read(p)
            if alive(record['pid']):
                if not owned(record['pid'],run,'run_p2_full_features.py'):raise RuntimeError('Foreign process at saved PID')
                children[record['worker']]=record
        while not (run/'STOP').exists() and not (run/'COMPLETE.json').exists():
            now=time.time();sample=telemetry(previous);previous=sample;control=read(run/f'controls/{node}.json');queues=read(run/'queues.json');states={};waiting={}
            allocated={j['gpu']:j for j in cfg['nodes'][node]['jobs']}
            sample['gpus']={g:v for g,v in sample['gpus'].items() if g in allocated}
            for worker,record in list(children.items()):
                pid=record['pid'];state=read(run/f'workers/{worker}.json');states[worker]=state
                if not alive(pid):
                    try:os.waitpid(pid,os.WNOHANG)
                    except ChildProcessError:pass
                    children.pop(worker);terminating.pop(worker,None)
                    retry[worker]=now+(60 if state.get('status') in ('stopped','finished') else 120)
                    continue
                last=max(record['launched_at'],state.get('heartbeat',0) if state.get('pid')==pid else 0)
                if (now-last>cfg['worker_stall_seconds'] or worker in terminating) and owned(pid,run,'run_p2_full_features.py'):
                    if worker not in terminating:os.killpg(pid,signal.SIGTERM);terminating[worker]=now
                    elif now-terminating[worker]>60:os.killpg(pid,signal.SIGKILL)
                    atomic(folder/f'alert-{worker}.json',dict(worker=worker,error='stale worker; restarting',utc=now))
                elif state.get('pid')==pid and state.get('status') in ('media','features','job_complete'):
                    alert=folder/f'alert-{worker}.json'
                    if alert.exists() and not read(alert).get('resolved_at'):atomic(alert,dict(read(alert),resolved_at=now))
            # A bounded staging buffer prevents 24 CPUs from rendering the entire corpus ahead of GPUs.
            media_target=min(cfg['media_workers_max'],control.get('media_workers',cfg['media_workers_initial']))
            if not (run/'source_index/COMPLETE.json').exists() or len(queues.get('features',[]))>=cfg['media_ready_backlog']:media_target=0
            for i in range(cfg['media_workers_max']):
                worker=f'{node}-media-{i}';retire=run/f'controls/retire-{worker}'
                if i>=media_target:
                    if worker in children:retire.touch()
                    continue
                if retire.exists():retire.unlink()
                if worker in children or now<retry.get(worker,0) or not queues.get('media'):continue
                record=launch(run,cfg,'media',worker,rank=i*103+cfg['nodes'][node]['jobs'][0]['rank']);children[worker]=record;atomic(folder/f'child-{worker}.json',record)
            # Full fusion is the explicit upstream barrier; no training-data shortcut.
            upstream_complete=(Path(cfg['fusion_run'])/'COMPLETE.json').exists()
            canary_state=read(run/'CANARY.json')
            for gpu,info in sample['gpus'].items():
                members=[x for x in children.values() if x.get('gpu')==gpu]
                if not upstream_complete:waiting[gpu]='waiting_for_full_fusion';continue
                if not canary_state.get('passed'):
                    if not members and info['memory_used_mib']<256:
                        with lock(run/'locks/canary-launch.lock') as got:
                            current=read(run/'canary_owner.json')
                            owner_state=read(run/f'workers/{current.get("worker", "none")}.json')
                            fresh=time.time()-owner_state.get('heartbeat',current.get('launched_at',0))<cfg['worker_stall_seconds']+180
                            if owner_state.get('status')=='failed' and time.time()-owner_state.get('heartbeat',0)>120:fresh=False
                            if got and (not current or not fresh):
                                worker=f'{node}-g{gpu}-canary';record=launch(run,cfg,'canary',worker,gpu=gpu);children[worker]=record;atomic(folder/f'child-{worker}.json',record);atomic(run/'canary_owner.json',record)
                    waiting[gpu]='waiting_for_native_parity_canary';continue
                feature_members=[x for x in members if x['role']=='features']
                estimate=max(8192,canary_state['peak_reserved_mib']*cfg.get('resident_peak_safety_factor',1.20))
                capacity=max(1,int((info['memory_total_mib']-cfg['gpu_headroom_mib'])//estimate))
                if cfg.get('vae_workspace_slots') and read(run/'capacity_preset.json').get('validated'):
                    calibration=read(run/'capacity_preset.json');capacity=calibration['max_validated_replicas']
                    estimate=calibration['resident_allocated_mib']*1.3+8192
                desired=min(capacity,cfg['feature_workers_max'],control.get('gpu_workers',{}).get(str(gpu),cfg['feature_workers_initial']))
                if any(read(run/f'alerts/oom-{x["worker"]}.json').get('utc',0)>now-2400 for x in feature_members):desired=1
                # OOM survives process exit and forces conservative restart capacity.
                for p in (run/'alerts').glob(f'oom-{node}-g{gpu}-*.json'):
                    if now-read(p).get('utc',0)<2400:desired=1
                if members and not feature_members:waiting[gpu]='canary_process_releasing';continue
                for i in range(cfg['feature_workers_max']):
                    worker=f'{node}-g{gpu}-f{i}';retire=run/f'controls/retire-{worker}'
                    if i>=desired:
                        if worker in children:retire.touch()
                        continue
                    if retire.exists():retire.unlink()
                    if worker in children or now<retry.get(worker,0) or not queues.get('features'):continue
                    if not feature_members and info['memory_used_mib']>=256:waiting[gpu]='waiting_for_upstream_GPU_release';break
                    if feature_members and info['memory_total_mib']-info['memory_used_mib']<estimate+cfg['gpu_headroom_mib']:waiting[gpu]='memory_headroom_guard';break
                    record=launch(run,cfg,'features',worker,rank=allocated[gpu]['rank']+i*61,gpu=gpu);children[worker]=record;feature_members.append(record);atomic(folder/f'child-{worker}.json',record)
                    # Permit at most one new model replica per GPU per polling cycle.
                    waiting[gpu]='feature_worker_launched';break
            value=dict(node=node,host=socket.gethostname(),pid=os.getpid(),heartbeat=now,telemetry=sample,children=children,workers=states,waiting=waiting,control=control,upstream_complete=upstream_complete)
            atomic(folder/'status.json',value)
            with (folder/'telemetry.jsonl').open('a') as f:f.write(json.dumps(dict(sample,active_feature_workers=sum(x['role']=='features' for x in children.values()),active_media_workers=sum(x['role']=='media' for x in children.values())))+'\n')
            time.sleep(cfg['poll_seconds'])


def audit_job(run,jid):
    """Validate payload hashes once in parallel while other GPUs continue computing."""
    run=Path(run);marker=run/f'state/final/{jid}.json'
    with lock(run/f'locks/audit-{jid}.lock') as acquired:
        if not acquired:return
        if marker.exists():return read(marker)
        rows=read(run/f'manifests/shards/{jid}.json');total=0
        from p2_listen.contracts import digest
        for row in rows:
            for part in ['input','teacher','vae','events']:
                folder=run/f'cache/{part}/{row["sample_id"]}';m=read(folder/'complete.json')
                if not m or digest(m['contract'])!=m['identity'] or file_sha(folder/'payload.pt')!=m['payload_sha256']:raise ValueError('Feature cache audit failed')
                if m['contract']['input']['sample_id']!=row['sample_id']:raise ValueError('Feature cache sample mismatch')
                total+=(folder/'payload.pt').stat().st_size
        value=dict(done=True,job_id=jid,samples=len(rows),payload_bytes=total,manifest_sha256=file_sha(run/f'manifests/shards/{jid}.json'),utc=time.time());atomic(marker,value);return value


def final_index(run,cfg,final):
    seen=set();count=0
    paths=[run/'manifests/observations.json',run/'manifests/targets.json']
    temps=[p.with_suffix('.pending.json') for p in paths];streams=[p.open('w') for p in temps]
    try:
        for f in streams:f.write('[\n')
        for jid in sorted(final):
            path=run/f'manifests/shards/{jid}.json'
            if file_sha(path)!=final[jid]['manifest_sha256']:raise ValueError('Final manifest changed')
            for row in read(path):
                sid=row['sample_id']
                if sid in seen:raise ValueError('Duplicate final sample')
                seen.add(sid)
                for f,key in zip(streams,['observation','target']):
                    if count:f.write(',\n')
                    f.write(json.dumps(row[key],ensure_ascii=False))
                count+=1
            atomic(run/'final_audit.json',dict(heartbeat=time.time(),job=jid,samples=count))
        if count!=cfg['expected_samples']:raise ValueError('Final feature count differs')
        for f in streams:f.write('\n]\n');f.flush();os.fsync(f.fileno());f.close()
        for tmp,path in zip(temps,paths):os.replace(tmp,path)
    finally:
        for f in streams:
            if not f.closed:f.close()
    atomic(run/'index.json',dict(schema='p2-full-features-index-v1',samples=count,jobs=final,parts=['input','teacher','vae','events','text'],training_started=False,formal_training_admitted=False,manifests={str(p.relative_to(run)):file_sha(p) for p in paths}))
    atomic(run/'COMPLETE.json',dict(utc=time.time(),samples=count,index_sha256=file_sha(run/'index.json'),training_started=False))


def controller(run):
    cfg=read(run/'config.json');fusion=Path(cfg['fusion_run']);last=0;pending={};retry={};index_pid=None;monitor_pid=None
    with lock(run/'controller.lock') as acquired:
        if not acquired:raise RuntimeError('Feature controller already active')
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
            while not (run/'STOP').exists() and not (run/'COMPLETE.json').exists():
                now=time.time()
                if now-last>60:ensure_nodes(run,cfg);last=now
                for role in ['index','monitor']:
                    if role=='index' and (run/'source_index/COMPLETE.json').exists():continue
                    record=read(run/f'watchdogs/{role}-launch.json')
                    if not record or not alive(record['pid']):
                        rec=launch(run,cfg,role,watchdog=role=='monitor');atomic(run/f'watchdogs/{role}-launch.json',rec)
                for future,key in list(pending.items()):
                    if not future.done():continue
                    pending.pop(future);stage,jid=key
                    try:
                        future.result();alert=run/f'alerts/{stage}-{jid}.json'
                        if alert.exists():atomic(alert,dict(read(alert),resolved_at=now))
                    except Exception as e:
                        retry[key]=now+120;atomic(run/f'alerts/{stage}-{jid}.json',dict(error=repr(e),traceback=traceback.format_exc()[-4000:],utc=now))
                published={p.stem:read(p) for p in (run/'state/published').glob('*.json')};media={p.stem:read(p) for p in (run/'state/media').glob('*.json')};features={p.stem:read(p) for p in (run/'state/features').glob('*.json')};final={p.stem:read(p) for p in (run/'state/final').glob('*.json')}
                jobs=sorted(j for x in published.values() for j in x['jobs']);ready_media=[j for j in jobs if j not in media];ready_features=[j for j in media if j not in features]
                atomic(run/'queues.json',dict(media=ready_media[:4096],features=sorted(ready_features)[:4096],updated_at=now))
                if (run/'source_index/COMPLETE.json').exists():
                    available=[p.stem for p in (fusion/'state/fusion').glob('*.json') if p.stem not in published]
                    todo=[('audit',j) for j in features if j not in final]+[('publish',s) for s in sorted(available)]
                    for stage,jid in todo:
                        if len(pending)>=8:break
                        if (stage,jid) in pending.values() or now<retry.get((stage,jid),0):continue
                        future=pool.submit(audit_job if stage=='audit' else publish_shard,str(run),jid);pending[future]=(stage,jid)
                counts={name:sum(x['samples'] for x in mapping.values()) for name,mapping in [('published',published),('media',media),('features',features),('verified',final)]}
                by_node={node:sum(x['samples'] for x in features.values() if x['worker'].startswith(node+'-')) for node in cfg['nodes']}
                summary=dict(run_id=run.name,heartbeat=now,pid=os.getpid(),counts=counts,features_by_node=by_node,expected_samples=cfg['expected_samples'],jobs=dict(published=len(jobs),media=len(media),features=len(features),verified=len(final)),upstream_complete=(fusion/'COMPLETE.json').exists(),source_index_complete=(run/'source_index/COMPLETE.json').exists(),canary=read(run/'CANARY.json'),nodes={p.parent.name:read(p) for p in (run/'nodes').glob('*/status.json')},alerts=[read(p) for p in (run/'alerts').glob('*.json') if not read(p).get('resolved_at')],training_started=False)
                atomic(run/'status.json',summary)
                with (run/'history.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now,counts=counts,jobs=summary['jobs']))+'\n')
                preview=ROOT/'data_previews/listen_benchmark_review/full_features';preview.mkdir(parents=True,exist_ok=True);atomic(preview/'status.json',summary)
                if summary['upstream_complete'] and counts['published']==cfg['expected_samples'] and counts['verified']==cfg['expected_samples'] and (run/'cache/text/complete.json').exists():
                    m=read(run/'cache/text/complete.json')
                    if file_sha(run/'cache/text/payload.pt')!=m['payload_sha256']:raise ValueError('Shared text cache corrupt')
                    final_index(run,cfg,final);return
                time.sleep(cfg['poll_seconds'])


def raw_finished(cfg):
    branches=read(Path(cfg['raw_run'])/'status.json').get('branches',{})
    return all(branches.get(b,{}).get('total_shards',0)>0 and
               branches[b].get('shards_done')==branches[b]['total_shards']
               for b in ('visual','audio','asr'))


def monitor_once(run,notify=True):
    cfg=read(run/'config.json');now=time.time();previous=read(run/'monitor/latest.json');nodes={};actions=[]
    if cfg.get('monitor_start_after_raw',False) and not raw_finished(cfg):
        return dict(status='waiting_for_raw_completion',report_sent=False,utc=now)
    status=read(run/'status.json');queues=read(run/'queues.json');upstream_done=(Path(cfg['fusion_run'])/'COMPLETE.json').exists()
    for node in cfg['nodes']:
        path=run/f'nodes/{node}/telemetry.jsonl';samples=[]
        if path.exists():
            for line in path.open():
                x=json.loads(line)
                if x['utc']>=now-cfg['monitor_seconds']:samples.append(x)
        if not samples:nodes[node]=dict(error='no telemetry');continue
        avg=lambda values:sum(values)/len(values) if values else None
        gpu_ids=[j['gpu'] for j in cfg['nodes'][node]['jobs']]
        gpus={str(g):dict(util=avg([x['gpus'][str(g)]['gpu_util'] for x in samples if str(g) in x['gpus']]),memory_mib=avg([x['gpus'][str(g)]['memory_used_mib'] for x in samples if str(g) in x['gpus']]),last_memory_mib=samples[-1]['gpus'][str(g)]['memory_used_mib']) for g in gpu_ids}
        cpu=avg([x['cpu_util'] for x in samples if x['cpu_util'] is not None]);io=avg([x['iowait_percent'] for x in samples if x['iowait_percent'] is not None])
        node_done=status.get('features_by_node',{}).get(node,0);past=previous.get('nodes',{}).get(node,{});dt=now-previous.get('utc',now)
        node_rate=(node_done-past.get('feature_samples',node_done))/dt if dt>0 else 0
        nodes[node]=dict(gpus=gpus,cpu_util=cpu,iowait_percent=io,samples=len(samples),last_utc=samples[-1]['utc'],feature_workers=samples[-1]['active_feature_workers'],media_workers=samples[-1]['active_media_workers'],feature_samples=node_done,feature_samples_per_second=node_rate)
        ctl=read(run/f'controls/{node}.json') or dict(media_workers=cfg['media_workers_initial'],gpu_workers={str(g):cfg['feature_workers_initial'] for g in gpu_ids})
        changed=False;rolled_back=False
        if ctl.get('trial_previous') and node_rate>0 and node_rate<.9*ctl.get('trial_baseline_rate',0) and len(queues.get('features',[]))>=48:
            ctl=ctl['trial_previous'];changed=True;rolled_back=True;actions.append(dict(node=node,action='rollback_concurrency_trial',reason='Throughput fell more than 10% with ready work'))
        before=json.loads(json.dumps({k:v for k,v in ctl.items() if not k.startswith('trial_')}))
        if upstream_done and status.get('canary',{}).get('passed') and len(samples)>=30 and not rolled_back:
            if len(queues.get('features',[]))<48 and queues.get('media') and (cpu or 0)<75 and (io or 0)<15 and ctl['media_workers']<cfg['media_workers_max']:
                ctl['media_workers']=min(cfg['media_workers_max'],ctl['media_workers']+4);changed=True;actions.append(dict(node=node,action='increase_media_workers',value=ctl['media_workers'],reason='GPU queue shallow and CPU available'))
            if len(queues.get('features',[]))>=48 and (cpu or 0)<85 and (io or 0)<20:
                for gpu,x in gpus.items():
                    limit=ctl['gpu_workers'].get(gpu,cfg['feature_workers_initial'])
                    if x['util']<80 and limit<cfg['feature_workers_max']:
                        ctl['gpu_workers'][gpu]=limit+1;changed=True;actions.append(dict(node=node,gpu=gpu,action='increase_feature_replicas',value=limit+1,reason='Ready work and low GPU utilization; node memory guard still applies'))
        if changed:
            if not rolled_back:ctl.update(trial_previous=before,trial_baseline_rate=node_rate)
            ctl['updated_at']=now;atomic(run/f'controls/{node}.json',ctl)
    elapsed=now-previous.get('utc',now);done=status.get('counts',{}).get('features',0)
    rate=(done-previous.get('feature_samples',done))/elapsed if elapsed>0 else 0
    report=dict(utc=now,interval_seconds=cfg['monitor_seconds'],nodes=nodes,feature_samples=done,feature_samples_per_second=rate,counts=status.get('counts',{}),actions=actions,upstream_complete=upstream_done,alerts=status.get('alerts',[]))
    path=run/f'monitor/report-{int(now)}.json';atomic(path,report);atomic(run/'monitor/latest.json',report)
    if notify:
        values=[g['util'] for v in nodes.values() for g in v.get('gpus',{}).values()];mean=sum(values)/len(values) if values else 0
        gpu_count=sum(len(n['jobs']) for n in cfg['nodes'].values())
        message=f'【用户授权的20分钟资源巡检】{len(cfg["nodes"])}机{gpu_count}卡平均GPU利用率 {mean:.1f}%，特征完成 {done}/{cfg["expected_samples"]}，本窗口 {rate:.3f} 段/秒，自动调整 {len(actions)} 项。报告：{path}。请读取报告和当前打标/整合/特征状态；若资源持续未充分利用，按用户要求在不改变数据语义、不停止其他任务的前提下检查并优化本流水线。不要创建新任务或子代理。'
        try:
            result=subprocess.run([cfg['codex_binary'],'queue','--thread',cfg['thread_id'],'--message',message],text=True,capture_output=True,timeout=30,cwd=ROOT)
            delivery=dict(utc=time.time(),returncode=result.returncode,stdout=result.stdout[-2000:],stderr=result.stderr[-2000:],report=str(path))
        except Exception as exc:delivery=dict(utc=time.time(),returncode=-1,error=repr(exc),report=str(path))
        atomic(run/'monitor/last_notification.json',delivery)
        if delivery['returncode']:raise RuntimeError('Could not deliver periodic report to current Codex thread')
    return report


def monitor_loop(run):
    cfg=read(run/'config.json')
    while not (run/'STOP').exists():
        if cfg.get('monitor_start_after_raw',False):
            if not raw_finished(cfg):
                atomic(run/'monitor/heartbeat.json',dict(heartbeat=time.time(),pid=os.getpid(),status='waiting_for_raw_completion',next_report_at=None))
                time.sleep(10);continue
            if not (run/'monitor/ENABLED.json').exists():atomic(run/'monitor/ENABLED.json',dict(utc=time.time(),trigger='all_raw_branches_finished'))
        enabled=read(run/'monitor/ENABLED.json').get('utc',0)
        last=max(read(run/'monitor/latest.json').get('utc',0),enabled)
        notification=read(run/'monitor/last_notification.json')
        retry_delivery=notification.get('returncode',0)!=0 and time.time()-notification.get('utc',0)>60
        if time.time()-last>=cfg['monitor_seconds'] or retry_delivery or (run/'COMPLETE.json').exists():
            monitor_once(run)
            if (run/'COMPLETE.json').exists():atomic(run/'monitor/FINISHED.json',dict(utc=time.time()));return
        atomic(run/'monitor/heartbeat.json',dict(heartbeat=time.time(),pid=os.getpid(),next_report_at=last+cfg['monitor_seconds']))
        time.sleep(10)


def watchdog(run,role,node):
    cfg=read(run/'config.json');identity=role+('-'+node if node else '');marker=run/f'watchdogs/{identity}.json';child=None
    with lock(run/f'watchdogs/{identity}.lock') as acquired:
        if not acquired:raise RuntimeError('Watchdog already running')
        record=read(marker);pid=record.get('child_pid');launched=time.time();term=None
        if pid and alive(pid) and not owned(pid,run,'manage_p2_full_features.py'):raise RuntimeError('Foreign manager PID')
        while not (run/'STOP').exists() and not (run/('monitor/FINISHED.json' if role=='monitor' else 'COMPLETE.json')).exists():
            state=read(run/'status.json' if role=='controller' else run/'monitor/heartbeat.json' if role=='monitor' else run/f'nodes/{node}/status.json')
            if role=='controller':state['heartbeat']=max(state.get('heartbeat',0),read(run/'final_audit.json').get('heartbeat',0))
            if pid and alive(pid) and time.time()-launched>180 and time.time()-state.get('heartbeat',time.time())>1800 and owned(pid,run,'manage_p2_full_features.py'):
                if term is None:os.killpg(pid,signal.SIGTERM);term=time.time()
                elif time.time()-term>60:os.killpg(pid,signal.SIGKILL)
            if not pid or not alive(pid):
                if child is not None:
                    try:os.waitpid(pid,os.WNOHANG)
                    except ChildProcessError:pass
                record=launch(run,cfg,role,node=node);pid=record['pid'];launched=time.time();term=None;child=True
            atomic(marker,dict(pid=os.getpid(),child_pid=pid,heartbeat=time.time(),role=role,node=node))
            time.sleep(10)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--role',choices=['controller','node','monitor','index','poll-once'],required=True);p.add_argument('--node',default='');p.add_argument('--watchdog',action='store_true');p.add_argument('--no-notify',action='store_true');a=p.parse_args();run=a.run.resolve()
    if a.watchdog:watchdog(run,a.role,a.node)
    elif a.role=='controller':controller(run)
    elif a.role=='node':node_loop(run,a.node)
    elif a.role=='index':build_source_index(run)
    elif a.role=='monitor':monitor_loop(run)
    else:print(json.dumps(monitor_once(run,not a.no_notify),ensure_ascii=False))


if __name__=='__main__':main()
