"""Source-locked visual quality queue and fail-closed Omni admission gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import sys
import time
from listen_benchmark.j_annotation import read,write,acquire,locked,JobQueue
from listen_visual_quality.io import canonical_hash,file_hash
from listen_visual_quality.decision import decide,evidence_index

ROOT=Path(__file__).resolve().parents[2]
POLICY=ROOT/'configs/listen_visual_quality_v1.json'
EVIDENCE=ROOT/'configs/listen_visual_quality_reviewed_evidence_20261030.json'
MODEL=ROOT/'related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task'


def measurement_path(work,sid):
    return Path(work)/'results'/canonical_hash(sid)[:2]/(sid+'.json')


class QualityGate:
    def __init__(self,config):
        self.work=Path(config['work']);self.contract=config['contract_sha256']
        if read(self.work/'config.json')['contract_sha256']!=self.contract:raise ValueError('Visual gate contract changed')
    def ready(self,jid):return (self.work/'measure/completed'/f'{jid}.json').exists()
    def decisions(self,job):
        r=read(self.work/'measure/completed'/f"{job['job_id']}.json")
        if r['contract_sha256']!=self.contract or r['source_result_sha256']!=job['source_result_sha256']:raise ValueError('Visual quality source/policy mismatch')
        out={x['sample_id']:x for x in r['decisions']}
        if len(out)!=len(r['decisions']) or set(out)!={x['sample_id'] for x in job['rows']}:raise ValueError('Incomplete visual quality receipt')
        for row in job['rows']:
            d=out[row['sample_id']]
            if d['source_sha256']!=canonical_hash(row['source']) or d['decision'] not in ('retain','exclude','review'):raise ValueError('Visual quality row mismatch')
        return out


def make_contract():
    from listen_visual_quality.cli import load_policy
    policy=load_policy(POLICY)
    files=[*sorted((ROOT/'src/listen_visual_quality').glob('*.py')),Path(__file__),
           ROOT/'related/visual_reaction_annotation/nod_headshake/face_detection/detect_face_multithread.py',POLICY,EVIDENCE,MODEL]
    return dict(schema='j-visual-gate-v1',policy=policy,files={str(p):file_hash(p) for p in files},
                packages={n:importlib.metadata.version(n) for n in ('av','numpy','mediapipe','opencv-contrib-python')},
                apply_crop_proposals=False,retain_only=True,model=str(MODEL))


def bootstrap(work,omni):
    work=Path(work).resolve();omni=Path(omni).resolve();work.mkdir(parents=True,exist_ok=True)
    with locked(work/'bootstrap.lock'):
        contract=make_contract();digest=canonical_hash(contract)
        if (work/'config.json').exists():
            cfg=read(work/'config.json')
            if cfg['contract_sha256']!=digest or cfg['omni_work']!=str(omni):raise ValueError('Incompatible visual queue')
            return cfg
        with locked(omni/'queues/caption/state.lock'):
            q=read(omni/'queues/caption/state.json');offset=q['offset'];seeds=list(dict.fromkeys([*q['inflight'],*q.get('deferred',{})]))
        if not (work/'jobs').exists():(work/'jobs').symlink_to(omni/'jobs',target_is_directory=True)
        cfg=dict(omni_work=str(omni),contract=contract,contract_sha256=digest,priority_start_offset=offset,created_at=time.time())
        # No production source or original result is modified.
        with (work/'ready.jsonl').open('w') as f:
            for jid in seeds:f.write(json.dumps(dict(job_id=jid))+'\n')
        write(work/'feed.json',dict(offset=offset,backfill_offset=0,published=len(seeds),foreground_eof=False))
        write(work/'config.json',cfg);return cfg


def feed(work,limit=128):
    work=Path(work);cfg=read(work/'config.json');omni=Path(cfg['omni_work'])
    with locked(work/'feed.lock'):
        s=read(work/'feed.json');q=read(work/'queues/measure/state.json')
        pending=s['published']-q['completed']
        if pending>1024:return
        entries=[]
        with (omni/'ready.jsonl').open() as f:
            f.seek(s['offset'])
            for _ in range(limit):
                line=f.readline()
                if not line or not line.endswith('\n'):break
                entries.append(json.loads(line));s['offset']=f.tell()
            s['foreground_eof']=s['offset']>=os.fstat(f.fileno()).st_size
        # Backfill already annotated sources only when foreground supply has headroom.
        if not entries and pending<64 and s['backfill_offset']<cfg['priority_start_offset']:
            with (omni/'ready.jsonl').open() as f:
                f.seek(s['backfill_offset'])
                for _ in range(min(16,limit)):
                    if f.tell()>=cfg['priority_start_offset']:break
                    line=f.readline()
                    if not line or not line.endswith('\n'):break
                    s['backfill_offset']=f.tell();item=json.loads(line)
                    if (omni/'caption/completed'/f"{item['job_id']}.json").exists():entries.append(item)
        with (work/'ready.jsonl').open('a') as f:
            for item in entries:f.write(json.dumps(item)+'\n')
            f.flush();os.fsync(f.fileno())
        s['published']+=len(entries);s['at']=time.time();write(work/'feed.json',s)


def measure_job(work,jid):
    from listen_visual_quality.measurement import measure_row
    work=Path(work);cfg=read(work/'config.json');policy=cfg['contract']['policy'];evidence=evidence_index(read(EVIDENCE))
    job=read(work/'jobs'/f'{jid}.json');src=job['rows'][0]['source']
    if file_hash(src['source_result'])!=job['source_result_sha256']:raise ValueError('J source changed')
    decisions=[];counts=dict(retain=0,exclude=0,review=0);execution_errors=0
    for row in job['rows']:
        sid=row['sample_id'];p=measurement_path(work,sid);source_hash=canonical_hash(row['source'])
        record=read(p) if p.exists() else None
        if record is not None and (record.get('contract_sha256')!=cfg['contract_sha256'] or record.get('source_sha256')!=source_hash):raise ValueError('Measurement cache source/contract mismatch')
        if record is None or record['status']!='ok':
            attempts=[]
            for attempt in range(3):
                record=measure_row(row,policy,str(ROOT),str(MODEL))
                attempts.append(dict(status=record['status'],error=record.get('error'),elapsed_s=record['elapsed_s']))
                if record['status']=='ok':break
            record.update(contract_sha256=cfg['contract_sha256'],source_sha256=source_hash,attempts=attempts,completed_at=time.time())
            write(p,record)
        decision=decide(record,policy,evidence);decision.update(measurement_file=str(p),measurement_sha256=file_hash(p),contract_sha256=cfg['contract_sha256'])
        decisions.append(decision);counts[decision['decision']]+=1;execution_errors+=record['status']!='ok'
    return dict(job_id=jid,clips=len(job['rows']),source_result_sha256=job['source_result_sha256'],contract_sha256=cfg['contract_sha256'],decisions=decisions,counts=counts,execution_errors=execution_errors,completed_at=time.time())


def initialize_cpu():
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')


def run(work,workers):
    sys.path.insert(0,str(ROOT/'scripts'))
    from listen_j_annotation_supervisor import identity
    work=Path(work).resolve();owner=acquire(work/'controller.lock',False)
    if owner is None:raise RuntimeError('Visual quality controller already running')
    cfg=read(work/'config.json')
    assert all(file_hash(p)==h for p,h in cfg['contract']['files'].items()),'Visual quality runtime changed'
    stop=[False]
    def stopping(*_):stop[0]=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    q=JobQueue(work,'measure');pending={};started=time.time();totals=dict(retain=0,exclude=0,review=0);errors=0
    state=dict(pid=os.getpid(),start_ticks=identity(os.getpid()),hostname=socket.gethostname(),workers=workers,started_at=started)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_cpu) as pool:
        while not stop[0] or pending:
            if not stop[0]:
                feed(work)
                while len(pending)<workers:
                    claim=q.claim('visual-quality-'+socket.gethostname())
                    if claim is None:break
                    jid,handle=claim;pending[pool.submit(measure_job,str(work),jid)]=(jid,handle)
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED) if pending else ([],[])
            for future in done:
                jid,handle=pending.pop(future)
                try:
                    receipt=future.result();q.finish(jid,receipt=receipt)
                    for k in totals:totals[k]+=receipt['counts'][k]
                    errors+=receipt['execution_errors']
                except Exception as exc:
                    q.finish(jid,error=repr(exc));state['last_error']=repr(exc)
                finally:handle.close()
            qs=read(q.path);state.update(heartbeat=time.time(),stage='draining' if stop[0] else 'running',active_sources=len(pending),completed_sources=qs['completed'],completed_clips=qs['completed_clips'],failed_sources=qs['failed'],decisions_this_process=totals,execution_errors_this_process=errors)
            write(work/'controller.json',state)
            if not pending:time.sleep(1)
    state.update(stage='stopped',heartbeat=time.time());write(work/'controller.json',state);owner.close()


def watchdog(work,workers):
    sys.path.insert(0,str(ROOT/'scripts'))
    from listen_j_annotation_supervisor import alive,spawn,identity
    work=Path(work).resolve();owner=acquire(work/'watchdog.lock',False)
    if owner is None:raise RuntimeError('Quality watchdog already running')
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    state=dict(pid=os.getpid(),start_ticks=identity(os.getpid()),started_at=time.time(),restarts=[])
    while not stop[0] and not (work/'STOP').exists():
        old=read(work/'controller.json') if (work/'controller.json').exists() else None
        if not alive(old):
            recent=[t for t in state['restarts'] if time.time()-t<3600]
            if len(recent)>=4:state['error']='four controller exits in one hour; inspect logs'
            else:
                c=spawn(work,'controller',[sys.executable,str(ROOT/'scripts/listen_j_visual_gate.py'),'run','--work',str(work),'--workers',str(workers)],dict(os.environ))
                state['child']=c;state['restarts']=recent+[time.time()];state['error']=None
                # Cover the short startup gap before controller.json is published.
                for _ in range(20):
                    if (work/'controller.json').exists() and read(work/'controller.json').get('pid')==c['pid']:break
                    if not alive(c):break
                    time.sleep(.5)
        state['heartbeat']=time.time();write(work/'watchdog.json',state);time.sleep(10)
    owner.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['bootstrap','run','watchdog']);p.add_argument('--work',type=Path,required=True);p.add_argument('--omni',type=Path);p.add_argument('--workers',type=int,default=48);a=p.parse_args()
    if a.command=='bootstrap':print(json.dumps(bootstrap(a.work,a.omni)))
    else:
        if not 1<=a.workers<=96:raise ValueError('Invalid worker count')
        (watchdog if a.command=='watchdog' else run)(a.work,a.workers)
