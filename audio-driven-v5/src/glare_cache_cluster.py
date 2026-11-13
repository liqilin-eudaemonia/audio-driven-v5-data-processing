"""Execution-only GLARE cache routing. Standard library; importing never uses CUDA.

Frozen input indices and numerical code stay untouched. A task is a whole set of
source tracks (prosody) or existing adjacent-pair groups (LIA). File locks provide
leases: a crash releases ownership; a receipt is committed only after validation.
"""
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'configs/glare_cache_28gpu_20261110.json'
STAGES=('prosody','lia')
_verified_inputs={}


class TaskDataError(RuntimeError):
    """Native task returned explicit per-input runtime failures; retry separately."""


def read(path):return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


@contextmanager
def locked(path,shared=False,blocking=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB))
        yield f


def config(path=DEFAULT):
    value=read(path)
    expected={'node8':('172.29.104.237',list(range(8))),
              'node12':('172.29.252.37',list(range(8))),
              'node4':('172.29.225.51',list(range(8))),
              'node11':('172.29.189.119',list(range(4)))}
    assert len(value['nodes'])==4 and {n['name'] for n in value['nodes']}==set(expected),'Expected four allocated nodes'
    for n in value['nodes']:
        host,gpus=expected[n['name']]
        assert n['host']==host and n['gpus']==gpus,'Host/GPU allocation differs from user request'
        assert set(n['prosody_preferred_gpus'])<=set(gpus)
        assert n['ssh_port']==22 and n['cpu_budget']>=16
    assert sum(len(n['gpus']) for n in value['nodes'])==28
    assert 28<=value['tasks_per_stage']<=4096
    assert value['official_policy']['early_stop'] is False
    assert value['official_policy']['generation']=='official_complete'
    assert value['official_policy']['normalization']=='official'
    assert value['automatic_start'] is False
    assert 1<=value['prosody']['batch_size']<=value['prosody']['queue_chunks']<=128
    execution(value)
    if value.get('benchmark'):
        assert Path(value['benchmark']['output_root']).resolve().is_relative_to(Path(value['work']).resolve())
    return value


def execution(cfg):
    p=dict(lia_workers_per_gpu=1,async_lia=False,async_prosody=False,
           lia_write_operations=16,prosody_write_operations=128,
           persistent_cpu_services=False,cpu_detectors_per_gpu=None,
           cpu_decoders_per_gpu=None,cpu_inflight_per_gpu=None,persistent_decode=False,gpu_fer=False)
    p.update(cfg.get('execution',{}))
    assert p['lia_workers_per_gpu'] in (1,2)
    assert isinstance(p['async_lia'],bool) and isinstance(p['async_prosody'],bool)
    assert isinstance(p['persistent_cpu_services'],bool)
    assert isinstance(p['persistent_decode'],bool)
    assert isinstance(p['gpu_fer'],bool)
    for key,upper in [('cpu_detectors_per_gpu',24),('cpu_decoders_per_gpu',16),('cpu_inflight_per_gpu',48)]:
        assert p[key] is None or isinstance(p[key],int) and 1<=p[key]<=upper
    assert 1<=p['lia_write_operations']<=64 and 1<=p['prosody_write_operations']<=256
    return p


def sources():
    names=['src/glare_cache_cluster.py','scripts/run_glare_cache_cluster.py',
           'scripts/glare_cache_cluster_node.py','scripts/glare_cache_cluster_worker.py',
           'scripts/glare_official_capacity_runtime.py','scripts/run_glare_source_prosody_full.py',
           'scripts/glare_visual_runtime.py','scripts/glare_decode_execution.py',
           'scripts/prepare_glare_official_conditions.py','scripts/glare_lia_fused_fp32.py',
           'scripts/glare_lia_fast_parallel.py','scripts/glare_lia_parallel.py',
           'scripts/glare_lia_invariants.py','scripts/glare_lia_execution.py',
           'scripts/glare_cluster_frozen_audio.py','scripts/glare_audio_execution_v2.py',
           'scripts/assemble_glare_training_caches.py','scripts/glare_async_writes.py',
           'scripts/prepare_glare_cache_benchmark.py','scripts/glare_cpu_service_pool.py',
           'scripts/glare_persistent_decode.py','scripts/glare_gpu_reactions.py']
    return {str(ROOT/n):sha(ROOT/n) for n in names}


def stat_key(path):
    s=Path(path).stat()
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)


@contextmanager
def ro(path):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    try:yield db
    finally:db.close()


def balanced(items,count):
    """LPT assigns each indivisible item once; deterministic tie breaking."""
    heap=[(0.,i) for i in range(count)];heapq.heapify(heap)
    for key,weight,units in sorted(items,key=lambda r:(-r[1],str(r[0]))):
        cost,task=heapq.heappop(heap)
        yield key,task,weight,units
        heapq.heappush(heap,(cost+weight,task))


def prepare(cfg):
    work=Path(cfg['work']);work.mkdir(parents=True,exist_ok=True)
    with locked(work/'prepare.lock'):
        if (work/'manifest.json').exists():return load_run(cfg,verify=True)
        plan=Path(cfg['source_plan']);old=Path(cfg['shared_lia'])
        p=read(plan/'protocol.json');l=read(old/'protocol.json');parent=Path(l['parent']);b=read(parent/'protocol.json')
        audio_protocol=read(old/'frozen_audio/protocol.json')
        assert audio_protocol['selection_sha256']==p['selection_sha256'] and audio_protocol['samples']==p['pairs']
        assert audio_protocol['pre_projection'] and audio_protocol['dtype']=='float32'
        assert p['selection_sha256']==l['selection_sha256']==b['selection_sha256']
        assert p['pairs']==l['samples']==b['samples']==284218
        frozen={str(plan/'index.sqlite'):p['index_sha256'],str(old/'groups.sqlite'):l['groups_sha256'],
                str(parent/'jobs.sqlite'):p['parent_jobs_sha256'],p['selection']:p['selection_sha256']}
        frozen.update(p['extractor']['code']);frozen.update(l['code']);frozen.update(b['code'])
        frozen.update(audio_protocol['code'])
        frozen[str(old/'frozen_audio/protocol.json')]=sha(old/'frozen_audio/protocol.json')
        frozen.update({str(x):sha(x) for x in (plan/'protocol.json',old/'protocol.json',parent/'protocol.json')})
        bench=cfg.get('benchmark');cohort=read(bench['cohort']) if bench else None
        if cohort:
            assert cohort['source_index_sha256']==p['index_sha256'] and cohort['groups_sha256']==l['groups_sha256']
            frozen[bench['cohort']]=sha(bench['cohort'])
        for path,want in frozen.items():
            key=(want,stat_key(path))
            if _verified_inputs.get(path)==key:continue
            assert sha(path)==want,'Frozen input/code changed: '+path
            assert stat_key(path)==key[1],'Frozen input changed during verification'
            _verified_inputs[path]=key
        assert sha(parent/'protocol.json')==p['parent_protocol_sha256']==l['parent_protocol_sha256']
        acceptance=old/'gpu7_only/lia_fused_fp32_acceptance/RESULT.json'
        frozen[str(acceptance)]=sha(acceptance)
        assert read(acceptance)['passed'],'Existing fused FP32 acceptance failed'
        count=cfg['tasks_per_stage'];tmp=work/'routing.sqlite.tmp'
        tmp.unlink(missing_ok=True)
        db=sqlite3.connect(tmp)
        db.executescript('CREATE TABLE prosody_assign(track_id TEXT PRIMARY KEY,task_id INTEGER);'
                         'CREATE TABLE lia_assign(gid INTEGER PRIMARY KEY,task_id INTEGER);'
                         'CREATE TABLE tasks(stage TEXT,task_id INTEGER,units INTEGER,weight REAL,items INTEGER,PRIMARY KEY(stage,task_id));')
        with ro(plan/'index.sqlite') as source:
            tracks=[(tid,n,n) for tid,n in source.execute('SELECT track_id,COUNT(*) FROM chunks GROUP BY track_id')]
            assert len(tracks)==p['tracks'] and sum(t[2] for t in tracks)==p['chunks']
        with ro(old/'groups.sqlite') as source:
            groups=[(gid,seconds,pairs) for gid,seconds,pairs in source.execute('SELECT gid,seconds,pairs FROM groups')]
        assert sum(g[2] for g in groups)==p['pairs']
        if cohort:
            tids=set(cohort['track_ids']);gids=set(cohort['group_ids'])
            tracks=[t for t in tracks if t[0] in tids];groups=[g for g in groups if g[0] in gids]
            assert len(tracks)==len(tids) and len(groups)==len(gids)
        for stage,items,table,key in [('prosody',tracks,'prosody_assign','track_id'),('lia',groups,'lia_assign','gid')]:
            sums=[[0,0.,0] for _ in range(count)]
            task_count=bench['tasks'][stage] if bench else count
            sums=[[0,0.,0] for _ in range(task_count)]
            for ident,task,weight,units in balanced(items,task_count):
                db.execute(f'INSERT INTO {table}({key},task_id) VALUES(?,?)',(ident,task))
                sums[task][0]+=units;sums[task][1]+=weight;sums[task][2]+=1
            db.executemany('INSERT INTO tasks VALUES(?,?,?,?,?)',[(stage,i,*v) for i,v in enumerate(sums) if v[2]])
            db.execute(f'CREATE INDEX {table}_task ON {table}(task_id)')
        db.commit();db.close();os.replace(tmp,work/'routing.sqlite')
        assets=dict(p['extractor']['assets_sha256']);assets[b['model']]=b['model_sha256']
        assets.update(audio_protocol['assets'])
        manifest=dict(schema='glare-official-cache-cluster-v1',created_at=time.time(),config=cfg,
                      config_sha256=digest(cfg),code=sources(),frozen=frozen,
                      frozen_stats={k:stat_key(k) for k in frozen},assets_sha256=assets,
                      routing_sha256=sha(work/'routing.sqlite'),pairs=p['pairs'],chunks=p['chunks'],tracks=p['tracks'],
                      parent=str(parent),cache_contract=p['extractor'],cache_root=p['cache_root'],
                      selection_sha256=p['selection_sha256'],official_policy=cfg['official_policy'])
        manifest['frozen_audio_protocol']=audio_protocol
        manifest['shared_protocol']=l
        manifest['scope']='isolated fixed-cohort benchmark' if bench else 'production full500h'
        manifest['assigned_units']=dict(prosody=sum(t[2] for t in tracks),lia=sum(g[2] for g in groups))
        manifest['required_caches']=['LIA motion','normalized speaker audio','motion statistics','reaction and validity mask',
                                     'Wav2Vec2 pre-projection','SER probabilities','official prosody intensity and validity mask']
        # Operational batch64 overrides historical batch_cap16 metadata only;
        # cache IDs, model, prompt, tokens, source normalization are unchanged.
        manifest['operational_batch_override']=dict(old_recorded_cap=p['extractor']['batch_cap'],new=cfg['prosody']['batch_size'])
        manifest['run_id']=digest(manifest)
        save(work/'manifest.json',manifest)
        return manifest


def load_run(cfg,verify=True):
    m=read(Path(cfg['work'])/'manifest.json')
    assert m['config_sha256']==digest(cfg),'Config changed; prepare a separate run directory'
    identity=dict(m);identity.pop('run_id')
    assert m['run_id']==digest(identity),'Manifest changed'
    if verify:
        assert m['code']==sources(),'Cluster execution source changed; re-prepare before launch'
        assert sha(Path(cfg['work'])/'routing.sqlite')==m['routing_sha256'],'Routing index changed'
    return m


class Lease:
    def __init__(self,queue,task,handle):self.queue=queue;self.task=task;self.handle=handle
    def finish(self,result):
        assert result.get('validated') is True,'Completion requires validated native output'
        save(self.queue.folder(self.task)/'done.json',dict(run_id=self.queue.run_id,task=self.task,
             task_sha256=digest(self.task),result=result,finished=time.time()))
        self.close()
    def close(self):
        if self.handle is not None:self.handle.close();self.handle=None
    def __enter__(self):return self
    def __exit__(self,typ,error,tb):
        retryable=isinstance(error,TaskDataError)
        try:
            if retryable:
                path=self.queue.folder(self.task)/'attempts.json'
                attempts=read(path)['attempts'] if path.exists() else []
                attempts.append(dict(at=time.time(),error=str(error),pid=os.getpid()))
                save(path,dict(run_id=self.queue.run_id,task_sha256=digest(self.task),attempts=attempts))
        finally:self.close()
        # Per-input failures do not kill a healthy GPU or make all other tasks
        # wait behind a bad file. API/integrity/model failures remain fatal.
        return retryable


class TaskQueue:
    def __init__(self,work,run_id,stage):
        assert stage in STAGES
        self.work=Path(work);self.run_id=run_id;self.stage=stage
        with ro(self.work/'routing.sqlite') as db:
            db.row_factory=sqlite3.Row
            self.tasks=[dict(r) for r in db.execute('SELECT * FROM tasks WHERE stage=? ORDER BY task_id',(stage,))]
    def folder(self,task):return self.work/'tasks'/self.stage/('%04d'%task['task_id'])
    def done(self,task):
        path=self.folder(task)/'done.json'
        if not path.exists():return False
        value=read(path)
        assert value['run_id']==self.run_id and value['task_sha256']==digest(task),'Stale task completion receipt'
        return True
    def claim(self,owner):
        for task in self.tasks:
            if self.done(task) or self.failed(task):continue
            folder=self.folder(task);folder.mkdir(parents=True,exist_ok=True)
            f=(folder/'lease.lock').open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:f.close();continue
            if self.done(task) or self.failed(task):f.close();continue
            save(folder/'owner.json',dict(owner=owner,pid=os.getpid(),host=socket.gethostname(),claimed=time.time(),run_id=self.run_id))
            return Lease(self,task,f)
        return None
    def summary(self):
        completed=[t for t in self.tasks if self.done(t)]
        failed=[t for t in self.tasks if not self.done(t) and self.failed(t)]
        return dict(tasks=len(self.tasks),done_tasks=len(completed),units=sum(t['units'] for t in self.tasks),
                    done_units=sum(t['units'] for t in completed),complete=len(completed)==len(self.tasks),
                    failed_tasks=len(failed),failed_task_ids=[t['task_id'] for t in failed],
                    exhausted=len(completed)+len(failed)==len(self.tasks))
    def failed(self,task):
        path=self.folder(task)/'attempts.json'
        if not path.exists():return False
        value=read(path)
        assert value['run_id']==self.run_id and value['task_sha256']==digest(task),'Stale error receipt'
        return len(value['attempts'])>=3
    def retry_errors(self):
        reset=[]
        for task in self.tasks:
            if not self.failed(task) or self.done(task):continue
            with locked(self.folder(task)/'lease.lock'):
                path=self.folder(task)/'attempts.json'
                os.replace(path,path.with_name('attempts.history.'+str(time.time_ns())+'.json'));reset.append(task['task_id'])
        return reset


def fences(cfg,m):
    old=Path(cfg['shared_lia']);raw=Path(m['cache_root'])
    # ChunkCache adds its canonical contract ID beneath cache_root.
    from glare_prosody_subset import canonical
    return [old/'run_all.lock',old/'controller.lock',
            *[old/'workers'/('gpu%d.lock'%g) for g in (4,5,6,7)],
            *[old/'frozen_audio/workers'/('gpu%d.lock'%g) for g in (4,5,6,7)],
            raw/canonical(m['cache_contract'])/'mutation.lock']


@contextmanager
def writer_fences(cfg,m,shared=True):
    with ExitStack() as stack:
        handles=[stack.enter_context(locked(p,shared=shared)) for p in fences(cfg,m)]
        yield handles


def assert_active(cfg,m):
    marker=read(Path(cfg['shared_lia'])/'cluster_active.json')
    assert marker['run_id']==m['run_id'],'Different cluster run owns the shared caches'


def node_by_name(cfg,name):return next(n for n in cfg['nodes'] if n['name']==name)


def assert_local(node):
    addresses=subprocess.check_output(['hostname','-I'],text=True).split()
    assert node['host'] in addresses,'Refusing node identity mismatch: '+node['name']


def cpu_budget(node):
    count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    quota=Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        q,p=quota.read_text().split()
        if q!='max':count=min(count,max(1,math.floor(int(q)/int(p))))
    return min(node['cpu_budget'],count)


def resources(cfg,node):
    budget=cpu_budget(node);slots=len(node['gpus'])
    # Bound total child-process fanout across every GPU on this host.
    per=max(4,(budget-2*slots)//slots)
    decode=max(1,min(cfg['lia']['decode_workers_cap'],per//3))
    # Each reaction service owns MP plus a FER subprocess (two TF threads).
    detectors=max(1,min(cfg['lia']['detector_services_cap'],(per-decode)//3))
    tune=execution(cfg)
    if tune['cpu_detectors_per_gpu'] is not None:
        decode=tune['cpu_decoders_per_gpu'] or decode
        # A native service performs MP then FER sequentially; their respective
        # main compute phases do not run concurrently within the same service.
        # Retain two CPU cores per service, plus decoder/GPU parent budgets.
        max_services=max(1,int((budget/slots-decode-2*tune['lia_workers_per_gpu'])//2))
        detectors=min(tune['cpu_detectors_per_gpu'],max_services)
    shm=os.statvfs('/dev/shm');available=shm.f_bavail*shm.f_frsize
    memory_cap=max(1,int(available*.6/(slots*2*1024**3)))
    inflight=tune['cpu_inflight_per_gpu'] or min(cfg['lia']['inflight_groups_cap'],decode+detectors)
    return dict(cpu_budget=budget,decode_workers=decode,detector_services=detectors,
                inflight_groups=max(1,min(inflight,memory_cap)),
                shm_available_gib=available/1024**3)


def verify_assets(m,path):
    proof=read(path)
    assert proof['run_id']==m['run_id'] and proof['assets_sha256']==m['assets_sha256']
    assert set(proof['stats'])==set(m['assets_sha256']),'Incomplete asset verification receipt'
    for asset,want in proof['stats'].items():assert stat_key(asset)==want,'Model asset changed: '+asset


def node_assets(cfg,m,node):
    path=Path(cfg['work'])/'nodes'/node['name']/'assets.json'
    try:verify_assets(m,path);return path
    except (FileNotFoundError,AssertionError):pass
    previous=read(path) if path.exists() else {};reused=0
    for asset,want in m['assets_sha256'].items():
        if previous.get('assets_sha256',{}).get(asset)==want and previous.get('stats',{}).get(asset)==stat_key(asset):
            reused+=1;continue
        assert sha(asset)==want,'Model checksum mismatch: '+asset
    save(path,dict(run_id=m['run_id'],assets_sha256=m['assets_sha256'],
                   stats={k:stat_key(k) for k in m['assets_sha256']},verified_at=time.time(),reused_asset_checks=reused))
    return path


def gpu_info():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(v[0]),name=v[1].strip(),total_mib=int(v[2]),used_mib=int(v[3]),utilization=int(v[4]))
            for line in rows.splitlines() if (v:=line.split(','))]


def preflight(cfg,m,node,full_assets=False):
    assert_local(node);checks={};errors=[]
    for path in [cfg['python'],*m['assets_sha256'],*m['frozen']]:
        if not Path(path).is_file():errors.append('missing: '+path)
    errors.extend('missing runtime directory: '+p for p in cfg['pythonpath'] if not Path(p).is_dir())
    # Hash immutable inputs once per node; subsequent workers check size/mtime/inode.
    for path,want in m['frozen'].items():
        if Path(path).is_file() and sha(path)!=want:errors.append('checksum: '+path)
    checks['frozen_stats']={p:stat_key(p) for p in m['frozen'] if Path(p).is_file()}
    checks['resources']=resources(cfg,node);checks['gpus']=gpu_info()
    assigned=[g for g in checks['gpus'] if g['index'] in node['gpus']]
    if len(assigned)!=len(node['gpus']):errors.append('missing assigned GPUs')
    checks['busy_assigned_gpus']=[g['index'] for g in assigned if g['used_mib']>cfg['preflight']['max_used_gpu_memory_mib']]
    env_paths=[ROOT/'experiments/listen_j_omni_20261030/nod_cpu_share/runtime_env/bin/python',
               ROOT/'experiments/listen_j_reaction_vocal_omni_20261103/reaction_runtime_env/bin/python',
               ROOT/'compare/AvatarForcing/pretrained_dir/wav2vec2-base-960h/preprocessor_config.json']
    errors.extend('missing: '+str(p) for p in env_paths if not p.is_file())
    probes=[(Path(cfg['python']),['torch','transformers','triton','librosa','cv2','soundfile','scipy']),
            (env_paths[0],['mediapipe','numpy','cv2']),
            (env_paths[1],['tensorflow','torch','fer.fer','cv2'])]
    checks['environments']=[]
    for interpreter,modules in probes:
        if not interpreter.is_file():continue
        probe='import importlib,json;names='+repr(modules)+';mods={m:importlib.import_module(m) for m in names};'
        if interpreter==Path(cfg['python']):
            probe+='from transformers import AutoProcessor,Qwen2AudioForConditionalGeneration,Wav2Vec2FeatureExtractor;'
        probe+='print(json.dumps({m:dict(file=getattr(v,"__file__",None),version=getattr(v,"__version__",None)) for m,v in mods.items()}))'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        if interpreter==Path(cfg['python']):env['PYTHONPATH']=':'.join(cfg['pythonpath'])
        else:env['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT/'scripts')
        result=subprocess.run([str(interpreter),'-c',probe],text=True,capture_output=True,timeout=90,
                              env=env)
        report=dict(python=str(interpreter),returncode=result.returncode,stdout=result.stdout[-2000:],stderr=result.stderr[-1000:])
        checks['environments'].append(report)
        if result.returncode:errors.append('environment: '+str(interpreter))
    with ro(Path(cfg['source_plan'])/'index.sqlite') as db:
        track=json.loads(db.execute('SELECT data FROM tracks LIMIT 1').fetchone()[0])
    if not Path(track['audio']['path']).is_file():errors.append('source WAV unavailable: '+track['audio']['path'])
    # Shared nonce proves the remote interpreter sees this exact prepared run.
    if read(Path(cfg['work'])/'manifest.json')['run_id']!=m['run_id']:errors.append('shared run identity mismatch')
    if full_assets and not errors:checks['assets_receipt']=str(node_assets(cfg,m,node))
    checks.update(node=node['name'],host=node['host'],run_id=m['run_id'],checked_at=time.time(),errors=errors,
                  ready=not errors and not checks['busy_assigned_gpus'],full_assets=full_assets)
    save(Path(cfg['work'])/'nodes'/node['name']/'preflight.json',checks)
    return checks


def validate_node_inputs(m,proof):
    assert proof['run_id']==m['run_id'] and not proof['errors']
    assert set(proof['frozen_stats'])==set(m['frozen'])
    for p,want in proof['frozen_stats'].items():assert stat_key(p)==want,'Frozen input changed: '+p


def slot_dir(cfg,node,gpu,worker_slot=0):
    root=Path(cfg['work'])/'nodes'/node['name']/('gpu%d'%gpu)
    return root if worker_slot==0 else root/('worker%d'%worker_slot)


def worker_resources(cfg,node,worker_slot=0):
    caps=dict(resources(cfg,node));count=execution(cfg)['lia_workers_per_gpu']
    assert 0<=worker_slot<count
    for key in ('decode_workers','detector_services','inflight_groups'):
        total=caps[key]
        assert total>=count,'CPU/shared-memory budget cannot support requested workers per GPU'
        caps[key]=total//count+int(worker_slot<total%count)
    return dict(caps,worker_slot=worker_slot,workers_per_gpu=count,allocation='split fixed per-GPU CPU and in-flight budget')


def environment(node,gpu,cfg=None):
    cfg=cfg or config()
    local='/tmp/glare-cache-'+node['name']+'-gpu'+str(gpu)
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TRITON_CACHE_DIR=local+'/triton',
                TORCHINDUCTOR_CACHE_DIR=local+'/inductor',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                PYTHONPATH=':'.join(cfg['pythonpath']),PYTHONDONTWRITEBYTECODE='1')
