#!/usr/bin/env python3
"""Speaker Whisper increment using the unchanged native FP32 batched extractor."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl,json,os,signal,subprocess,sys,threading,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_j_current_qwen_hidden import Reader as HReader,ensure,item_from_row,remote,env
from p2_j_current_av_cache import free_gpus
import p2_whisper_av.cache as native
NativeReader=native.Reader
SOURCE=ROOT/'experiments/p2_j_train_dev500h_qwen_hidden_20261105'
AUDIT=ROOT/'experiments/p2_j_train_dev500h_aux_reuse_audit_20261105'
OLD=ROOT/'experiments/p2_j_speaker_whisper_cache_20261104_prefetch_v2'
WORK=ROOT/'experiments/p2_j_train_dev500h_whisper_cache_20261105'


def adapter_identity():
    return {str(p):sha(p) for p in [Path(__file__),ROOT/'scripts/p2_j_current_qwen_hidden.py',ROOT/'src/p2_whisper_av/cache.py',ROOT/'src/p2_whisper_av/features.py']}


def prepare(work):
    w=Path(work);w.mkdir(parents=True,exist_ok=True);a=read(AUDIT/'SUMMARY.json');ensure(a['status']=='complete' and not a['errors'],'Aux reuse audit not complete')
    old=NativeReader(OLD,require_complete=True);h=HReader(SOURCE);ids=read(AUDIT/'whisper_increment_ids.json');ensure(len(ids)==6542,'Unexpected Whisper increment')
    ensure(native.model_contract(old.protocol['encoder']['path'])==old.protocol['encoder'],'Whisper model/numerics changed')
    p=dict(schema=native.CACHE_VERSION,source=str(SOURCE),source_protocol_sha256=sha(SOURCE/'protocol.json'),source_complete_sha256=sha(SOURCE/'COMPLETE.json'),sample_ids=ids,sample_ids_sha256=digest(ids),samples=len(ids),encoder=old.protocol['encoder'],code=native.feature_code_identity(),adapter_code=adapter_identity(),
           old=str(OLD),old_protocol_sha256=sha(OLD/'protocol.json'),old_complete_sha256=sha(OLD/'COMPLETE.json'),reuse_summary_sha256=sha(AUDIT/'SUMMARY.json'),reuse_map_sha256=sha(AUDIT/'reuse_map.jsonl'),full_samples=284218,source_split_counts=h.protocol['source_split_counts'],selection_sha256=h.protocol['selection_sha256'],intermediate_wav=False,intermediate_video=False)
    atomic_json(w/'protocol.json',p,immutable=True);atomic_json(w/'cluster.json',read(SOURCE/'cluster.json'),immutable=True)
    atomic_json(w/'config.json',dict(j_cache=str(SOURCE),whisper_cache=str(w),whisper=dict(model_path=p['encoder']['path'],input_dim=1280)),immutable=True)
    print(json.dumps(dict(prepared=True,increment=len(ids),reuse=277676,full=284218)))


class IncrementReader(NativeReader):
    def __init__(self,work,*,require_complete=False,source=None):
        self.work=Path(work);self.protocol=read(self.work/'protocol.json');p=self.protocol
        ensure(p['code']==native.feature_code_identity() and p['adapter_code']==adapter_identity(),'Whisper adapter/native code changed')
        ensure(sha(SOURCE/'protocol.json')==p['source_protocol_sha256'] and sha(SOURCE/'COMPLETE.json')==p['source_complete_sha256'],'Frozen current source changed')
        ensure(source is None or Path(source).resolve()==SOURCE,'Wrong Whisper speaker source')
        ensure(len(set(p['sample_ids']))==p['samples'] and digest(p['sample_ids'])==p['sample_ids_sha256'],'Wrong increment members')
        self.ids=set(p['sample_ids']);self.source=SOURCE;self.local=threading.local()
        if require_complete:ensure(read(self.work/'COMPLETE.json')['success'],'Whisper increment not complete')
    def item(self,sid):
        ensure(sid in self.ids,'Outside Whisper increment')
        if not hasattr(self.local,'h'):self.local.h=HReader(SOURCE,verify=False)
        return item_from_row(self.local.h.row(sid))


class Reader:
    """Full 284218-member view, preserving old Whisper files and original splits."""
    def __init__(self,work=WORK,require_complete=True):
        self.work=Path(work);self.new=IncrementReader(work,require_complete=require_complete);p=self.new.protocol;self.source=HReader(SOURCE,verify=False)
        ensure(sha(OLD/'protocol.json')==p['old_protocol_sha256'] and sha(OLD/'COMPLETE.json')==p['old_complete_sha256'],'Audited old Whisper changed')
        self.old=NativeReader(OLD,require_complete=True);self.ids=set(self.source.protocol['sample_ids'])
        if require_complete:ensure(read(self.work/'MIXED_COMPLETE.json')['samples']==284218,'Full Whisper map incomplete')
    def load(self,sid,device='cpu'):
        ensure(sid in self.ids,'Outside current 500h');return self.new.load(sid,device) if sid in self.new.ids else self.old.load(sid,device)
    def sample_ids(self):return list(self.source.protocol['sample_ids'])


def worker(work,rank):
    import torch
    torch.set_num_threads(2);native.Reader=IncrementReader
    result=native.extract(read(Path(work)/'config.json'),rank=rank,world=32,device='cuda:0',batch_size=16,read_workers=4)
    print(json.dumps(result),flush=True)


def node(work,name):
    folder=Path(work)/'nodes'/name;folder.mkdir(parents=True,exist_ok=True);cluster=read(Path(work)/'cluster.json');index=next(i for i,n in enumerate(cluster['nodes']) if n['name']==name)
    children={};logs=[];started=set()
    with (folder/'node.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                for gpu in free_gpus():
                    if gpu in started:continue
                    rank=index*8+gpu;cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--work',work,'--rank',str(rank)]
                    log=(folder/f'g{gpu}.log').open('a');logs.append(log);children[gpu]=subprocess.Popen(cmd,cwd=ROOT,env=env(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);started.add(gpu)
                    atomic_json(folder/f'g{gpu}.pid.json',dict(pid=children[gpu].pid,rank=rank,gpu=gpu))
                states={str(g):p.poll() for g,p in children.items()};atomic_json(folder/'status.json',dict(time=time.time(),states=states,node=name,gpus_started=sorted(started)))
                snap=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
                with (folder/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu_csv=snap.stdout))+'\n')
                if len(started)==8 and all(x is not None for x in states.values()):break
                time.sleep(5)
            atomic_json(folder/'EXIT.json',dict(time=time.time(),states=states))
        finally:
            for p in children.values():
                if p.poll() is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for f in logs:f.close()


def launch(cluster,n,work):
    code='''import sys,subprocess,json,os,time
from pathlib import Path
root,work,node,script=sys.argv[1:];f=Path(work)/'nodes'/node;f.mkdir(parents=True,exist_ok=True);pf=f/'launch.json'
if pf.exists():
 d=json.loads(pf.read_text());p=Path('/proc')/str(d['pid'])/'cmdline'
 if p.exists() and script.encode() in p.read_bytes() and work.encode() in p.read_bytes():print(json.dumps(dict(already_running=True,pid=d['pid'])));sys.exit()
cmd=[sys.executable,script,'node','--work',work,'--node',node];e=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
with (f/'supervisor.log').open('a') as log:p=subprocess.Popen(cmd,cwd=root,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,command=cmd,time=time.time());pf.write_text(json.dumps(d));print(json.dumps(d))'''
    return json.loads(remote(cluster,n,[cluster['python'],'-c',code,str(ROOT),str(work),n['name'],str(Path(__file__).resolve())]))


def chain(work):
    w=Path(work);p=read(w/'protocol.json');c=read(w/'cluster.json')
    def state(stage,**kw):atomic_json(w/'chain_status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
    with (w/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (w/'MIXED_COMPLETE.json').exists():state('complete');return
            with ThreadPoolExecutor(max_workers=4) as pool:launches=list(pool.map(lambda n:launch(c,n,w),c['nodes']))
            state('extracting',launches=launches)
            while True:
                rows=[read(f) for f in (w/'progress').glob('rank*.json')];done=sum(x['new']+x['reused'] for x in rows);finished=sum(x['finished'] for x in rows)
                state('extracting',increment_done=done,increment_total=6542,full_ready=277676+done,total=284218,finished_ranks=finished)
                if len(rows)==finished==32:break
                exits=[w/'nodes'/n['name']/'EXIT.json' for n in c['nodes']]
                if all(f.exists() for f in exits):raise RuntimeError('Whisper nodes exited without full increment; inspect logs')
                time.sleep(10)
            state('auditing');native.Reader=IncrementReader;result=native.finalize(read(w/'config.json'),read_workers=16)
            ensure(result['success'] and result['samples']==6542,'Wrong Whisper increment coverage')
            full=dict(success=True,samples=284218,reused=277676,new=6542,source_split_counts=p['source_split_counts'],selection_sha256=p['selection_sha256'],protocol_sha256=sha(w/'protocol.json'),increment_complete_sha256=sha(w/'COMPLETE.json'),old_complete_sha256=p['old_complete_sha256'],audio_source='speaker',dtype='float32',training_started=False,time=time.time())
            atomic_json(w/'MIXED_COMPLETE.json',full,immutable=True);state('complete',result=full)
        except BaseException:state('attention_required',error=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','worker','node','chain']);p.add_argument('--work',default=str(WORK));p.add_argument('--rank',type=int,default=0);p.add_argument('--node',default='node12');a=p.parse_args()
    if a.command=='prepare':prepare(a.work)
    elif a.command=='worker':worker(a.work,a.rank)
    elif a.command=='node':node(a.work,a.node)
    elif a.command=='chain':chain(a.work)

if __name__=='__main__':main()
