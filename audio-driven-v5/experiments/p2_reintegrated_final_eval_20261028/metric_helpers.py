"""Work stealing for unchanged metric rows and independent CPU detections."""
import sys,os,fcntl,uuid,subprocess,threading,time
from pathlib import Path
from eval_common import HERE,ROOT,PY,read,write,sha
sys.path.insert(0,str(ROOT/'src'))

def row(work,sid,seed):
    from p2_omni_eval.pipeline import compute_row,check_protocol,has_error
    from p2_omni_eval.backends import Backends
    from p2_omni_eval.common import digest,job_key,artifact_valid,complete
    work=Path(work);p=read(work/'protocol.json');check_protocol(p)
    selected=next(r for r in p['rows'] if r['sample_id']==sid and r['sample_seed']==seed)
    dest=work/'rows'/job_key(sid,seed);dest.mkdir(parents=True,exist_ok=True)
    ident=dict(protocol=digest(p),row=selected,device='cuda:0')
    with (dest/'row.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        if artifact_valid(dest,ident):return
        backend=Backends(p,'cuda:0');attempt=dest/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
        result=compute_row(selected,p,backend,attempt);write(attempt/'result.json',result);write(dest/'result.json',result)
        if has_error(result):raise RuntimeError('Metric helper row failed: '+str(dest))
        complete(dest,ident,[dest/'result.json',*attempt.glob('*.npy')])

def detector(rank):
    from p2_glare_rgb.detector import Engine
    from p2_glare_unfiltered.rules import process
    from p2_glare_eval.common import artifact_valid
    engine=None;completed=0;skipped=0
    for item in reversed(read(HERE/'metric_helpers/videos.json')):
        h=item['sha256'];src=HERE/'detections/rgb'/h;out=HERE/'detections/unfiltered'/h
        locks=HERE/'detections/locks';locks.mkdir(parents=True,exist_ok=True)
        with (locks/(h+'.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:skipped+=1;continue
            if all((p/'complete.json').exists() and artifact_valid(p,read(p/'complete.json')['contract']) for p in [src,out]):continue
            if engine is None:engine=Engine()
            engine.detect(item['video'],src,h);process(item['video'],src,out,h);completed+=1
            write(HERE/'metric_helpers'/f'detector_{rank}.json',dict(completed=completed,busy_skipped=skipped,done=False))
    write(HERE/'metric_helpers'/f'detector_{rank}.json',dict(completed=completed,busy_skipped=skipped,done=True))

def node(index):
    children=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        MKL_NUM_THREADS='2',TF_ENABLE_ONEDNN_OPTS='0',TF_CPP_MIN_LOG_LEVEL='2')
    for local in range(4):
        rank=index*4+local;log=(HERE/'metric_helpers'/f'detector_{rank}.log').open('a')
        children.append(subprocess.Popen([str(ROOT/'.venv-p2-glare-detector/bin/python'),str(__file__),'detect',str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in children]
    if any(codes):raise RuntimeError(str(codes))

def launch():
    import p2_guided_chain.backend
    sys.path.append(str(ROOT/'scripts'));from listen_cluster import launch as run
    cfg=read(ROOT/'experiments/p2_full50h_duration384_20261028/cluster.json')
    run(cfg,[[str(PY),str(__file__),'node',str(i)] for i in range(3)],HERE/'metric_helpers/launches'/str(time.time_ns()))

if __name__=='__main__':
    if sys.argv[1]=='row':row(sys.argv[2],sys.argv[3],int(sys.argv[4]))
    elif sys.argv[1]=='detect':detector(int(sys.argv[2]))
    elif sys.argv[1]=='node':node(int(sys.argv[2]))
    elif sys.argv[1]=='launch':launch()
    else:raise ValueError(sys.argv[1])
