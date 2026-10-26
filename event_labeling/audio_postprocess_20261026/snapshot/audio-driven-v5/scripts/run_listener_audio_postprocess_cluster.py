#!/usr/bin/env python3
"""Run the approved audio-only postprocessing on the fixed train50h subset."""
import argparse,json,os,shlex,subprocess,sys,time,traceback,tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json
PYTHON='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
KEY='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'
NODES={'node17':'172.29.15.91','node12':'172.29.252.11','node4':'172.29.225.51'}
SCRIPT=Path(__file__).resolve()


def remote(node,code):
    r=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=10','root@'+NODES[node],shlex.join([PYTHON,'-c',code])],capture_output=True,text=True,timeout=45)
    if r.returncode:raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout)


def verify_code(run):
    assert all(sha(ROOT/k)==v for k,v in read(run/'code_manifest.json').items()),'Source changed during run'


def node_worker(run,node):
    from listen_benchmark.full_fusion import lock
    with lock(run/f'nodes/{node}.lock') as acquired:
        if not acquired:raise RuntimeError('Node already active')
        verify_code(run);node_index=list(NODES).index(node);pending=[];active={};done=set();attempts={}
        for lane in range(4):
            for gpu in range(8):pending.append((node_index*32+lane*8+gpu,gpu))
        logs=run/'logs';logs.mkdir(exist_ok=True);started=time.monotonic()
        while pending or active:
            # Four independent native encoders per GPU, no data/precision changes.
            while pending and len(active)<32:
                rank,gpu=pending.pop(0)
                if (run/'workers'/f'{rank:03d}.json').exists() and read(run/'workers'/f'{rank:03d}.json').get('finished'):
                    done.add(rank);continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
                with (logs/f'worker-{rank:03d}.log').open('a') as log:
                    child=subprocess.Popen([PYTHON,str(SCRIPT),'worker','--run',str(run),'--rank',str(rank)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[rank]=(child,gpu);attempts[rank]=attempts.get(rank,0)+1
                time.sleep(.2)
            for rank,(child,gpu) in list(active.items()):
                rc=child.poll()
                if rc is None:continue
                del active[rank]
                if rc==0:done.add(rank)
                elif attempts[rank]<2:pending.append((rank,gpu))
                else:raise RuntimeError(f'worker {rank} failed twice; inspect its log')
            smi=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],text=True,capture_output=True)
            record=dict(timestamp=time.time(),node=node,active=len(active),finished_shards=len(done),total_shards=32,seconds=time.monotonic()-started,gpu=smi.stdout.strip().splitlines(),loadavg=os.getloadavg(),finished=not pending and not active)
            atomic_json(run/'nodes'/f'{node}.json',record)
            with (run/'nodes'/f'{node}-history.jsonl').open('a') as history:history.write(json.dumps(record)+'\n')
            if active:time.sleep(5)
        return record


def coordinator(run):
    from listen_benchmark.full_fusion import lock
    from listen_audio_postprocess.bulk import build_records,audit_all
    from listen_audio_postprocess.prepare import prepare_view
    with lock(run/'controller.lock') as acquired:
        if not acquired:raise RuntimeError('Controller already active')
        if (run/'COMPLETE.json').exists():return read(run/'COMPLETE.json')
        verify_code(run);started=time.monotonic()
        def stage(name,**extra):
            value=dict(stage=name,elapsed_s=time.monotonic()-started,training_started=False,**extra);atomic_json(run/'status.json',value);print(json.dumps(value),flush=True)
        cfg=read(run/'launch_config.json');base=Path(cfg['base_config']).parent
        stage('official_metadata_and_labels')
        if not (run/'records_complete.json').exists():
            build_records(cfg['dataset'],base,cfg['feature_run'],run,24)
        assert sha(run/'records.jsonl')==read(run/'records_complete.json')['records_sha256']
        stage('prepare_virtual_view')
        def records():
            with (run/'records.jsonl').open() as f:
                for line in f:yield json.loads(line)
        preparation_pool=ThreadPoolExecutor(max_workers=1)
        preparation=preparation_pool.submit(prepare_view,cfg['base_config'],records(),run/'view',32,read(run/'records_complete.json')['records_sha256'])
        while not (run/'view/config.json').exists():
            if preparation.done():preparation.result()
            time.sleep(.1)
        obs=read(run/'view/manifests/observations.json');targets={x['sample_id']:x for x in read(run/'view/manifests/targets.json')}
        for rank in range(96):atomic_json(run/'tasks'/f'{rank:03d}.json',[dict(observation=o,target=targets[o['sample_id']]) for o in obs[rank::96]],immutable=True)
        verify_code(run);stage('audio_cache_24gpu',samples=len(obs),workers=96)
        def launch(node):
            code='import subprocess,json; from pathlib import Path; '
            code+=f'p=Path({str(run)!r}); log=(p/"logs"/{("node-"+node+".log")!r}).open("a"); '
            code+=f'child=subprocess.Popen({[PYTHON,str(SCRIPT),"node","--run",str(run),"--node",node]!r},cwd={str(ROOT)!r},stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print(json.dumps(dict(pid=child.pid)))'
            return remote(node,code)
        (run/'logs').mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=3) as pool:launches=dict(zip(NODES,pool.map(launch,NODES)))
        atomic_json(run/'node_launches.json',launches)
        last_done=0;last_time=time.monotonic();last_progress=last_time
        while True:
            if preparation.done():preparation.result()
            reports=[read(p) for p in (run/'workers').glob('*.json')];done=sum(x['completed'] for x in reports)
            finished=sum(x.get('finished',False) for x in reports);now=time.monotonic()
            resources={n:read(run/'nodes'/f'{n}.json') for n in NODES if (run/'nodes'/f'{n}.json').exists()}
            stage('audio_cache_24gpu',completed=done,total=len(obs),finished_shards=finished,total_shards=96,
                  window_samples_per_s=(done-last_done)/max(now-last_time,.001),nodes=resources)
            if done>last_done:last_progress=now
            last_done,last_time=done,now
            if finished==96:break
            failures=list((run/'nodes').glob('*.error.json'))
            if failures:raise RuntimeError(f'Node failure: {failures}')
            if now-last_progress>600:raise RuntimeError('No cache progress for ten minutes; retained all logs and partial caches')
            time.sleep(15)
        preparation.result();preparation_pool.shutdown()
        stage('final_audit');result=audit_all(run,16);stage('complete',summary=result)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['init','run','node','worker','audit']);p.add_argument('--run',type=Path,required=True);p.add_argument('--node',choices=list(NODES));p.add_argument('--rank',type=int)
    a=p.parse_args();run=a.run.resolve();run.mkdir(parents=True,exist_ok=True)
    try:
        if a.command=='init':
            cfg=dict(dataset=str(ROOT/'experiments/p2_train50h_backchannel_face_20261026'),base_config=str(ROOT/'experiments/p2_e1_train50h_lora_tb_20261026/config.json'),feature_run=str(ROOT/'experiments/p2_features_full_crop384_20261025'),nodes=NODES,gpus=24,workers_per_gpu=4,shards=96,training_started=False)
            atomic_json(run/'launch_config.json',cfg,immutable=True)
            files=list((ROOT/'src/listen_audio_postprocess').glob('*.py'))+[SCRIPT,ROOT/'scripts/p2_audio_postprocess.py',ROOT/'scripts/compare_listener_official_six.py',ROOT/'src/p2_listen/training_data.py',ROOT/'scripts/p2_listen.py']
            files += [ROOT/'src/listen_benchmark'/n for n in ['official_audio_gate.py','official_event_alignment.py','chinese_asr_filter.py']]
            manifest={str(x.relative_to(ROOT)):sha(x) for x in files};atomic_json(run/'code_manifest.json',manifest,immutable=True)
            with tarfile.open(run/'code_snapshot.tar.gz','w:gz') as tar:
                for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)))
            print(json.dumps(dict(run=str(run),files=len(files))))
        elif a.command=='worker':
            verify_code(run)
            from listen_audio_postprocess.bulk import cache_worker
            cache_worker(run,a.rank)
        elif a.command=='node':print(json.dumps(node_worker(run,a.node)))
        elif a.command=='audit':
            from listen_audio_postprocess.bulk import audit_all
            print(json.dumps(audit_all(run)))
        else:print(json.dumps(coordinator(run)))
    except Exception:
        atomic_json(run/('nodes/'+a.node+'.error.json' if a.command=='node' else 'error-'+a.command+('-'+str(a.rank) if a.rank is not None else '')+'.json'),dict(error=traceback.format_exc(),time=time.time()))
        raise
if __name__=='__main__':main()
