"""Refresh this production run and assemble all training caches after completion."""
import json,os,sys,time,subprocess,traceback
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as c
from run_glare_cache_cluster import status
CONFIG=ROOT/'experiments/glare_cache_28gpu_20261110/production_gpu_fer_config.json'
cfg=c.config(CONFIG);m=c.load_run(cfg);work=Path(cfg['work'])
with c.locked(work/'finalize.lock'):
    state=dict(pid=os.getpid(),run_id=m['run_id'],started=time.time(),stage='watching')
    try:
        while True:
            s=status(cfg,m);state.update(heartbeat=time.time(),queue=s['queue'])
            if s['stopped']:
                state['stage']='stopped';break
            nodes=list(s['nodes'].values())
            if s['complete'] and all(n['stage']=='complete' for n in nodes):
                state['stage']='assembling';c.save(work/'finalize_status.json',state)
                with (work/'assemble.log').open('a') as log:
                    p=subprocess.run([cfg['python'],str(ROOT/'scripts/run_glare_cache_cluster.py'),
                        '--config',str(CONFIG),'assemble'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state.update(stage='complete' if p.returncode==0 else 'assembly_failed',returncode=p.returncode,finished=time.time());break
            if any(n.get('stage') in ('failed','finished_with_errors','stopped') for n in nodes):
                state['stage']='needs_attention';break
            c.save(work/'finalize_status.json',state);time.sleep(60)
    except Exception:
        state.update(stage='failed',error=traceback.format_exc())
    c.save(work/'finalize_status.json',state)
