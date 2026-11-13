"""Roll out a measured CPU-preparation profile, one GPU per host at a time."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from listen_benchmark.j_annotation import read,write
from listen_reaction_cluster_control import remote


def card_ready(record, now):
    return (record.get('prep_workers')==6
            and record.get('stage') not in ('loading_model','stopped','restarting')
            and record.get('completed_clips',0)>=5
            and now-record.get('heartbeat',0)<=45)


def reload_code(work,node,gpu):
    return f'''
import os,sys,signal,json
from pathlib import Path
sys.path[:0]=[{str(ROOT/'scripts')!r},{str(ROOT/'src')!r}]
import listen_j_reaction_vocal_omni as m
w=Path({str(work)!r});node={node!r};gpu={gpu!r}
rec=m.read(w/'workers'/f'{{node}}-gpu-{{gpu}}.json')
sup=m.read(w/'nodes'/node/'supervisor.json');child=sup['children'][f'gpu-{{gpu}}']
if rec.get('prep_workers')==6:
 print(json.dumps(dict(already_applied=True,pid=rec['pid'])))
elif rec.get('pid')!=child['pid']:
 raise RuntimeError('Worker identity mismatch; leave the process running')
elif m.alive(child):
 os.kill(child['pid'],signal.SIGTERM)
 print(json.dumps(dict(planned_reload=True,pid=child['pid'])))
else:
 print(json.dumps(dict(awaiting_supervisor=True)))
'''


def run(work):
    work=Path(work).resolve();folder=work/'cluster_control/prep6_rollout';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'rollout.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=read(work/'config.json')
        state={'pid':__import__('os').getpid(),'started_at':time.time(),'waves':[],'status':'running'}
        def status():
            state['heartbeat']=time.time();write(folder/'state.json',state)
        status()
        for gpu in range(8):
            if (folder/'STOP').exists():
                state['status']='stopped';status();return
            cards=[]
            for node in cfg['nodes']:
                rec=read(work/'workers'/f'{node}-gpu-{gpu}.json')
                if card_ready(rec,time.time()):continue
                path=work/'nodes'/node/'caption_overrides.json'
                overrides=read(path) if path.exists() else {}
                overrides[str(gpu)]={'workers':6,'prefetch':12,'reason':'measured_late_decode_supply_20261104'}
                write(path,overrides)
                cards.append((node,gpu))
            wave={'gpu':gpu,'cards':cards,'started_at':time.time()};state['waves'].append(wave);status()
            def start(card):
                node,g=card
                try:return node,remote(cfg['nodes'][node],reload_code(work,node,g),40)
                except Exception as exc:return node,{'error':repr(exc)}
            with ThreadPoolExecutor(max_workers=4) as pool:wave['actions']=dict(pool.map(start,cards))
            if any('error' in r for r in wave['actions'].values()):
                state['status']='needs_observation';status();return
            deadline=time.time()+600
            while True:
                pending=[]
                for node,g in cards:
                    r=read(work/'workers'/f'{node}-gpu-{g}.json')
                    if not card_ready(r,time.time()):
                        pending.append({'node':node,'gpu':g,'stage':r.get('stage'),'pid':r.get('pid')})
                wave['pending']=pending;status()
                if not pending:break
                if time.time()>deadline or (folder/'STOP').exists():
                    # Observation expiry never authorizes another kill/restart.
                    state['status']='needs_observation';status();return
                time.sleep(10)
            wave['ready_at']=time.time();status()
            print(json.dumps({'wave_ready':gpu,'cards':cards}),flush=True)
        state['status']='completed';state['completed_at']=time.time();status()


if __name__=='__main__':run(sys.argv[1])
