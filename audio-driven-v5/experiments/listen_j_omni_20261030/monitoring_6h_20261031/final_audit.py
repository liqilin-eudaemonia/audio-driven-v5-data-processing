"""Six-hour four-node completion audit. Never closes the goal on partial time."""
import hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
FOLDER=Path(__file__).resolve().parent;W=FOLDER.parent;ROOT=W.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from listen_benchmark.j_annotation import read,write
from listen_j_omni_watch_window import probe


def main():
    plan=read(FOLDER/'plan.json')
    if time.time()<plan['end']:raise RuntimeError('Six-hour authorization window has not elapsed; completion is unproven')
    terminal=read(FOLDER/'complete.json');state=read(FOLDER/'state.json');cfg=read(W/'config.json')
    assert terminal['end']==plan['end'] and terminal['at']>=plan['end'] and state['stage']=='complete'
    slots=list(range(int((plan['end']-plan['start'])//1800)+1))
    assert terminal['report_slots']==slots and state['reported_slots']==slots
    reviews=[]
    for i in slots:
        p=FOLDER/f'report-{i:02d}.json';report=read(p);review=read(FOLDER/f'review-{i:02d}.json')
        assert review['report_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
        assert set(report['nodes'])==set(cfg['nodes'])
        reviews.append(dict(slot=i,at=report['at'],reviewed_at=review['reviewed_at']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={n:pool.submit(probe,W,n,c['host']) for n,c in cfg['nodes'].items()};nodes={n:f.result() for n,f in fs.items()}
    assert len(nodes)==4 and sum(len(n['workers']) for n in nodes.values())==32
    for n,d in nodes.items():
        assert not d.get('observation_error') and not d.get('supervisor_error') and not d.get('blocked_gpus')
        assert d['watchdog_alive'] and d['supervisor_alive']
        for worker in d['workers']:
            assert worker['alive'] and worker['matching_heartbeat'] and not worker['last_error']
            assert worker['heartbeat_age_s']<60 and 'visual-gate-v1' in worker['execution_revision']
        if cfg['nodes'][n].get('visual_quality_workers'):
            q=d['visual_quality'];assert q['watchdog_alive'] and q['controller_alive']
            assert not q['state']['failed_sources'] and q['state']['execution_errors_this_process']==0
    caption=read(W/'queues/caption/state.json');quality=read(W/'visual_quality/queues/measure/state.json')
    assert not caption['failed'] and not quality['failed']
    nod=[]
    for n,nc in cfg['nodes'].items():
        if not nc.get('old_work'):continue
        old=Path(nc['old_work']);c=read(old/'config.json');q=read(old/'queues/nod/state.json')
        assert not q['failed']
        for i in range(c['nod_workers']):
            h=read(old/'workers'/f'nod-{i}.json');assert h['revision']>=4 and time.time()-h['heartbeat']<60
        nod.append(dict(node=n,workers=c['nod_workers'],completed_clips=q['completed_clips']))
    for rel,expected in cfg['code_hashes'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
    # Require an explicit final optimization decision and all rollout evidence;
    # an inference-only benchmark cannot stand in for production validation.
    decision=read(W/'visual_quality_rollout/optimization_decision.json')
    assert decision['status'] in ('validated_and_deployed','no_further_safe_gain')
    assert decision['evidence'] and all(Path(p).exists() for p in decision['evidence'])
    result=dict(passed=True,at=time.time(),window=plan,report_reviews=reviews,nodes=nodes,nod=nod,caption_resolved=caption['completed_clips'],visual_quality_resolved=quality['completed_clips'],optimization=decision,scope='Six-hour monitoring/optimization finished; the full annotation dataset need not yet be complete; persistent production watchdogs remain active.')
    write(FOLDER/'final_audit.json',result);print(json.dumps(dict(passed=True,caption_resolved=caption['completed_clips'],visual_quality_resolved=quality['completed_clips'],nodes=len(nodes))))

if __name__=='__main__':main()
