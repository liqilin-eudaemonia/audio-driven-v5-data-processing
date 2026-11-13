"""End-of-window audit; installing a watcher never counts as nine-hour completion."""
from pathlib import Path
import importlib.util, math, time
G=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chain_final_guard',G/'watch.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def run():
    import torch
    torch.set_num_threads(2)
    protocol=M.read(G/'protocol.json')
    assert time.time()>=protocol['deadline'], 'Nine hours have not elapsed'
    finish=M.read(G/'WINDOW_FINISHED.json')
    assert finish['success'] and finish['elapsed']>=9*3600
    reports=[]
    for hour in range(10):
        value=M.read(G/f'hour{hour:02d}.json')
        assert value['cluster']['observed']
        assert [n['node']for n in value['cluster']['nodes']]==['node12','node8','node4','node11']
        assert [len(n['cards'])for n in value['cluster']['nodes']]==[8,8,8,4]
        assert all({c['index']for c in n['cards']}==set(M.Q.CL['nodes'][i]['gpus'])for i,n in enumerate(value['cluster']['nodes']))
        reports.append(dict(hour=hour,at=value['cluster']['at'],arm=value['health']['arm'],
                            step=value['health']['step'],healthy28=value['cluster']['healthy28']))
    arm=M.phase();active='full'if arm=='complete'else arm
    snap=M.probe(active)
    assert snap['observed']
    if arm=='complete':
        assert not any(n['workers']or n['controllers']for n in snap['nodes'])
        health=dict(arm=arm,step=7000,target=7000)
    else:
        health=M.health(arm)
        assert snap['healthy28'] and not health['flags']
        assert not any(n['foreign']for n in snap['nodes'])
        ratio=health['speed'].get('normalized_ratio')
        assert ratio is None or ratio<=1.3,'Unresolved slowdown requires inspection'
    contract=M.contract(active)
    out=M.info(active)['out']
    marker=sorted(out.glob('step_*.complete.json'),reverse=True)[0]
    path=marker.with_suffix('').with_suffix('.pt')
    if active=='no_whisper':training=M.Q.R.load()
    else:
        spec=importlib.util.spec_from_file_location('full_final_runtime',M.FULL/'run_runtime.py')
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);training=runtime.load()
    cp=training.checkpoint.checked(path)
    assert cp['identity']==M.read(out/'run.json') and len(cp['rank_rng'])==28
    assert cp['optimizer']['state'] and cp['optimizer']['param_groups']
    if active=='full':
        assert all((out/f'step_{s:06d}.complete.json').exists()for s in range(5600,cp['step']+1,100))
    result=dict(success=True,at=time.time(),start=protocol['start'],deadline=protocol['deadline'],
                elapsed=time.time()-protocol['start'],hourly_reports=reports,actual_phase=arm,
                health={k:v for k,v in health.items()if k not in ('owner','last_rows')},cluster=snap,
                contract=contract,latest_complete_checkpoint=str(path),checkpoint_step=cp['step'],
                checkpoint_sha256=M.read(marker)['sha256'],model_adam_28rng_verified=True,
                reserved_gpu='node11 GPU4-7',training_continues=arm!='complete')
    M.atomic_json(G/'FINAL_ACCEPTED.json',result)
    print({k:v for k,v in result.items()if k not in ('cluster','hourly_reports','contract')},flush=True)

if __name__=='__main__':run()
