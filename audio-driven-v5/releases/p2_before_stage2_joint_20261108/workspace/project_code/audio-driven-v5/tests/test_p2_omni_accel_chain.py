from pathlib import Path
import copy,json,sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import p2_omni_accel_chain as chain
import p2_omni_accel_parallel as parallel

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))

def jobs():return {loss:dict(status='running',slots=[i],observed_step=i*10) for i,loss in enumerate(chain.LOSSES)}

def test_no_handoff_until_every_full_stage1(tmp_path):
    for i,loss in enumerate(chain.LOSSES):
        write(tmp_path/'configs'/f'guided_{loss}_s1.json',dict(training=dict(steps=1249)))
    for i,loss in enumerate(chain.LOSSES):
        assert chain.finished_parents(tmp_path) is None
        write(tmp_path/'parallel_runs'/f'guided_{loss}_s1'/'w24_from_000000/step_001249.complete.json',dict(step=1249,sha256=loss))
    found=chain.finished_parents(tmp_path)
    assert set(found)==set(chain.LOSSES)
    assert all(item['checkpoint'].endswith('step_001249.pt') for item in found.values())

def test_ambiguous_parent_rejected(tmp_path):
    for loss in chain.LOSSES:
        write(tmp_path/'configs'/f'guided_{loss}_s1.json',dict(training=dict(steps=1249)))
        write(tmp_path/'parallel_runs'/f'guided_{loss}_s1'/'a/step_001249.complete.json',dict(sha256='x'))
    write(tmp_path/'parallel_runs/guided_av_s1/b/step_001249.complete.json',dict(sha256='x'))
    with pytest.raises(ValueError,match='Ambiguous'):chain.finished_parents(tmp_path)

@pytest.mark.parametrize('slots,world',[([0],6),([1,3],12),([0,1,2,3],24)])
def test_commands_preserve_parent_config_and_duration_entry(slots,world):
    cluster=dict(nodes=[dict(host=str(i)) for i in range(3)],python='python',network={})
    cc,rows=parallel.commands(cluster,slots,'config.json',parent='parent.pt',stop_after=16)
    assert len(rows)==3
    for command in rows:
        assert '--nproc_per_node='+str(world//3) in command
        assert str(ROOT/'scripts/p2_duration_train.py') in command
        assert command[command.index('--parent')+1]=='parent.pt'
        assert command[command.index('--config')+1]=='config.json'
        assert command[-2:]==['--stop-after','16']
    assert all('gpus' not in node for node in cluster['nodes'])

def test_resume_uses_checkpoint_not_parent():
    c=dict(nodes=[dict(host=str(i)) for i in range(3)],python='p',network={})
    _,commands=parallel.commands(c,[0],'c',resume='resume.pt',parent='parent.pt')
    assert all('--resume' in cmd and '--parent' not in cmd for cmd in commands)

def test_expansion_never_overlaps_or_creates_18_ranks():
    state=jobs();assert parallel.expansion(state,set(state)) is None
    state['av']['status']='complete';loss,slots=parallel.expansion(state,set(state)-{'av'})
    assert loss=='av_event' and slots==[0,1]
    state[loss]['expanding_to']=slots
    assert parallel.expansion(state,set(state)-{'av'}) is None
    state[loss]['slots']=state[loss].pop('expanding_to')
    state['av_feature']['status']='complete'
    loss2,slots2=parallel.expansion(state,{'av_event','av_event_feature'})
    assert loss2=='av_event_feature' and slots2==[2,3]
    state['av_event']['status']='complete';state['av_event_feature']['slots']=slots2
    assert parallel.expansion(state,{'av_event_feature'})==('av_event_feature',[0,1,2,3])

def test_direct_six_to_twentyfour_when_only_one_survivor():
    state=jobs()
    for loss in list(state)[:3]:state[loss]['status']='complete'
    assert parallel.expansion(state,{'av_event_feature'})==('av_event_feature',[0,1,2,3])

def test_decision_compares_four_arm_makespan_not_individual_step():
    serial={k:8. for k in chain.LOSSES};concurrent={k:20. for k in chain.LOSSES}
    equal={k:dict(model=True,optimizer=True) for k in chain.LOSSES}
    d=chain.decision(serial,concurrent,equal,dict(model=True,optimizer=True))
    assert d['policy']=='parallel6' and d['projected_time_reduction_fraction']==.375
    concurrent['av_event_feature']=34.
    assert chain.decision(serial,concurrent,equal,dict(model=True))['policy']=='serial24'
    concurrent['av_event_feature']=20.;equal['av']['optimizer']=False
    assert chain.decision(serial,concurrent,equal,dict(model=True))['policy']=='serial24'
    equal['av']['optimizer']=True
    assert chain.decision(serial,concurrent,equal,dict(model=False))['policy']=='serial24'

def test_historical_probe_checkpoint_returned_after_extension(tmp_path,monkeypatch):
    out=tmp_path/'run';write(out/'step_000016.complete.json',dict(step=16));write(tmp_path/'cluster.json',{})
    cfg=tmp_path/'cfg.json';write(cfg,dict(output=str(out)))
    monkeypatch.setattr(chain,'probe_config',lambda *args:cfg)
    monkeypatch.setattr(chain,'latest',lambda p:(out/'step_000018.pt',18))
    assert chain.probe(tmp_path,'av','serial24',[0,1,2,3],16,1)==out/'step_000016.pt'

def test_bitwise_check_includes_dtype_and_signed_zero():
    import torch
    assert chain.same(torch.tensor([1.]),torch.tensor([1.]))
    assert not chain.same(torch.tensor([1.]),torch.tensor([1],dtype=torch.int64))
    assert not chain.same(torch.tensor([0.]),torch.tensor([-0.]))

@pytest.mark.parametrize('restart_drain',[False,True])
def test_scheduler_completes_or_restores_serial_after_interrupted_drain(tmp_path,monkeypatch,restart_drain):
    from types import SimpleNamespace
    import time as real_time
    q=tmp_path/'training_queue';state={};launched=[];cp_steps={}
    cluster=dict(nodes=[dict(host=str(i)) for i in range(3)],python='p',network={},minimum_free_gib=135)
    write(tmp_path/'cluster.json',cluster);write(q/'policy.json',dict(policy='parallel6'))
    for i,loss in enumerate(chain.LOSSES):
        c=dict(output=str(tmp_path/'runs'/loss),training=dict(steps=4),duration_sampling=dict(policy='duration384'))
        path=tmp_path/'configs'/f'guided_{loss}_s2.json';write(path,c)
        state[loss]=dict(stage=2,status='pending',slots=[i],parent=loss+'_parent.pt',config=str(path),failures=0,start_step=0)
    write(q/'jobs.json',state)
    if restart_drain:write(q/'fallback_requested.json',dict(error='simulated prior controller failure'))
    def fake_prepare(root,loss,stage,slots,source=None):
        c=json.loads((root/'configs'/f'guided_{loss}_s2.json').read_text());c['output']=str(root/'runs'/f'{loss}_w{len(slots)*6}')
        path=root/'prepared'/f'{loss}.json';write(path,c);return str(path)
    def fake_launch(cc,cmds,logs,outputs,cancel=None,stop_file=None):
        c=json.loads(Path(cmds[0][cmds[0].index('--config')+1]).read_text());out=Path(c['output'])
        world=int(next(x.split('=')[1] for x in cmds[0] if x.startswith('--nproc_per_node=')))*3
        launched.append(world);cp_steps[str(out)]=(out/'step_000004.pt',4)
        write(out/'status.json',dict(step=4))
    monkeypatch.setattr(parallel,'install_backend',lambda:None)
    monkeypatch.setattr(parallel,'verify',lambda root:None)
    monkeypatch.setattr(parallel,'resources',lambda cluster:[dict(gpus=[[i,140*1024,0] for i in range(8)]) for _ in range(3)])
    monkeypatch.setattr(parallel,'latest',lambda out:cp_steps.get(str(out),(None,0)))
    monkeypatch.setattr(parallel.native,'prepare_run',fake_prepare)
    monkeypatch.setattr(parallel,'launch_checked',fake_launch)
    monkeypatch.setattr(parallel,'time',SimpleNamespace(time=real_time.time,time_ns=real_time.time_ns,sleep=lambda seconds:real_time.sleep(.001)))
    parallel.run(tmp_path)
    final=json.loads((q/'state.json').read_text())
    assert final['status']=='complete' and len(launched)==4
    assert launched==([24]*4 if restart_drain else [6]*4)
    assert final['policy']==('serial24' if restart_drain else 'parallel6')
