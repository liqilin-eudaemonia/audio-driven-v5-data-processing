import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('semantic_runner',ROOT/'scripts/run_listener_semantic_prompt.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('field,value',[
    ('protocol_sha256','old_protocol'),('prompt','review of an old proposal'),
    ('job_id','another_sample'),('attempt',2),('stage','proposal'),
])
def test_cache_rejects_wrong_identity_without_relabeling(tmp_path,field,value):
    req=dict(job_id='s',prompt='actual review request')
    entry=dict(job_id='s',prompt=req['prompt'],stage='review',attempt=0,protocol_sha256='current',finish_reason='stop',raw='{"accepted":true,"issues":[]}')
    entry[field]=value
    p=tmp_path/'cached.json';p.write_text(json.dumps(entry));before=p.read_bytes()
    with pytest.raises(ValueError,match='identity/request mismatch'):
        runner.cached_answer(p,req,'review',0,'current')
    assert p.read_bytes()==before


def test_exact_cache_can_resume(tmp_path):
    req=dict(job_id='s',prompt='same request')
    p=tmp_path/'cached.json'
    p.write_text(json.dumps(dict(**req,stage='proposal',attempt=0,protocol_sha256='p',finish_reason='stop',raw='{"decisions":[],"conflicts":[]}')))
    assert runner.cached_answer(p,req,'proposal',0,'p')==dict(decisions=[],conflicts=[])


def test_preflight_rejection_never_claims_complete_or_loads_gpu(tmp_path):
    import subprocess
    import sys
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    record=dict(sample_id='s',duration_s=2,events=[dict(event_id='e',modality='visual',start=0,end=1,
        description='The listener smiles, indicating agreement while nodding.')])
    inputs=tmp_path/'input.jsonl';inputs.write_text(json.dumps(dict(job_id='s',record=record,provenance=dict(role='historical_regression_only')))+'\n')
    out=tmp_path/'run'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_listener_semantic_prompt.py'),'--inputs',str(inputs),
        '--output',str(out),'--model',str(model)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((out/'summary.json').read_text())
    assert report['status']=='review_required' and report['accepted']==0
    assert report['rejected'][0]['stage']=='preflight'
    assert not (out/'COMPLETE.json').exists()


def test_cluster_refuses_cross_protocol_response_reuse_before_launch(tmp_path):
    import subprocess
    import sys
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_listener_semantic_prompt_cluster24.py'),
        '--inputs',str(tmp_path/'unused.jsonl'),'--output',str(tmp_path/'out'),
        '--reuse-from',str(tmp_path/'old_run')],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Cross-protocol response reuse is disabled' in result.stderr
    assert not (tmp_path/'out').exists()


def test_cluster_retries_handshake_but_not_running_worker(tmp_path,monkeypatch):
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('cluster_retry',ROOT/'scripts/run_listener_semantic_prompt_cluster24.py')
    cluster=importlib.util.module_from_spec(spec);spec.loader.exec_module(cluster)
    calls=[]
    def fake_run(*args,**kwargs):
        calls.append(1)
        if len(calls)==1:
            kwargs['stdout'].write('kex_exchange_identification: Connection reset by peer\n')
            return SimpleNamespace(returncode=255)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cluster.subprocess,'run',fake_run)
    monkeypatch.setattr(cluster.time,'sleep',lambda _:None)
    out=tmp_path/'worker';log=tmp_path/'log.txt'
    args=(0,'node','host',0,tmp_path/'input.jsonl',out,'model',4,.46,3,log)
    report=cluster.run_shard(*args)
    assert report['connection_returncodes']==[255,0]
    out.mkdir();(out/'protocol.json').write_text('{}');calls.clear()
    report=cluster.run_shard(*args)
    assert report['connection_returncodes']==[255]  # no duplicate live GPU worker
