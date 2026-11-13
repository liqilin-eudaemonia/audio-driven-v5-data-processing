"""No early GPU handoff; preserve review statuses and reject stale/missing audits."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('speaker_whisper_handoff',ROOT/'scripts/p2_speaker_whisper_after_annotation.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
from p2_listen.contracts import atomic_json,read


def fixture(work):
    from listen_benchmark.caption_guard import digest,shared_identity
    atomic_json(work/'producer_complete.json',dict(eof=True))
    (work/'ready.jsonl').write_text('one\ntwo\n')
    state=dict(offset=8,inflight={},deferred={},failed={},completed_clips=2)
    atomic_json(work/'queues/caption/state.json',state)
    atomic_json(work/'queues/detect/state.json',dict(completed_clips=2,inflight={},failed={}))
    counts=dict(pass_=2,generation_failed=0,insufficient_evidence=0,processing_failed=0,not_processed=0)
    counts['pass']=counts.pop('pass_')
    audit=dict(status='succeeded',counts=counts,duplicate_ids=[],missing_ids=[],
        identity_version=digest(shared_identity()),
        audit_queue_identity=digest(dict(state=state,ready_bytes=8)))
    atomic_json(work/'completion_status.json',audit)
    return audit


def test_waits_for_producer_and_audit(tmp_path):
    assert queue.predecessor_ready(tmp_path,2)[0] is False
    fixture(tmp_path)
    (tmp_path/'completion_status.json').unlink()
    ready,reason=queue.predecessor_ready(tmp_path,2)
    assert not ready and reason['reason']=='waiting_fixed_manifest_audit'


def test_ready_requires_fixed_manifest_count_and_current_audit(tmp_path):
    fixture(tmp_path)
    ready,result=queue.predecessor_ready(tmp_path,2)
    assert ready and result['predecessor_status']=='succeeded'
    assert not queue.predecessor_ready(tmp_path,3)[0]
    audit=read(tmp_path/'completion_status.json');audit['identity_version']='obsolete'
    atomic_json(tmp_path/'completion_status.json',audit)
    ready,result=queue.predecessor_ready(tmp_path,2)
    assert not ready and result['reason']=='annotation_audit_stale'


@pytest.mark.parametrize('bucket',['not_processed','processing_failed'])
def test_never_hands_off_missing_or_failed_processing(tmp_path,bucket):
    audit=fixture(tmp_path);audit['counts']['pass']=1;audit['counts'][bucket]=1
    audit['status']='completed_with_failures';atomic_json(tmp_path/'completion_status.json',audit)
    assert not queue.predecessor_ready(tmp_path,2)[0]


def test_preserves_review_flags_and_rejected_evidence_without_promoting_them(tmp_path):
    audit=fixture(tmp_path);audit['counts']['pass']=0
    audit['counts']['generation_failed']=1;audit['counts']['insufficient_evidence']=1
    audit['status']='completed_with_failures';atomic_json(tmp_path/'completion_status.json',audit)
    ready,result=queue.predecessor_ready(tmp_path,2)
    assert ready and result['predecessor_status']=='completed_with_failures'
    assert result['counts']['pass']==0 and result['counts']['generation_failed']==1


def test_does_not_handoff_inflight_or_deferred_queue(tmp_path):
    fixture(tmp_path)
    path=tmp_path/'queues/caption/state.json';state=read(path);state['inflight']={'pair':'worker'}
    atomic_json(path,state)
    assert not queue.predecessor_ready(tmp_path,2)[0]


def test_pid_reuse_or_wrong_command_is_not_adopted():
    record=queue.process_record()
    assert not queue.alive_record(dict(record,start_ticks='wrong'),Path(sys.executable))
    assert not queue.alive_record(record,ROOT/'scripts/p2_speaker_whisper_after_annotation.py','run')


def test_independent_speaker_cache_can_overlap_audit_only_after_all_receipts(tmp_path):
    fixture(tmp_path)
    atomic_json(tmp_path/'producer_complete.json',dict(eof=True,published_clips=2,published_sources=1))
    for branch in ('caption','detect'):
        path=tmp_path/'queues'/branch/'state.json';q=read(path);q['completed']=1;atomic_json(path,q)
    (tmp_path/'completion_status.json').unlink()
    ready,info=queue.predecessor_resource_ready(tmp_path,2,1)
    assert ready and info['cpu_audit_pending'] is True
    assert info['predecessor_status']=='processing_drained_cpu_audit_pending'
    path=tmp_path/'queues/caption/state.json';q=read(path);q['failed']={'source':'failed'};atomic_json(path,q)
    assert not queue.predecessor_resource_ready(tmp_path,2,1)[0]


def test_queued_run_launches_only_speaker_cache_then_full_audit(tmp_path,monkeypatch):
    import listen_cluster
    work=tmp_path/'queue';work.mkdir();predecessor=tmp_path/'upstream';predecessor.mkdir()
    cache=tmp_path/'cache';cache.mkdir();config=tmp_path/'config.json';cluster=tmp_path/'cluster.json'
    manifest=tmp_path/'fixed.jsonl';manifest.write_text('frozen-list\n')
    atomic_json(config,{'cache_only':True});atomic_json(predecessor/'config.json',{'fixed':True})
    atomic_json(cache/'protocol.json',{'speaker':'only'})
    nodes=[dict(name='node'+str(i),host='host'+str(i),gpus=list(range(8))) for i in range(4)]
    atomic_json(cluster,dict(nodes=nodes,python=sys.executable))
    cfg=dict(predecessor=str(predecessor),cache=str(cache),config=str(config),cluster=str(cluster),
        config_sha256=queue.sha(config),cluster_sha256=queue.sha(cluster),cache_code={},
        predecessor_config_sha256=queue.sha(predecessor/'config.json'),predecessor_manifest=str(manifest),
        predecessor_manifest_sha256=queue.sha(manifest),predecessor_clips=2,samples=2,world=32,
        poll_seconds=.01,max_attempts=5,batch_size=16,read_workers=2,cpu_threads=2,master_port=29844,
        minimum_free_bytes=0,stall_seconds=900,startup_seconds=900)
    atomic_json(work/'queue.json',cfg)
    monkeypatch.setattr(queue,'predecessor_ready',lambda *_:(True,{'predecessor_status':'succeeded'}))
    monkeypatch.setattr(queue,'resource_checks',lambda *_:[{'free':True}]*4)
    events=[]
    def execute(command,*args,**kwargs):
        events.append(command[2])
        if command[2]=='finalize':
            atomic_json(cache/'COMPLETE.json',dict(success=True,samples=2,audio_source='speaker',
                protocol_sha256=queue.sha(cache/'protocol.json')))
    def launch(cluster,commands,*args,**kwargs):
        events.append('extract')
        assert len(commands)==4
        assert all('extract' in command and 'train' not in command for command in commands)
    monkeypatch.setattr(queue,'execute',execute);monkeypatch.setattr(listen_cluster,'launch',launch)
    queue.run(work)
    assert events==['prepare','extract','finalize']
    assert read(work/'state.json')['phase']=='complete'
    assert read(work/'state.json')['training_started'] is False
    assert (predecessor/'cluster_control/STOP').exists()
