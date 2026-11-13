import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from listen_benchmark.caption_guard import caption_prep_workers, detect_worker_ceiling, caption_job_eligible
from listen_benchmark.j_omni_reaction_vocal import corrected_reaction_and_nod, ReactionNotReady
from listen_reaction_cluster_control import checkpoint_verified, ensure_code, stop_code


def test_supply_priority_and_preparation_limits():
    assert caption_prep_workers(100, 40, 24999) == 2
    assert caption_prep_workers(5, 100, 50000) == 2
    assert caption_prep_workers(50, 100, 50000) == 3
    assert detect_worker_ceiling(160, 10000, 8) > detect_worker_ceiling(160, 50000, 8)
    assert detect_worker_ceiling(160, 0, 8) < 160


def test_caption_preparation_never_measures_missing_reaction():
    with tempfile.TemporaryDirectory() as tmp:
        cfg={'reaction_input': {'reaction_cache':tmp}}
        with patch.dict('os.environ', {'J_OMNI_CAPTION_NO_MEASURE':'1'}):
            with patch('listen_benchmark.j_omni_reaction_vocal._measure_reaction') as measure:
                with unittest.TestCase().assertRaises(ReactionNotReady):
                    corrected_reaction_and_nod({}, {'sample_id':'missing'}, tmp, tmp, tmp, cfg)
                measure.assert_not_called()


def test_checkpoint_must_have_complete_marker_and_matching_hash():
    import hashlib
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'step_005500.pt'
        assert not checkpoint_verified(path,5500)
        path.write_bytes(b'complete checkpoint')
        assert not checkpoint_verified(path,5500)
        path.with_suffix('.complete.json').write_text(json.dumps({'step':5500,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
        assert checkpoint_verified(path,5500)
        path.write_bytes(b'corrupt')
        with unittest.TestCase().assertRaises(ValueError):checkpoint_verified(path,5500)


def test_remote_commands_compile_and_use_exact_training_config():
    compile(ensure_code(Path('/tmp/work'),'node17',True),'<ensure>','exec')
    code=stop_code(Path('/tmp/only-this-config.json'))
    compile(code,'<stop>','exec')
    assert 'config not in a' in code


def test_finished_pair_without_receipt_is_still_claimable():
    with tempfile.TemporaryDirectory() as tmp:
        job=Path(tmp)/'jobs'/'pair.json';job.parent.mkdir()
        job.write_text(json.dumps({'rows':[{'sample_id':'done'}]}))
        with patch('listen_benchmark.caption_guard.partition_caption_rows',return_value=([],[])):
            assert caption_job_eligible(tmp,'pair')
        with patch('listen_benchmark.caption_guard.partition_caption_rows',return_value=([],[{'sample_id':'waiting'}])):
            assert not caption_job_eligible(tmp,'pair')


def test_writer_only_marks_completion_reader_owns_retirement():
    from listen_j_reaction_vocal_omni import CrossPairPrefetchStream
    stream=object.__new__(CrossPairPrefetchStream)
    stream.active={'pair':{'waiting':{'clip'}}}
    with patch.object(stream,'_retire_finished') as retire:
        stream.mark_done('pair','clip')
        retire.assert_not_called()
    assert stream.active['pair']['waiting']==set()


def test_live_worker_stall_detection_respects_stage_and_supply():
    from listen_j_reaction_vocal_omni import stalled_caption_worker
    child={'pid':123,'started_at':1}
    rec={'pid':123,'stage':'inference','submitted_requests':4,'heartbeat':100}
    assert stalled_caption_worker(child,rec,600,10000) is None
    assert stalled_caption_worker(child,rec,700,10000)['limit_s']==570
    assert stalled_caption_worker(child,rec,10000,0) is None
    assert stalled_caption_worker(child,{'pid':999,'stage':'write','heartbeat':1},500,10000) is None
    assert stalled_caption_worker(child,{'pid':999,'stage':'write','heartbeat':1},10000,10000) is None


def test_supervisor_publishes_identity_before_single_cluster_audit():
    import socket
    import listen_j_reaction_vocal_omni as m
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp);(work/'nodes'/'node').mkdir(parents=True)
        (work/'config.json').write_text(json.dumps({'nodes':{'node':{'hostname':socket.gethostname()}}}))
        (work/'completion_status.json').write_text(json.dumps({'status':'succeeded'}))
        def audit(_work):
            s=json.loads((work/'nodes'/'node'/'supervisor.json').read_text())
            assert s['pid']>0 and s['stage']=='initializing' and s['start_ticks']
            return []
        with patch('listen_benchmark.caption_guard.shared_identity',return_value={'version':1}),patch('listen_benchmark.caption_guard.republish_caption_jobs',side_effect=audit) as republish,patch.object(m,'complete',return_value=True):
            m.supervise(work,'node')
            m.supervise(work,'node')
            assert republish.call_count==1


def test_ssh_handshake_retry_uses_persistent_connection():
    import subprocess
    from listen_reaction_cluster_control import remote
    failed=subprocess.CompletedProcess([],255,stdout='',stderr='kex_exchange_identification: reset')
    good=subprocess.CompletedProcess([],0,stdout='{"ok":true}\n',stderr='')
    node={'host':'192.0.2.123','hostname':'definitely-not-this-host'}
    with patch('listen_reaction_cluster_control.subprocess.run',side_effect=[failed,good]) as run,patch('listen_reaction_cluster_control.time.sleep'):
        assert remote(node,'print(1)')=={'ok':True}
        assert run.call_count==2
        assert 'ControlMaster=auto' in run.call_args.args[0]


def test_detection_scaling_restores_validated_legacy_policy():
    from listen_j_reaction_vocal_omni import detection_worker_target
    assert detection_worker_target(33,160,.55,251,1000,128)==41
    assert detection_worker_target(64,160,.77,251,1000,128)==68
    assert detection_worker_target(80,160,.83,251,1000,128)==35
    assert detection_worker_target(127,160,.5,150,1000,128)==128
    assert detection_worker_target(80,160,.5,150,20,128)==80
