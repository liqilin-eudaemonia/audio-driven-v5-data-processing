"""Successor gating, paired initialization and launch safety, with no GPU use."""
import importlib.util
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import control
spec=importlib.util.spec_from_file_location('e0_node_launcher',HERE/'run_node.py');launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


def test_running_e1_never_passes():
    assert not control.completed_gate(None,[dict(state='running')]*3)
    assert not control.completed_gate(dict(step=1268,budget=1268,stopped=False),[dict(state='running')]*3)


@pytest.mark.parametrize('status', [None,dict(step=100,budget=1268,stopped=True),dict(step=1268,budget=1268,stopped=True)])
def test_all_nodes_exited_without_full_success_refused(status):
    with pytest.raises(RuntimeError):control.completed_gate(status,[dict(state='finished',returncode=0)]*3)


def test_one_failed_node_stops_succession():
    with pytest.raises(RuntimeError):control.completed_gate(None,[dict(state='finished',returncode=0),dict(state='failed',returncode=1),dict(state='running')])


def test_complete_requires_all_three_nodes():
    s=dict(step=1268,budget=1268,stopped=False);n=dict(state='finished',returncode=0)
    assert not control.completed_gate(s,[n,n])
    assert control.completed_gate(s,[n,n,n])


def test_new_e0_uses_identical_config_seed_and_no_resume():
    cmd=launcher.command(2)
    assert cmd[cmd.index('--arm')+1]=='E0'
    assert cmd[cmd.index('--train-seed')+1]=='20261023'
    assert cmd[cmd.index('--config')+1]==str(control.TRAIN/'config.json')
    assert '--resume' not in cmd and '--nproc-per-node=8' in cmd and '--nnodes=3' in cmd


def test_checkpoint_corruption_stops_launch(tmp_path):
    cp=tmp_path/'step_001268.pt';cp.write_bytes(b'checkpoint')
    control.write(cp.with_suffix('.complete.json'),dict(step=1268,sha256=control.sha(cp)))
    assert control.verify_checkpoint(tmp_path)['bytes']==10
    cp.write_bytes(b'changed')
    with pytest.raises(ValueError):control.verify_checkpoint(tmp_path)


def test_pair_identity_must_match_initialization(monkeypatch,tmp_path):
    e1=tmp_path/'E1';e0=tmp_path/'E0';monkeypatch.setattr(control,'E1',e1);monkeypatch.setattr(control,'E0',e0)
    ident=dict(seed=20261023,budget=1268,world=24,initialization='same',observations='data',targets='targets',feature_loss_enabled=False)
    control.write(e1/'run.json',dict(ident,arm='E1'));control.write(e0/'run.json',dict(ident,arm='E0'))
    assert control.verify_pair()['identical_except_arm']
    control.write(e0/'run.json',dict(ident,arm='E0',initialization='different'))
    with pytest.raises(ValueError,match='initialization'):control.verify_pair()
