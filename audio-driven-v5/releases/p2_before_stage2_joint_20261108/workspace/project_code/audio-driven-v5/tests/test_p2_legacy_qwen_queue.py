from pathlib import Path
import sys,json
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import p2_legacy_qwen_queue as queue

def test_latest_ignores_partial_and_corrupt(tmp_path,monkeypatch):
    from p2_matrix16 import checkpoint
    for step in (0,100,200):
        (tmp_path/f'step_{step:06d}.pt').write_bytes(b'x')
        (tmp_path/f'step_{step:06d}.complete.json').write_text(json.dumps({'step':step}))
    (tmp_path/'step_000300.pt').write_bytes(b'partial')
    def check(path):
        step=int(path.stem.split('_')[1])
        if step==200:raise ValueError('bad checksum')
        return {'step':step}
    monkeypatch.setattr(checkpoint,'checked',check)
    assert queue.latest(tmp_path)==(tmp_path/'step_000100.pt',100)

def test_empty_checkpoint_directory(tmp_path):
    assert queue.latest(tmp_path)==(None,0)

def test_exact_three_machine_configuration():
    r=Path(__file__).resolve().parents[1]/'experiments/p2_legacy50h_qwen_lora_20261027'
    c=json.loads((r/'cluster.json').read_text())
    commands=queue.distributed_commands(c,Path('/trainer'),['train'])
    assert len(commands)==3
    assert [n['host'] for n in c['nodes']]==['172.29.15.91','172.29.252.11','172.29.225.51']
    for command in commands:
        assert '--nproc_per_node=8' in command
        assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' in command
        assert 'P2_TRAIN_MEMORY_GIB=134' in command
