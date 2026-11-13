import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('combined_train31',ROOT/'scripts/p2_whisper_event_feature_train31.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def cluster():
    c=json.loads((ROOT/'configs/p2_j_qwen_whisper_event_feature_32gpu_20261105.json').read_text())
    c['nodes'][-1]['gpus']=list(range(7));return c


def test_rank_layout_exactly_covers31_without_touching_reserved_gpu():
    rows=m.layout(cluster())
    assert [r for n in rows for r in n['ranks']]==list(range(31))
    assert rows[-1]['ranks']==list(range(24,31)) and rows[-1]['gpus']==list(range(7))


@pytest.mark.parametrize('index',[0,1,2,3])
def test_wrong_card_layout_is_rejected(index):
    c=cluster();c['nodes'][index]['gpus']=[0]
    with pytest.raises(ValueError):m.layout(c)


def test_commands_are_explicit_rank_supervisors_not_uniform_torchrun():
    commands=m.commands(cluster(),ROOT/'configs/example.json')
    assert len(commands)==4
    for i,cmd in enumerate(commands):
        assert 'CUDA_VISIBLE_DEVICES='+','.join(map(str,range(7 if i==3 else 8))) in cmd
        assert cmd[cmd.index('--node-index')+1]==str(i)
        assert 'torch.distributed.run' not in cmd


def test_loss_and_all_five_gradient_groups_required():
    row=dict(av=1.,event=2.,feature_video=3.,feature_audio=4.,weighted_total=1.275,
             grad_norm={k:.1 for k in ['connector','events','features','text_lora','whisper']})
    m.verify_row(row)
    row['grad_norm']['whisper']=0
    with pytest.raises(ValueError):m.verify_row(row)


def test_reserved_busy_gpu_does_not_block_selected31(monkeypatch):
    import sys
    from types import SimpleNamespace
    c=cluster();nodes=[]
    for node in c['nodes']:
        nodes.append(dict(node=node['name'],gpus=[dict(gpu=i,used_mib=100000 if node['name']=='node8' and i==7 else 1,util=100 if node['name']=='node8' and i==7 else 0) for i in range(8)]))
    monkeypatch.setitem(sys.modules,'p2_whisper_event_feature_test32',SimpleNamespace(resources=lambda _:dict(nodes=nodes)))
    assert m.selected_resources(c)['selected_all_idle']
