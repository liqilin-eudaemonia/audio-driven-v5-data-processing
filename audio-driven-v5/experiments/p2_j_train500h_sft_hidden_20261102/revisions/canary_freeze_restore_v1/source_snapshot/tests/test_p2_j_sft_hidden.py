"""Critical continuation and source-preservation checks; CPU only."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch

import pytest
import torch

import p2_j_sft_hidden as h
from p2_listen.contracts import atomic_json,sha,digest


@pytest.fixture
def predecessor(tmp_path):
    root=tmp_path/'teacher';root.mkdir();atomic_json(root/'protocol.json',dict(source='source'))
    return dict(teacher=str(root),samples=32,source=str(tmp_path/'source'),teacher_protocol_sha256=sha(root/'protocol.json'))


def complete(p,**changes):
    row=dict(success=True,samples=p['samples'],source=p['source'],protocol_sha256=p['teacher_protocol_sha256'])
    row.update(changes);atomic_json(Path(p['teacher'])/'COMPLETE.json',row)


def test_predecessor_requires_final_audit(predecessor):
    assert not h.predecessor_ready(predecessor)
    complete(predecessor);assert h.predecessor_ready(predecessor)


@pytest.mark.parametrize('changes',[dict(success=False),dict(samples=31),dict(source='/wrong'),dict(protocol_sha256='old')])
def test_wrong_predecessor_cannot_start(predecessor,changes):
    complete(predecessor,**changes)
    with pytest.raises(ValueError):h.predecessor_ready(predecessor)


def test_modified_teacher_protocol_cannot_start(predecessor):
    complete(predecessor);atomic_json(Path(predecessor['teacher'])/'protocol.json',dict(changed=True))
    with pytest.raises(ValueError):h.predecessor_ready(predecessor)


def test_production_requires_same_protocol_canary(predecessor,tmp_path):
    complete(predecessor);r=SimpleNamespace(protocol=predecessor,work=tmp_path,protocol_sha='new')
    h.require_ready(r,canary=True)
    atomic_json(tmp_path/'CANARY.json',dict(success=True,protocol_sha256='old'))
    with pytest.raises(ValueError):h.require_ready(r)
    atomic_json(tmp_path/'CANARY.json',dict(success=True,protocol_sha256='new'));h.require_ready(r)


def test_identity_binds_adapter_and_prompt():
    r=object.__new__(h.Reader);r.protocol=dict(schema='overlay',adapter=dict(files={'weight':'A'}));r.protocol_sha='P'
    r.source=SimpleNamespace(manifest=dict(entries=dict(s='source-identity')))
    a=r.identity(dict(sample_id='s',prompt='original'))
    b=r.identity(dict(sample_id='s',prompt='changed'))
    assert digest(a)!=digest(b)
    r.protocol=dict(schema='overlay',adapter=dict(files={'weight':'B'}))
    assert digest(a)!=digest(r.identity(dict(sample_id='s',prompt='original')))


def test_completion_order_keeps_slow_and_cached_inputs():
    r=SimpleNamespace(item=lambda sid:dict(sample_id=sid),load=lambda sid,missing_ok: {} if sid=='cached' else None)
    def prepare(item):
        time.sleep(.1 if item['sample_id']=='slow' else .001)
        return item['sample_id']
    with patch.object(h,'prepare_input',prepare),ThreadPoolExecutor(2) as pool:
        rows=list(h.prepared_stream(r,['slow','fast','cached'],pool,3))
    assert {i['sample_id'] for i,v in rows}=={'slow','fast','cached'}
    assert [i['sample_id'] for i,v in rows].index('fast')<[i['sample_id'] for i,v in rows].index('slow')
    assert next(v for i,v in rows if i['sample_id']=='cached') is None


def test_preparation_failure_is_not_dropped():
    r=SimpleNamespace(item=lambda sid:dict(sample_id=sid),load=lambda sid,missing_ok:None)
    with patch.object(h,'prepare_input',side_effect=ValueError('bad crop')),ThreadPoolExecutor(1) as pool:
        with pytest.raises(ValueError,match='bad crop'):list(h.prepared_stream(r,['s'],pool,1))


def test_overlay_assembler_replaces_only_guided_hidden(tmp_path):
    r=object.__new__(h.Reader);r.work=tmp_path;r.protocol=dict(adapter=dict(files={'adapter_model.safetensors':'A'}))
    old_calls=[];new_calls=[]
    r.source=SimpleNamespace(work=tmp_path/'source',manifest={},part=lambda sid,part,device:old_calls.append(part) or 'old-'+part)
    r.load=lambda sid,device:new_calls.append(sid) or 'new-H'
    def assemble(view,sid,variant,device):
        assert variant=='guided'
        return dict(metadata={},H=view.part(sid,'input_plan',device),vae=view.part(sid,'vae',device),text=view.part(sid,'text_guided',device))
    with patch.object(h.CacheReader,'training_sample',assemble):result=r.training_sample('s',include_teacher=False)
    assert result['H']=='new-H' and old_calls==['vae','text_guided'] and new_calls==['s']


def test_resource_probe_refuses_foreign_gpu_process():
    path=Path(__file__).resolve().parents[1]/'scripts/p2_j_sft_hidden.py'
    spec=importlib.util.spec_from_file_location('sft_hidden_cli',path);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    gpu='\n'.join(f'{i}, 140000, 0' for i in range(8))
    with patch.object(cli,'remote',side_effect=[gpu,'12345']):assert not cli.resource_probe({},dict(name='node'))['available']
    with patch.object(cli,'remote',side_effect=[gpu,'']):assert cli.resource_probe({},dict(name='node'))['available']


def test_read_only_frozen_adapter_base_model_access():
    # PEFT's public get_base_model() is needed: .model on the wrapper is the
    # native parent module, not necessarily the text transformer expected by H.
    from peft import LoraConfig,get_peft_model
    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False)])
        def forward(self,x):return self.model.layers[0](x)
    native=Native();base=native(torch.ones(1,2)).detach().clone()
    peft=get_peft_model(native,LoraConfig(r=1,lora_alpha=1,target_modules=['0'],init_lora_weights=False))
    peft.requires_grad_(False).eval();handle=peft.get_base_model()
    assert hasattr(handle.model,'layers') and not any(p.requires_grad for p in handle.parameters())
    assert not torch.equal(base,handle(torch.ones(1,2)))
    with peft.disable_adapter():assert torch.equal(base,handle(torch.ones(1,2)))
