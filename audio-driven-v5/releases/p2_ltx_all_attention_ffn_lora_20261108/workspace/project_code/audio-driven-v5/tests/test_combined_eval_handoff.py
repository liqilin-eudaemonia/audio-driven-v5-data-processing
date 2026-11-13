import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p2_combined_whisper_eval as m


def test_negative_condition_comparison_includes_text_and_tensor_values():
    a={'prompt':'','mask':torch.tensor([True]),'nested':{'video':torch.tensor([1.,2.])}}
    b={'prompt':'','mask':torch.tensor([True]),'nested':{'video':torch.tensor([1.,2.])}}
    assert m.same_values(a,b)
    b['prompt']='changed';assert not m.same_values(a,b)
    b['prompt']='';b['nested']['video'][0]=3;assert not m.same_values(a,b)


def test_combined_loader_and_source_routing_without_changing_training_identity():
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_whisper_event_feature_train31 import install
    install(m.read(m.TRAIN/'config.json'))
    from p2_whisper_event_feature.config import source_identity
    from p2_whisper_event_feature.inference import load_model
    from p2_whisper_event_feature.checkpoint import checked
    from p2_whisper_av import inference,checkpoint
    before=source_identity();old=inference.load_model;old_check=checkpoint.checked
    assert before==m.read(m.TRAIN/'run.json')['source']
    with m.combined_runtime():
        assert inference.load_model is load_model
        assert checkpoint.checked is checked
        assert source_identity()==before
    assert inference.load_model is old and checkpoint.checked is old_check
    assert source_identity()==before
