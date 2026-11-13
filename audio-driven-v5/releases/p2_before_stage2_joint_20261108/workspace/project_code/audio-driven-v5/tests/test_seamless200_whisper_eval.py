"""CPU-only matched-noise, CFG and source/queue safety regressions."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from test_p2_whisper_av import build,opened
from p2_testsets_eval.whisper_runtime import matched_sample,speaker_wave
from p2_listen.runtime import guided_sample,append_negative
from omni_av_p2.ltx import append_conditions


def test_matched_sampling_zero_branch_matches_baseline_and_open_branch_works():
    _,_,model,bundle,prepared,_,_=build()
    u=model.connector(bundle['condition'])
    negative=bundle['text']['positive']
    options=dict(steps=2,video_cfg=3.,audio_cfg=7.,sigmas=torch.tensor([1.,.5,0.]))
    torch.manual_seed(137)
    reference=guided_sample(model.renderer,append_conditions(prepared,u),append_negative(negative,u),**options)
    torch.manual_seed(137)
    found=matched_sample(model,prepared,u,bundle['whisper'],negative,**options)
    for k in reference:torch.testing.assert_close(reference[k],found[k],rtol=0,atol=0)
    opened(model)
    calls=[]
    original=model.renderer.forward
    def forward(**kw):
        for mod in ('video','audio'):
            v=kw[mod]
            assert v.qwen_context is not None
            assert torch.equal(v.qwen_time_positions,bundle['whisper'].bounds)
        calls.append(kw)
        return original(**kw)
    with patch.object(model.renderer,'forward',side_effect=forward):
        torch.manual_seed(137)
        changed=matched_sample(model,prepared,u,bundle['whisper'],negative,**options)
    assert len(calls)==4 # conditional and unconditional on every Euler step
    assert any(not torch.equal(changed[k],found[k]) for k in changed)
    keep=~prepared.video_loss_mask
    assert torch.equal(changed['video'][keep],prepared.video.latent.float()[keep])


def test_wave_preserves_exact_source_and_silence(tmp_path):
    import soundfile as sf
    from p2_testsets_eval.protocol import save,sha
    source=tmp_path/'speaker.wav'
    wave=np.zeros(48000,np.float32);wave[16000:24000]=.1
    sf.write(source,wave,16000,subtype='FLOAT')
    audit=dict(speaker_audio_path=str(source),speaker_audio_sha256=sha(source),
        speaker_first_sample=8000,speaker_last_sample=40000,source_only=True,
        target_description_in_model_input=False,listener_future_media_in_model_input=False)
    save(tmp_path/'input_audit.json',audit)
    item=dict(geometry=dict(duration_s=2.))
    found,meta=speaker_wave(item,tmp_path)
    assert np.array_equal(found,wave[8000:40000]) and meta['origin_s']==0.
    assert not meta['silence_removed']
    audit['source_only']=False;save(tmp_path/'input_audit.json',audit)
    with pytest.raises(ValueError,match='Source-only'):speaker_wave(item,tmp_path)


def test_resource_gate_honors_only_authorized_cards_and_connection_failure():
    spec=importlib.util.spec_from_file_location('whisper_eval',ROOT/'scripts/p2_seamless200_whisper_eval.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cfg=dict(nodes=[dict(name=f'node{i}',host='localhost',ssh_port=22,ssh_user='root',gpus=list(range(4 if i==3 else 8))) for i in range(4)])
    report='\n'.join(f'{i}, {0 if i<4 else 140000}, 143000, {0 if i<4 else 100}' for i in range(8))
    # node8 has four permitted idle cards; its four busy, unpermitted cards do not block it.
    with patch.object(m.subprocess,'check_output',return_value=report):
        snap=m.idle_snapshot(cfg)
    assert snap['nodes'][3]['idle'] and not snap['all_idle']
    with patch.object(m.subprocess,'check_output',side_effect=OSError('offline')):
        assert not m.idle_snapshot(cfg)['all_idle']
