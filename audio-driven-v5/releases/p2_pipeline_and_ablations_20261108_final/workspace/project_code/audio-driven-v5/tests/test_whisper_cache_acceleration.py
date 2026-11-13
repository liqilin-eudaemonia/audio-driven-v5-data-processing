"""Bounded CPU/GPU overlap, actual Whisper batch checks, and publication gates."""
from dataclasses import replace
import math
from pathlib import Path
import threading
from unittest.mock import patch

import numpy as np
import pytest
import torch

from p2_whisper_av import cache
from p2_whisper_av.features import (SpeakerFeatures,FrozenWhisper,configure_encoder_numerics,
                                   validate_encoder_batching)
from p2_listen.cache import load_part
from p2_listen.contracts import atomic_json,read


def features(n=321,dim=4):
    starts=torch.arange(math.ceil(n/320))*320
    return SpeakerFeatures(torch.zeros(1,len(starts),dim),torch.ones(1,len(starts),dtype=torch.bool),
        torch.stack((starts,(starts+320).clamp(max=n)),-1).double().div(16000).float()[None,None])


def test_next_audio_batch_starts_before_current_batch_is_consumed():
    class Reader:
        def item(self,sid):return {'sample_id':sid}
        def load(self,sid,**kw):return None
    started=threading.Event();release=threading.Event();seen=[]
    def wave(item):
        sid=item['sample_id'];seen.append(sid)
        if sid in ('3','4'):
            started.set()
            if not release.wait(3):raise TimeoutError('Test failed to release next batch')
        return np.zeros(321,dtype=np.float32)
    with patch.object(cache,'speaker_wave',wave):
        stream=cache.prefetched_audio_batches(Reader(),['1','2','3','4','5','6'],2,4)
        try:
            first,_=next(stream)
            assert [r['sid'] for r in first]==['1','2']
            assert started.wait(2)
            assert not {'5','6'} & set(seen)  # Only one future batch, not an unbounded queue.
        finally:
            release.set();stream.close()


def test_prefetch_propagates_stop_and_read_failure():
    class Reader:
        def item(self,sid):raise OSError('audio source unavailable')
        def load(self,*args,**kwargs):return None
    with pytest.raises(InterruptedError):
        list(cache.prefetched_audio_batches(Reader(),['x'],1,stopped=lambda:True))
    with pytest.raises(OSError,match='audio source unavailable'):
        list(cache.prefetched_audio_batches(Reader(),['x'],1))


def test_explicit_numerics_without_cuda_initialization():
    before=torch.cuda.is_initialized()
    result=configure_encoder_numerics()
    assert result['encoder_dtype']=='float32' and not result['autocast']
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision()=='highest'
    assert torch.cuda.is_initialized()==before


def test_real_tiny_whisper_short_and_cross_window_validation():
    from transformers import WhisperConfig,WhisperModel,WhisperFeatureExtractor
    encoder=FrozenWhisper.__new__(FrozenWhisper)
    encoder.encoder=WhisperModel(WhisperConfig(d_model=16,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32)).encoder.eval().requires_grad_(False)
    encoder.processor=WhisperFeatureExtractor(feature_size=80)
    encoder.device='cpu';encoder.dtype=torch.float32
    result=validate_encoder_batching(encoder,batch_size=4)
    assert result['passed'] and result['full_batch_exercised']
    assert result['reference_lengths_samples'][1]>26*16000
    assert result['numerics']['matmul_allow_tf32'] is False


class FakeEncoder:
    device='cpu';dtype=torch.float32
    def encode(self,wave):return features(len(wave))
    def encode_batch(self,waves,**kwargs):return [self.encode(wave) for wave in waves]


def setup_reader(tmp_path):
    atomic_json(tmp_path/'protocol.json',{'test':'speaker'})
    class Reader:
        work=tmp_path
        protocol={'sample_ids':['one','two','three'],'encoder':{'input_dim':4}}
        ids={'one','two','three'}
        def __init__(self,*args,**kwargs):pass
        def item(self,sid):return {'sample_id':sid}
        def folder(self,sid):return self.work/'cache'/sid
        def identity(self,item):return {'sample_id':item['sample_id'],'audio':'speaker'}
        def load(self,sid,**kwargs):
            value=load_part(self.folder(sid),self.identity(self.item(sid)),missing_ok=kwargs.get('missing_ok',False))
            return None if value is None else cache.validate_features(value,4)
    return Reader


def test_extraction_publishes_verified_batches_and_resumes_without_encoder(tmp_path):
    Reader=setup_reader(tmp_path)
    c={'whisper_cache':str(tmp_path),'j_cache':'j','whisper':{'model_path':'model'}}
    with patch.object(cache,'Reader',Reader),patch.object(cache,'model_contract',return_value={'input_dim':4}),\
         patch.object(cache,'speaker_wave',return_value=np.zeros(321,dtype=np.float32)),\
         patch.object(cache,'FrozenWhisper',return_value=FakeEncoder()) as loader:
        result=cache.extract(c,batch_size=2,read_workers=4)
        assert result['new']==3 and result['finished'] and result['write_workers']==1
        assert result['numerics']['matmul_allow_tf32'] is False
        assert read(tmp_path/'validation/rank000.json')['passed']
        assert loader.call_count==1
        result=cache.extract(c,batch_size=2,read_workers=4)
        assert result['new']==0 and result['reused']==3 and loader.call_count==1
    assert all(Reader().load(sid).hidden.dtype==torch.float32 for sid in Reader.ids)


def test_failed_startup_validation_publishes_no_production_feature(tmp_path):
    Reader=setup_reader(tmp_path)
    class Broken(FakeEncoder):
        def encode_batch(self,waves,**kwargs):
            values=super().encode_batch(waves,**kwargs)
            return [replace(x,hidden=x.hidden+1) for x in values]
    c={'whisper_cache':str(tmp_path),'j_cache':'j','whisper':{'model_path':'model'}}
    with patch.object(cache,'Reader',Reader),patch.object(cache,'model_contract',return_value={'input_dim':4}),\
         patch.object(cache,'speaker_wave',return_value=np.zeros(321,dtype=np.float32)),\
         patch.object(cache,'FrozenWhisper',return_value=Broken()):
        with pytest.raises(AssertionError):
            cache.extract(c,batch_size=2,read_workers=4)
    assert not list((tmp_path/'cache').glob('*/payload.pt'))
    assert read(tmp_path/'validation/rank000.json')['passed'] is False
    assert read(tmp_path/'progress/rank000.json')['finished'] is False


def test_gpu_batch_can_run_while_previous_batch_waits_for_disk(tmp_path):
    Reader=setup_reader(tmp_path);writer_started=threading.Event();next_encode=threading.Event()
    class Encoder(FakeEncoder):
        production_calls=0
        def encode_batch(self,waves,**kwargs):
            if all(len(wave)==321 for wave in waves):
                self.production_calls+=1
                if self.production_calls==2:
                    assert writer_started.wait(2)
                    next_encode.set()
            return super().encode_batch(waves,**kwargs)
    original=cache.publish_batch
    def writer(*args):
        writer_started.set()
        assert next_encode.wait(2)  # Would deadlock if the main thread waited before encoding batch 2.
        return original(*args)
    c={'whisper_cache':str(tmp_path),'j_cache':'j','whisper':{'model_path':'model'}}
    with patch.object(cache,'Reader',Reader),patch.object(cache,'model_contract',return_value={'input_dim':4}),\
         patch.object(cache,'speaker_wave',return_value=np.zeros(321,dtype=np.float32)),\
         patch.object(cache,'FrozenWhisper',return_value=Encoder()),patch.object(cache,'publish_batch',writer):
        assert cache.extract(c,batch_size=2)['new']==3


def test_background_writer_error_is_not_reported_as_finished(tmp_path):
    Reader=setup_reader(tmp_path)
    c={'whisper_cache':str(tmp_path),'j_cache':'j','whisper':{'model_path':'model'}}
    with patch.object(cache,'Reader',Reader),patch.object(cache,'model_contract',return_value={'input_dim':4}),\
         patch.object(cache,'speaker_wave',return_value=np.zeros(321,dtype=np.float32)),\
         patch.object(cache,'FrozenWhisper',return_value=FakeEncoder()),\
         patch.object(cache,'publish_batch',side_effect=OSError('disk write failed')):
        with pytest.raises(OSError,match='disk write failed'):
            cache.extract(c,batch_size=2)
    assert read(tmp_path/'progress/rank000.json')['finished'] is False
