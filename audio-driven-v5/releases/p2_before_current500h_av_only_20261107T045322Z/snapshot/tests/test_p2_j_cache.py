from pathlib import Path
import time

import numpy as np
import pytest


def test_prefetch_does_not_decode_the_whole_training_set(monkeypatch):
    from p2_j_cache import runtime as rt
    seen=[]
    monkeypatch.setattr(rt,'sample',lambda work,m,sid:{'sample_id':sid})
    def read(item,work):
        seen.append(item['sample_id'])
        return item['sample_id']
    stream=rt.prefetched(Path('/unused'),{},list(range(1000)),read)
    try:
        assert next(stream)[1]==0
        time.sleep(.05)
        assert len(seen)<=2
        assert next(stream)[1]==1
        time.sleep(.05)
        assert len(seen)<=3
    finally:
        stream.close()


def test_qwen_input_reader_never_reads_listener_waveform(tmp_path,monkeypatch):
    from p2_j_cache import data
    pixels=np.zeros((1,512,512,3),dtype=np.uint8)
    speaker=np.ones((16000,1),dtype=np.float32)
    monkeypatch.setattr(data,'read_audio',lambda *a:pytest.fail('Listener target entered Qwen input reader'))
    monkeypatch.setattr(data,'read_speaker',lambda item:(speaker,16000))
    def frames(source,**kwargs):
        assert kwargs['limit']==1
        return pixels,[2.]
    monkeypatch.setattr(data,'decode_frames',frames)
    item=dict(sample_id='input_only',source={},source_files={})
    result=data.read_input(item,tmp_path)
    assert result['speaker'] is speaker
    assert Path(result['reference']).is_file()
    assert not list(tmp_path.rglob('*.wav'))


def test_whole_caption_preserves_language_and_does_not_invent_timing():
    from p2_j_cache.data import caption_prompt
    caption='The listener says “嗯” and smiles.\n'
    prompt=caption_prompt(caption)
    assert prompt.endswith(caption)
    assert 'response schedule' not in prompt and '[0.0' not in prompt
    with pytest.raises(ValueError,match='control tokens'):
        caption_prompt('嗯<|im_end|>')


def test_fractional_48k_endpoint_uses_exact_16k_grid_without_changing_raw_audio():
    import soundfile as sf
    from scipy.signal import resample_poly
    from p2_j_cache.data import observation
    for samples in (96000,96001,96002,138874):
        wave=np.linspace(-.5,.5,samples,dtype=np.float32)[:,None]
        original=wave.copy();duration=samples/48000
        item=dict(sample_id='fractional',source=dict(duration_s=duration))
        result=observation(item,dict(speaker=wave,rate=48000,reference='first.png'))
        actual,rate=sf.read(result['speaker_audio'],dtype='float32')
        assert rate==16000 and len(actual)==round(duration*16000)
        np.testing.assert_array_equal(actual,resample_poly(wave[:,0],1,3)[:len(actual)])
        np.testing.assert_array_equal(wave,original)


def test_audio_revision_can_preserve_only_unaffected_cache_code():
    from p2_j_cache.runtime import cache_identity
    m=dict(schema='test',code={'new':'sha2'},cache_code={'text':{'old':'sha1'}},prompts={})
    assert cache_identity(m,None,'text',{})['code']=={'old':'sha1'}
    item=dict(sample_id='s',source={},speaker={})
    m['prompts']['omni_messages']=[]
    assert cache_identity(m,item,'input',{})['code']=={'new':'sha2'}
    m['sample_cache_code']={'input':dict(code={'previous':'sha0'},sample_ids={'s':True})}
    assert cache_identity(m,item,'input',{})['code']=={'previous':'sha0'}
    assert cache_identity(m,dict(item,sample_id='new_sample'),'input',{})['code']=={'new':'sha2'}


def test_nearest_frame_at_eof_keeps_native_gap_limit(monkeypatch):
    import av
    from contextlib import nullcontext
    from types import SimpleNamespace
    from PIL import Image
    from p2_j_cache.data import decode_frames
    import listen_visual_quality.measurement as measurement
    import listen_benchmark.frame_crop as crop
    class Container:
        def __init__(self):
            self.streams=SimpleNamespace(video=[SimpleNamespace(width=2,height=2,average_rate=25,time_base=1/25,codec_context=SimpleNamespace())])
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def seek(self,*args,**kwargs):pass
        def decode(self,**kwargs):
            return iter([SimpleNamespace(pts=i,time_base=1/25) for i in range(3)])
    monkeypatch.setattr(measurement,'SourceBytes',lambda source:nullcontext(None))
    monkeypatch.setattr(av,'open',lambda member:Container())
    monkeypatch.setattr(crop,'crop_original_frame_roi',lambda frame,transform:Image.fromarray(np.full((2,2,3),frame.pts,dtype=np.uint8)))
    source=dict(source={},source_start_s=0,duration_s=.13,video_transform=dict(source_size=[2,2]))
    pixels,times=decode_frames(source)
    assert times==[0,.04,.08,.08]
    np.testing.assert_array_equal(pixels[-1],pixels[-2])
    with pytest.raises(ValueError,match='Incomplete source video'):
        decode_frames(dict(source,duration_s=.21))


def test_reviewed_gap_uses_only_exact_existing_pts(monkeypatch):
    import av
    from contextlib import nullcontext
    from types import SimpleNamespace
    from PIL import Image
    from p2_j_cache.data import decode_frames, SourceVideoGap
    import listen_visual_quality.measurement as measurement
    import listen_benchmark.frame_crop as crop
    class Container:
        def __init__(self):
            self.streams=SimpleNamespace(video=[SimpleNamespace(width=2,height=2,average_rate=30,time_base=1/30,codec_context=SimpleNamespace())])
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def seek(self,*args,**kwargs):pass
        def decode(self,**kwargs):return iter([SimpleNamespace(pts=i,time_base=1/30) for i in [0,1,4,5]])
    monkeypatch.setattr(measurement,'SourceBytes',lambda source:nullcontext(None))
    monkeypatch.setattr(av,'open',lambda member:Container())
    monkeypatch.setattr(crop,'crop_original_frame_roi',lambda f,t:Image.fromarray(np.full((2,2,3),f.pts,dtype=np.uint8)))
    source=dict(source={},source_start_s=0,duration_s=.16,video_transform=dict(source_size=[2,2]))
    with pytest.raises(SourceVideoGap):decode_frames(source)
    override={'2':dict(target_pts_s=.08,previous_pts_s=1/30,next_pts_s=4/30,selected_pts_s=1/30)}
    pixels,times=decode_frames(source,gap_overrides=override)
    assert times==[0,1/30,1/30,4/30]
    np.testing.assert_array_equal(pixels[2],pixels[1])
    bad={'2':dict(override['2'],next_pts_s=.15)}
    with pytest.raises(SourceVideoGap):decode_frames(source,gap_overrides=bad)
    # An override cannot relax EOF coverage or a different target index.
    with pytest.raises(ValueError,match='Incomplete'):decode_frames(dict(source,duration_s=.28),gap_overrides=override)
    with pytest.raises(SourceVideoGap):decode_frames(source,gap_overrides={'3':override['2']})


def test_gap_policy_changes_only_affected_vae_identity():
    from p2_j_cache.runtime import cache_identity
    item=dict(sample_id='s',source={},speaker={},audio_recipe={})
    m=dict(schema='test',code={},prompts={'omni_messages':[]})
    before=cache_identity(m,item,'vae',{})
    m['video_sampling_overrides']={'s':{'frames':'reviewed'}}
    assert cache_identity(m,item,'vae',{})==dict(before,video_sampling_override={'frames':'reviewed'})
    assert 'video_sampling_override' not in cache_identity(m,item,'input',{})
    assert cache_identity(m,dict(item,sample_id='other'),'vae',{})==dict(before,sample_id='other')


def test_worker_quarantines_bad_job_and_completes_next_without_reload(tmp_path,monkeypatch):
    from p2_j_cache import cluster as c
    from p2_j_cache.data import SourceVideoGap
    from p2_listen.contracts import atomic_json,read
    for name,items in [('000000',['bad','prefetched']),('000001',['good'])]:
        atomic_json(tmp_path/'jobs/features'/(name+'.json'),items)
    atomic_json(tmp_path/'jobs/features.json',['000000','000001'])
    monkeypatch.setattr(c,'open_manifest',lambda w:{'config':{}})
    monkeypatch.setattr(c,'activate',lambda x:None)
    monkeypatch.setattr(c,'extractor_config',lambda x:None)
    monkeypatch.setattr(c,'dependencies',lambda *a:{})
    monkeypatch.setattr(c,'weights_identity',lambda *a:{})
    monkeypatch.setattr(c,'sample',lambda w,m,sid:{'sample_id':sid})
    monkeypatch.setattr(c,'cache_identity',lambda *a:{})
    written=[]
    monkeypatch.setattr(c,'persist',lambda w,p,sid,i,v:written.append(sid))
    for name in ('set_device','reset_peak_memory_stats'):
        monkeypatch.setattr(c.torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(c.torch.cuda,'max_memory_reserved',lambda:0)
    loaded=[]
    class Session:
        def __init__(self,*args):loaded.append(1)
        def prepare(self,item,work):
            if item['sample_id']=='bad':raise SourceVideoGap('bad: Source video gap')
            return item
        def encode(self,item,prepared):return {'vae':{}},{}
    monkeypatch.setattr(c,'FeatureSession',Session)
    c.cluster_worker(tmp_path,'features','test',0,prefetch=2)
    assert loaded==[1] and written==['good']
    assert not (tmp_path/'state/features/000000.json').exists()
    assert read(tmp_path/'quarantine/features/000000.json')['sample_id']=='bad'
    assert read(tmp_path/'state/features/000001.json')['success']
    assert read(tmp_path/'workers/test.json')['status']=='finished'
