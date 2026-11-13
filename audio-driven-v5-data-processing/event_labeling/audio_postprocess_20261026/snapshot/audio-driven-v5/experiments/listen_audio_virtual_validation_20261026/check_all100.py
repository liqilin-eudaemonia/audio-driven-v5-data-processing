"""Full 100-case acceptance: physical WAV reference, native VAE, training loader, resume."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.config import load,activate
from p2_listen.contracts import read,sha,atomic_json
from p2_listen.dataset import load_observations,load_targets
from p2_listen.cache import load_part
from p2_listen.training_data import load_training_sample
from p2_listen.annotations import rasterize
from p2_listen import features
from listen_audio_postprocess.overlay import sample_contract,build_one
from listen_audio_postprocess.recipes import read_listener_audio,implementation_identity
from listen_audio_postprocess.encoder import NativeAudioEncoder

RUN=Path(__file__).resolve().parent;VIEW=RUN/'view100';started=time.monotonic()
c=load(VIEW/'config.json');activate(c);torch.set_num_threads(2)
observations=load_observations(c,'train');targets=load_targets(c,observations)
review_path=ROOT/'experiments/listen_official_chinese100_20261026/results.json'
review=read(review_path);preview=ROOT/'data_previews/listen_benchmark_review/audio_official_chinese100'
assert len(observations)==len(review['cases'])==100
assert [o['sample_id'] for o in observations]==[x['sample_id'] for x in review['cases']]
assert len(list((VIEW/'cache/audio_postprocess').glob('*/complete.json')))==100
assert len(list((VIEW/'cache/events').glob('*/complete.json')))==100
base_root=Path(c['data']['audio_postprocess']['base_output'])
base_config=load(base_root/'config.json')
assert (VIEW/'cache/input').resolve()==(base_root/'cache/input').resolve()
assert (VIEW/'cache/text').resolve()==(base_root/'cache/text').resolve()
encoder=NativeAudioEncoder(c,'cpu');reports=[];events_total=0;without_audio=0;silent=0

def eq_tree(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:eq_tree(a[k],b[k])
    else:assert a==b

for index,(obs,case) in enumerate(zip(observations,review['cases'])):
    tick=time.monotonic();sid=obs['sample_id'];target=targets[sid]
    root,folder,expected,recipe,label,identity=sample_contract(c,obs,target)
    audio_folder=VIEW/'cache/audio_postprocess'/sid;marker_hash=sha(audio_folder/'complete.json');payload_hash=sha(audio_folder/'payload.pt')
    base_marker_hash=sha(folder/'complete.json');base_payload_hash=sha(folder/'payload.pt')
    original_label_path=base_root/'labels/canonical'/f'{sid}.json';original_label_hash=sha(original_label_path);original_label=read(original_label_path)
    base=load_part(folder,expected)
    x,layout,teacher,current,event_targets=load_training_sample(c,obs,target,'cpu')
    assert teacher is None
    eq_tree(base['av']['latents'],current['av']['latents']);eq_tree(base['support']['video'],current['support']['video'])
    patch=load_part(audio_folder,identity);assert 'av' not in patch and 'video' not in patch
    virtual,sr,meta=read_listener_audio(recipe)
    physical,physical_sr=sf.read(preview/case['media']['processed'],dtype='float32',always_2d=True)
    assert sr==physical_sr;np.testing.assert_array_equal(virtual,physical)
    assert virtual.shape[0]==recipe['source']['samples']
    physical_encoded=encoder.encode(physical,physical_sr,base['geometry'],base['av']['latents']['latents'].shape)
    reuse_base=bool(patch.get('reuse_base_audio'))
    if reuse_base:
        assert not meta['waveform_changed']
        # The immutable original cache may have been encoded on a different device.
        # Its exact reuse, and equal current-device encoding of the unchanged
        # virtual and physical waves, are separate contracts.
        eq_tree(current,base)
        virtual_encoded=encoder.encode(virtual,sr,base['geometry'],base['av']['latents']['latents'].shape)
        eq_tree(physical_encoded,virtual_encoded)
    else:
        eq_tree(physical_encoded['audio_latents'],current['av']['audio_latents'])
    eq_tree(physical_encoded['audio_support'],current['support']['audio'])
    assert torch.isfinite(current['av']['audio_latents']['latents']).all()
    for k in ['geometry','audio_support_margin_s','audio_endpoint_tokens_removed']:assert physical_encoded[k]==current[k]
    eq_tree(rasterize(label,layout,c['events']['K_train']),event_targets)
    visual=[e for e in label['events'] if e['type']!='listener_vocalization']
    assert visual==[e for e in original_label['events'] if e['type']!='listener_vocalization']
    old_events=load_part(base_root/'cache/events'/sid,features.identity(base_config,'events',obs,label=original_label))
    for k in event_targets:
        if k!='listener_vocalization':eq_tree(event_targets[k],old_events[k])
    expected_events={e['event_id']:e for e in case['aligned_audio_events']}
    actual_ranges={key:[] for key in expected_events}
    for e in label['events']:
        if e['type']!='listener_vocalization':continue
        source_id=e['source']['audio_postprocess']['source_event_id'];source=expected_events[source_id]
        assert e['description']==source['description'] and e.get('text')==source.get('text')
        actual_ranges[source_id].append(dict(start=e['start'],end=e['end']))
    for eid,e in expected_events.items():assert actual_ranges[eid]==e['intervals']
    events_total+=len(actual_ranges);without_audio+=not bool(actual_ranges);silent+=not bool(physical.any())
    resumed=build_one(c,obs,target,lambda:(_ for _ in ()).throw(AssertionError('Resume tried to load encoder')))
    assert resumed['reused']
    assert sha(audio_folder/'payload.pt')==payload_hash and sha(audio_folder/'complete.json')==marker_hash
    assert sha(folder/'payload.pt')==base_payload_hash and sha(folder/'complete.json')==base_marker_hash and sha(original_label_path)==original_label_hash
    reports.append(dict(sample_rank=index+1,sample_id=sid,physical_virtual_waveform_bit_exact=True,
                        physical_virtual_audio_latents_bit_exact=True,reused_unchanged_original_audio=reuse_base,
                        updated_cache_matches_physical_encoding=(not reuse_base),video_and_visual_targets_bit_exact=True,
                        audio_labels_match_review=True,training_load_ok=True,teacher_omitted=True,
                        resume_without_encoder=True,old_cache_and_labels_unchanged=True,
                        remaining_audio_events=len(actual_ranges),audio_fully_silent=not bool(physical.any()),
                        audio_payload_bytes=(audio_folder/'payload.pt').stat().st_size,base_payload_bytes=(folder/'payload.pt').stat().st_size,
                        audio_shape=list(current['av']['audio_latents']['latents'].shape),seconds=time.monotonic()-tick))
    atomic_json(RUN/'acceptance100_progress.json',dict(completed=index+1,total=100,elapsed_s=time.monotonic()-started,last_sample_id=sid))
    if (index+1)%10==0:print(f'PASS {index+1}/100: physical-waveform/VAE/labels/loader/resume',flush=True)

assert events_total==63 and without_audio==62 and silent==53
assert not list(VIEW.rglob('*.wav')) and not (VIEW/'cache/vae').exists()
frozen=read(RUN/'frozen_code_before.json');assert all(sha(ROOT/name)==value for name,value in frozen.items())
summary=dict(status='passed',samples=100,audio_caches=100,event_caches=100,physical_virtual_waveform_bit_exact=100,
             physical_virtual_audio_latents_bit_exact=100,
             updated_audio_caches_verified=sum(not r['reused_unchanged_original_audio'] for r in reports),
             unchanged_original_audio_caches_verified=sum(r['reused_unchanged_original_audio'] for r in reports),
             training_load_passed=100,resume_without_encoder=100,
             video_and_visual_targets_unchanged=100,old_cache_and_labels_unchanged=100,
             retained_audio_events=events_total,samples_with_audio_labels=100-without_audio,samples_without_audio_labels=without_audio,
             fully_silent_audio_samples=silent,device='cpu',cuda_initialized=torch.cuda.is_initialized(),
             wav_files_written=0,video_cache_copies=0,audio_payload_bytes=sum(r['audio_payload_bytes'] for r in reports),
             avoided_base_payload_copy_bytes=sum(r['base_payload_bytes'] for r in reports),
             seconds=time.monotonic()-started,full_dataset_started=False,training_started=False)
atomic_json(RUN/'acceptance100.json',dict(summary=summary,implementation=implementation_identity(),review_sha256=sha(review_path),records=reports))
print(json.dumps(summary,indent=2),flush=True)
