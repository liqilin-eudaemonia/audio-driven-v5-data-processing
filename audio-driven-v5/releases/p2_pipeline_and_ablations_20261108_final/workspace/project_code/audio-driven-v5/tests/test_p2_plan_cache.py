import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from p2_listen.contracts import ROOT, atomic_json, digest, read, sha
from p2_plan_cache import protocol as pr
from p2_plan_cache.encoding import append_plan


def annotation(sid='sample'):
    return dict(schema='listen-existing-result-fusion-verbatim-v2', sample_id=sid, duration_s=3.,
                events=[dict(event_id='v1', modality='visual', start=.1234567, end=2.,
                             description='The listener smiles.', text=None),
                        dict(event_id='a1', modality='audio', start=1., end=2.5,
                             intervals=[dict(start=1., end=1.2), dict(start=2.1, end=2.5)],
                             description='A short acknowledgment.', text='Yeah')])


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path/'dataset'; root.mkdir()
    config = read(ROOT/'experiments/p2_train50h_postprocessed_reselected_20261027/config.json')
    config['output'] = str(root)
    atomic_json(root/'config.json', config)
    observations = []
    records = []
    index = []
    for sid in ('sample', 'sample2'):
        audio = root/(sid+'.wav'); audio.write_bytes(b'speaker-placeholder')
        image = root/(sid+'.png'); image.write_bytes(b'first-frame-placeholder')
        observations.append(dict(schema='p2-listen-observation-v1', sample_id=sid,
            experiment_split='train', official_split='train', group_id='group', grouping_ids=['participant'],
            speaker_audio=str(audio), listener_image=str(image), requested_duration_s=3.,
            geometry=dict(fps=25, height=512, width=512),
            media_hashes=dict(speaker_audio=sha(audio), listener_image=sha(image))))
        records.append(annotation(sid))
        label=root/'labels/canonical'/(sid+'.json'); recipe=root/'audio_postprocess/recipes'/(sid+'.json')
        atomic_json(label, {'sample_id':sid}); atomic_json(recipe, {'sample_id':sid})
        index.append(dict(sample_id=sid, canonical_sha256=sha(label), recipe_sha256=sha(recipe)))
    atomic_json(root/'manifests/observations.json', observations)
    (root/'integrated_labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (root/'dataset.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
    files = {name:sha(root/name) for name in ['config.json','manifests/observations.json',
                                            'integrated_labels.jsonl','dataset.jsonl']}
    atomic_json(root/'COMPLETE.json', dict(status='passed', samples=2, files=files))
    return root


def prepared(dataset, tmp_path):
    work=tmp_path/'work'
    pr.prepare(dataset/'config.json', work)
    return work, pr.open_protocol(work)


def test_prepare_exact_old_format_postprocessed_intervals_resume(dataset, tmp_path):
    from p2_guided_prompt_en.formatting import render
    work,p=prepared(dataset,tmp_path)
    item=pr.sample(work,p,'sample');g=item['guidance']
    assert g['prompt'] == render(annotation())['prompt']
    assert g['source_record'] == annotation()
    assert '[1.0–1.2 seconds]' in g['prompt'] and '[2.1–2.5 seconds]' in g['prompt']
    assert '[1.0–2.5 seconds]' not in g['prompt']
    assert len([e for e in g['entries'] if e['modality']=='audio']) == 2
    assert not (dataset/'cache').exists()  # No original cache mutation or model load.
    assert pr.prepare(dataset/'config.json',work)['success']
    with pytest.raises(ValueError,match='Existing identity'):
        pr.prepare(dataset/'config.json',work,['sample'])


def test_annotation_and_dataset_mutations_fail(dataset,tmp_path):
    work,p=prepared(dataset,tmp_path)
    item=read(work/'samples/sample.json');item['guidance']['prompt']+='changed'
    atomic_json(work/'samples/sample.json',item)
    with pytest.raises(ValueError,match='Frozen sample'):pr.sample(work,p,'sample')
    with (dataset/'integrated_labels.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='Frozen dataset'):pr.open_protocol(work)


@pytest.mark.parametrize('change', ['control_token','duplicate','missing','duration'])
def test_invalid_annotation_cannot_publish_protocol(dataset,tmp_path,change):
    records=[annotation('sample'),annotation('sample2')]
    if change=='control_token':records[0]['events'][1]['text']='<|im_end|>'
    elif change=='duplicate':records.append(annotation('sample'))
    elif change=='missing':records.pop()
    else:records[0]['duration_s']=4.
    (dataset/'integrated_labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    complete=read(dataset/'COMPLETE.json');complete['files']['integrated_labels.jsonl']=sha(dataset/'integrated_labels.jsonl')
    atomic_json(dataset/'COMPLETE.json',complete)
    with pytest.raises(ValueError):pr.prepare(dataset/'config.json',tmp_path/'work')
    assert not (tmp_path/'work/protocol.json').exists()


def test_reject_dataset_output_and_bad_selection(dataset,tmp_path):
    with pytest.raises(ValueError,match='separate'):pr.prepare(dataset/'config.json',dataset/'new')
    with pytest.raises(ValueError,match='selected'):pr.prepare(dataset/'config.json',tmp_path/'work',['absent'])


def test_preserve_original_non_english_without_translation():
    r=annotation();r['events'][1].update(description='哎',text='哎')
    value=pr.render_clean(r)
    assert 'Audio response: "哎"' in value['prompt']
    assert '"Ah"' not in value['prompt'] and not value['translations']
    assert value['source_record']==r and value['policy']['translate'] is False


def test_shards_are_disjoint_and_complete(dataset,tmp_path):
    _,p=prepared(dataset,tmp_path)
    parts=[pr.select(p,shard_index=i,shards=3) for i in range(3)]
    assert sorted(sum(parts,[]))==p['sample_ids']
    assert len(sum(parts,[]))==len(set(sum(parts,[])))
    with pytest.raises(ValueError):pr.select(p,shard_index=3,shards=3)


def test_system_task_change_invalidates_planned_hidden_cache(dataset,tmp_path,monkeypatch):
    from p2_plan_cache.runtime import identity
    from p2_listen.cache import save_part, load_part
    from p3_listen import prompts
    work,p=prepared(dataset,tmp_path)
    item=pr.sample(work,p,'sample')
    before=identity(p,item,'input_plan',{'model':'same'})
    folder=tmp_path/'hidden'
    save_part(folder,before,{'hidden':torch.ones(1)})
    monkeypatch.setattr(prompts,'OMNI',prompts.OMNI+'\nUpdated listening task.')
    after=identity(p,item,'input_plan',{'model':'same'})
    assert before['prompt_sha256']==after['prompt_sha256']
    with pytest.raises(ValueError,match='Cache identity differs'):
        load_part(folder,after)


class Tokenizer:
    all_special_ids=[511]
    unk_token_id=None
    def encode(self,text,add_special_tokens=False):return [ord(c)%300+1 for c in text]
    def convert_tokens_to_ids(self,text):return 511


def prefix(n=4,width=32):
    return dict(prefix=torch.randn(1,n,width), prefix_valid=torch.ones(1,n,dtype=torch.bool),
                prefix_positions=torch.arange(n)[None,None].expand(3,1,-1))


def test_append_tokens_masks_and_native_continuation():
    pref=prefix();embed=torch.nn.Embedding(512,32)
    full=append_plan(pref,Tokenizer(),embed,'Yeah',100,100)
    assert torch.equal(full['prefix'][:,:4],pref['prefix'])
    assert full['plan_mask'].tolist()==[[False]*4+[True]*4+[False]]
    assert full['prefix_positions'][0,0].tolist()==list(range(9))
    assert full['plan_token_ids'].tolist()==[Tokenizer().encode('Yeah')]
    with pytest.raises(OverflowError):append_plan(pref,Tokenizer(),embed,'Yeah',8,100)
    pref['prefix_valid'][0,0]=False
    with pytest.raises(ValueError,match='unpadded'):append_plan(pref,Tokenizer(),embed,'Yeah',100,100)


@pytest.mark.parametrize('mutation', ['prompt','mask','hidden','nan'])
def test_hidden_validation_rejects_tampering(dataset,tmp_path,mutation):
    from p2_listen.config import activate
    from p2_plan_cache.runtime import validate_payload
    work,p=prepared(dataset,tmp_path);activate(p['config'])
    item=pr.sample(work,p,'sample');value=fake_hidden(item)
    validate_payload(value,item,'input_plan')
    if mutation=='prompt':value['prompt']='wrong'
    elif mutation=='mask':value['plan_mask'].zero_()
    elif mutation=='hidden':value['condition']['hidden']=value['condition']['hidden'][:,:,:2]
    else:value['condition']['hidden'][0,0,0,0]=float('nan')
    with pytest.raises(ValueError):validate_payload(value,item,'input_plan')


def fake_hidden(item):
    g=item['guidance'];n=4;m=3;length=n+m+1
    coords=torch.zeros(1,1,3);valid=torch.ones(1,1,dtype=torch.bool)
    condition=dict(hidden=torch.randn(1,2,length,32),hidden_valid=torch.ones(1,length,dtype=torch.bool),
                   layer_ids=torch.tensor([1,2]),video_coordinates=coords,audio_coordinates=coords.clone(),
                   video_valid=valid,audio_valid=valid.clone())
    mask=torch.zeros(1,length,dtype=torch.bool);mask[:,n:n+m]=True
    return dict(condition=condition,layout={},prompt=g['prompt'],prompt_sha256=g['prompt_sha256'],
                plan_mask=mask,plan_token_ids=torch.ones(1,m,dtype=torch.long),
                position_ids=torch.arange(length)[None,None].expand(3,1,-1),
                prefix_tokens=n,plan_tokens=m,terminal_tokens=1,qwen_frozen=True,truncated=False)


def test_identity_changes_with_plan_input_model_and_limits(dataset,tmp_path):
    from p2_plan_cache.runtime import identity
    work,p=prepared(dataset,tmp_path);item=pr.sample(work,p,'sample')
    original=identity(p,item,'input_plan',{'model':'v1'})
    changed=copy.deepcopy(item);changed['observation']['media_hashes']['speaker_audio']='other'
    assert original!=identity(p,changed,'input_plan',{'model':'v1'})
    assert original!=identity(p,item,'input_plan',{'model':'v2'})
    changed=copy.deepcopy(item);changed['guidance']['prompt']+='more'
    assert original!=identity(p,changed,'input_plan',{'model':'v1'})
    assert original!=identity(dict(p,max_qwen_tokens=100),item,'input_plan',{'model':'v1'})


def test_writer_resume_corruption_and_pair_reader(dataset,tmp_path,monkeypatch):
    from p2_listen.config import activate
    from p2_plan_cache import runtime as rt
    from p2_plan_cache import reader as rd
    from p2_plan_cache import encoding as enc
    from p2_listen import features
    work,p=prepared(dataset,tmp_path);activate(p['config'])
    import omni_av_p2.data as native
    calls=[]
    monkeypatch.setattr(rt,'dependencies',lambda *args:{'fake':'test'})
    monkeypatch.setattr(rd,'dependencies',lambda *args:{'fake':'test'})
    monkeypatch.setattr(native,'load_omni',lambda *args:(object(),object()))
    def extract(c,obs,prompt,*args):
        calls.append(obs['sample_id'])
        return fake_hidden(pr.sample(work,p,obs['sample_id']))
    monkeypatch.setattr(enc,'extract_planned_input',extract)
    assert rt.cache(work,'input_plan',sample_ids=['sample'],device='cpu')['success']
    assert rt.cache(work,'input_plan',sample_ids=['sample'],device='cpu')['results'][0]['status']=='reused'
    assert calls==['sample']
    assert rd.verify(work,['sample'],parts=['input_plan'])['success']
    assert not rd.verify(work,parts=['input_plan'])['success']  # Missing second sample is failure.
    item=pr.sample(work,p,'sample')
    positive=dict(prompt=item['guidance']['prompt'], video=torch.zeros(1,1024,4096),
                  audio=torch.zeros(1,1024,2048),mask=torch.ones(1,1024,dtype=torch.bool))
    payload=dict(positive=positive,prompt_sha256=item['guidance']['prompt_sha256'],
                 sequence_length=1024,tokens=1024,truncated=False)
    from p2_listen.cache import save_part
    save_part(work/'cache/text/sample',rt.identity(p,item,'text',{'fake':'test'}),payload)
    monkeypatch.setattr(features,'load_text',lambda *args:dict(positive={'prompt':'fixed'},negative={'prompt':''}))
    bundle=rd.PlanCacheReader(work).conditions('sample','A2')
    assert bundle['condition'].hidden.shape[2]==8
    assert bundle['text']['positive']['prompt']==item['guidance']['prompt']
    assert bundle['text']['negative']['prompt']==''
    with (work/'cache/input_plan/sample/payload.pt').open('ab') as f:f.write(b'corrupt')
    assert not rt.cache(work,'input_plan',sample_ids=['sample'],device='cpu')['success']
    assert calls==['sample']  # Corruption never silently overwritten/re-extracted.


def test_native_tiny_qwen_causal_prefix_and_connector_plan_visibility():
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTextConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerTextModel
    from p2_listen.config import activate
    activate(read(ROOT/'experiments/p2_train50h_postprocessed_reselected_20261027/config.json'))
    from omni_av_p2.config import Config
    from omni_av_p2.connector import ConditionInputs, LayerMemory
    from omni_av_p2.data import select_hidden_layers
    config=Qwen2_5OmniTextConfig(vocab_size=512,hidden_size=32,intermediate_size=64,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096,
        rope_scaling={'type':'default','mrope_section':[1,1,2]})
    config._attn_implementation='eager'
    model=Qwen2_5OmniThinkerTextModel(config).eval().requires_grad_(False)
    pref=prefix()
    with torch.no_grad():
        base=model(inputs_embeds=pref['prefix'],position_ids=pref['prefix_positions'],
                   attention_mask=pref['prefix_valid'],output_hidden_states=True,use_cache=False)
        full=append_plan(pref,Tokenizer(),model.get_input_embeddings(),'Yeah',4096,4096)
        out=model(inputs_embeds=full['prefix'],position_ids=full['prefix_positions'],
                  attention_mask=full['prefix_valid'],output_hidden_states=True,use_cache=False)
    h=select_hidden_layers(out.hidden_states,(1,2),3)
    assert torch.allclose(h[:,:,:4],select_hidden_layers(base.hidden_states,(1,2),3),atol=1e-6)
    assert h.shape==(1,2,9,32) and not h.requires_grad
    item=dict(guidance=dict(prompt='Yeah',prompt_sha256=digest('Yeah')))
    data=fake_hidden(item)['condition'];data.update(hidden=h,hidden_valid=full['prefix_valid'])
    x=ConditionInputs(**data);x.validate()
    memory=LayerMemory(Config(hidden_size=32,hidden_layers=(1,2),fusion_width=32,fusion_heads=4))
    _,valid=memory(x)
    assert valid.shape==(1,18) and valid.all()  # Both selected layers retain every plan token.


def test_text_writer_uses_same_prompt_without_negative_duplication(dataset,tmp_path,monkeypatch):
    from p2_plan_cache import runtime as rt
    from p2_plan_cache import reader as rd
    from p2_listen.config import activate
    work,p=prepared(dataset,tmp_path);activate(p['config'])
    import omni_av.ltx_runtime as native
    import p2_plan_cache.encoding as encoding
    calls=[]
    class Encoder:
        def __init__(self,*args):calls.append('load')
        def close(self):calls.append('close')
    def encode(model,prompt,maximum):
        calls.append(prompt)
        return dict(video=torch.zeros(1,1024,4096),audio=torch.zeros(1,1024,2048),
                    mask=torch.ones(1,1024,dtype=torch.bool),prompt=prompt), dict(
                    tokens=3,sequence_length=1024,truncated=False)
    monkeypatch.setattr(native,'NativeTextConditioner',Encoder)
    monkeypatch.setattr(encoding,'encode_text',encode)
    monkeypatch.setattr(rt,'dependencies',lambda *args:{'fake':'test'})
    monkeypatch.setattr(rd,'dependencies',lambda *args:{'fake':'test'})
    assert rt.cache(work,'text',sample_ids=['sample'],device='cpu')['success']
    assert calls==['load',pr.sample(work,p,'sample')['guidance']['prompt'],'close']
    assert rt.cache(work,'text',sample_ids=['sample'],device='cpu')['results'][0]['status']=='reused'
    value=rd.PlanCacheReader(work).part('sample','text')
    assert 'negative' not in value
    assert value['tokens']==3 and int(value['positive']['mask'].sum())==1024


def test_longest_training_prompt_uses_8192_without_truncation():
    from p2_plan_cache.encoding import text_length
    assert text_length(4981)==8192
    assert text_length(4096)==4096
    with pytest.raises(OverflowError):text_length(4981,4096)


def test_a2_reader_composes_overlay_without_teacher_or_original_hidden(dataset,tmp_path,monkeypatch):
    from p2_plan_cache.reader import PlanCacheReader
    from p2_listen.config import activate
    work,p=prepared(dataset,tmp_path);activate(p['config'])
    import listen_audio_postprocess.overlay as overlay
    import p2_plan_cache.reader as reader_module
    import p2_listen.features as features
    reader=PlanCacheReader(work)
    monkeypatch.setattr(reader,'conditions',lambda sid,variant,device:dict(metadata={'variant':variant}))
    monkeypatch.setattr(overlay,'sample_contract',lambda *args:(Path('/new'),Path('/base'),{},None,{},{}))
    monkeypatch.setattr(features,'identity',lambda *args,**kwargs:{})
    visited=[]
    def load(folder,expected,device):
        visited.append(str(folder))
        if str(folder)=='/base':return {'av':{'video':'original','audio':'old'}}
        if str(folder).startswith('/new/'):return {'new_audio':True}
        return {'events':'updated'}
    monkeypatch.setattr(reader_module,'load_part',load)
    monkeypatch.setattr(overlay,'compose',lambda base,patch:dict(video=base['av']['video'],audio=patch))
    result=reader.training_sample('sample',{'sample_id':'sample'})
    assert result['teacher'] is None and result['av']['audio']=={'new_audio':True}
    assert result['events']=={'events':'updated'}
    assert not any('/teacher' in p or '/input/' in p for p in visited)
