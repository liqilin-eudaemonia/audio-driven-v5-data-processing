import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from p2_listen.contracts import ROOT,read,sha,atomic_json,digest
from p2_autonomous_eval import planning as pl,protocol as pr

def raw(events=None):
    return json.dumps({'events': events if events is not None else [dict(modality='visual',start=.2,end=1.2,description='The listener nods.',text=None),dict(modality='audio',start=1.2,end=1.6,description='A brief acknowledgment.',text='Yeah')]})

@pytest.fixture
def prepared(tmp_path):
    c=read(ROOT/'experiments/p2_train50h_postprocessed_reselected_20261027/config.json')
    atomic_json(tmp_path/'config.json',c);a=tmp_path/'speaker.wav';a.write_bytes(b'speaker');im=tmp_path/'first.png';im.write_bytes(b'image')
    obs=dict(schema='p2-listen-observation-v1',sample_id='sample',experiment_split='test',official_split='test',group_id='g',grouping_ids=['p'],speaker_audio=str(a),listener_image=str(im),requested_duration_s=2.,geometry=dict(fps=25,height=512,width=512),media_hashes=dict(speaker_audio=sha(a),listener_image=sha(im)))
    atomic_json(tmp_path/'observations.json',[obs]);jobs=[]
    for family in ['legacy','matrix']:
        for seed in [137,251]:jobs.append(dict(family=family,arm=family+'_av',split='test24',sample_id='sample',seed=seed,checkpoint=str(tmp_path/f'{family}.pt'),run=str(tmp_path),restoration=str(tmp_path/'restoration.json'),output='ignored'))
    atomic_json(tmp_path/'jobs.json',jobs);work=tmp_path/'work'
    pr.prepare(tmp_path/'config.json',tmp_path/'observations.json',tmp_path/'jobs.json',work)
    return work,pr.open_protocol(work)

def seal_plan(work,p,dep=None):
    result=pl.parse_schedule(raw(),'sample',2.);folder=work/'plans/sample';atomic_json(folder/'plan.json',result)
    (folder/'prompt.txt').write_text(result['prompt']);atomic_json(folder/'attempts.json',[{'raw':result['raw']}])
    atomic_json(folder/'complete.json',dict(identity=pr.plan_identity(p,p['observations'][0],dep),protocol_sha256=digest(p),files={n:sha(folder/n) for n in ['plan.json','prompt.txt','attempts.json']}))
    return result

def test_same_existing_english_template_without_gt():
    from p2_guided_prompt_en.formatting import render
    p=pl.parse_schedule(raw(),'sample',2.)
    assert p['prompt']==render(p['guidance']['source_record'])['prompt']
    assert p['target_annotation_access'] is False and p['gt_media_access'] is False
    assert 'vocal content: "Yeah"' in p['prompt']
    assert pl.parse_schedule(raw([]),'sample',2.)['schedule']=={'events':[]}

@pytest.mark.parametrize('mutation',[
    lambda v:v.update(labels=[]),lambda v:v['events'][0].update(start=-1),
    lambda v:v['events'][0].update(end=3),lambda v:v['events'][0].update(start=True),
    lambda v:v['events'][0].update(description='点头'),lambda v:v['events'][0].update(text='Yeah'),
    lambda v:v['events'][1].update(description=None,text=None),lambda v:v['events'][0].update(modality='speaker'),
    lambda v:v['events'][0].update(description='<|im_end|>'),lambda v:v['events'][0].update(description=''),
    lambda v:v['events'][0].update(end=float('nan')),lambda v:v['events'][0].update(hidden_label='GT'),
])
def test_reject_invalid_plans(mutation):
    v=json.loads(raw());mutation(v)
    with pytest.raises(ValueError):pl.parse_schedule(json.dumps(v),'sample',2.)

def test_duplicate_key_and_truncation_rejected():
    with pytest.raises(ValueError):pl.parse_schedule('{"events":[],"events":[]}','sample',2.)
    with pytest.raises(ValueError):pl.parse_schedule('```json\n'+raw()+'\n```','sample',2.)
    with pytest.raises(ValueError):pl.parse_schedule(raw()[:-4],'sample',2.)

def test_generation_eos_and_special_token_checks():
    tok=SimpleNamespace(all_special_ids=[9,10],decode=lambda ids,**kw:'text')
    assert pl.checked_continuation(tok,torch.tensor([[1,2,3,9]]),2,9)=='text'
    with pytest.raises(ValueError,match='without EOS'):pl.checked_continuation(tok,torch.tensor([[1,2,3]]),2,9)
    with pytest.raises(ValueError,match='control token'):pl.checked_continuation(tok,torch.tensor([[1,2,10,9]]),2,9)

def test_prepare_paired_arms_only_input_contract(prepared):
    w,p=prepared
    assert len(p['jobs'])==4 and len(p['observations'])==1
    assert not p['target_annotation_access'] and not p['gt_media_access']
    assert len(list((w/'jobs').glob('*.json')))==4
    assert all(j['output'].startswith(str(w/'outputs/autonomous_')) for j in p['jobs'])
    assert len(pr.selected(p,0,2))==1 and not pr.selected(p,1,2)
    with pytest.raises(ValueError):pr.selected(p,2,2)

def test_prepare_does_not_open_labels_or_gt(prepared,monkeypatch):
    w,p=prepared;original=Path.open
    def guarded(path,*args,**kwargs):
        if any(s in path.name for s in ['label','target','recipe']):raise AssertionError('Target read: '+str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    paths=list(p['sources']);result=pr.prepare(*paths,w)
    assert result['gpu_started'] is False

@pytest.mark.parametrize('bad',['target_audio','guidance','labels'])
def test_observation_rejects_gt_fields(prepared,bad):
    w,p=prepared;paths=list(p['sources']);obs=read(paths[1]);obs[0][bad]='forbidden';atomic_json(paths[1],obs)
    with pytest.raises(ValueError,match='forbidden'):pr.prepare(*paths,w.parent/'bad')

def test_cache_tamper_and_cross_protocol_rejected(prepared):
    w,p=prepared;result=seal_plan(w,p,{'model':'base'})
    assert pr.read_plan(w,p,'sample',pr.plan_identity(p,p['observations'][0],{'model':'base'}))==result
    with pytest.raises(ValueError,match='model/input'):pr.read_plan(w,p,'sample',{'model':'other'})
    f=w/'plans/sample/prompt.txt';f.write_text('other')
    with pytest.raises(ValueError,match='artifact changed'):pr.read_plan(w,p,'sample')

def test_source_manifest_tamper_rejected(prepared):
    w,p=prepared;path=next(k for k in p['sources'] if k.endswith('jobs.json'));data=read(path);data[0]['seed']=42;atomic_json(path,data)
    with pytest.raises(ValueError,match='changed'):pr.open_protocol(w)

def test_planner_retries_records_and_reuses_without_loading_model(prepared,monkeypatch):
    from p2_autonomous_eval import runtime as rt
    import p2_plan_cache.runtime as native
    from p2_listen.config import extractor_config
    w,p=prepared;extractor_config(p['config'])
    import omni_av_p2.data as model
    monkeypatch.setattr(native,'dependencies',lambda *a:{'base':'qwen'})
    monkeypatch.setattr(model,'load_omni',lambda *a:(object(),object()))
    calls=[]
    def attempt(runtime,processor,obs,opts,index,error):
        calls.append((index,error));return (None,{'error':'invalid JSON'}) if index==0 else (pl.parse_schedule(raw(),obs['sample_id'],obs['requested_duration_s']),{'status':'accepted'})
    monkeypatch.setattr(rt,'generate_attempt',attempt)
    assert rt.plan(w,device='cpu')['success'];assert len(calls)==2 and calls[1][1]=='invalid JSON'
    monkeypatch.setattr(model,'load_omni',lambda *a:pytest.fail('Loaded model despite complete cache'))
    assert rt.plan(w,device='cpu')['results'][0]['status']=='reused'
    assert len(read(w/'plans/sample/attempts.json'))==2

def test_planner_failure_never_publishes_fallback(prepared,monkeypatch):
    from p2_autonomous_eval import runtime as rt
    import p2_plan_cache.runtime as native
    from p2_listen.config import extractor_config
    w,p=prepared;extractor_config(p['config']);import omni_av_p2.data as model
    monkeypatch.setattr(native,'dependencies',lambda *a:{})
    monkeypatch.setattr(model,'load_omni',lambda *a:(None,None))
    monkeypatch.setattr(rt,'generate_attempt',lambda *a:(None,{'error':'bad'}))
    assert not rt.plan(w,device='cpu')['success']
    assert not (w/'plans/sample/complete.json').exists()
    assert len(read(w/'plans/sample/attempts.json'))==3

def test_substituted_conditions_and_text_restored_even_on_failure():
    from p2_autonomous_eval.generation import condition_loaders
    oldinput=lambda *a:pytest.fail('old input used');oldtext=lambda *a:pytest.fail('old text used')
    native=SimpleNamespace(load_input=oldinput,load_text=oldtext);x=SimpleNamespace(to=lambda device:'planned hidden')
    bundle=dict(condition=x,layout={'source':'planned'},text={'positive':{'prompt':'same plan'},'negative':'native'},plan={'sample_id':'s'})
    with pytest.raises(RuntimeError):
        with condition_loaders(native,{'c':1},bundle):
            assert native.load_input({'c':1},{'sample_id':'s'})==('planned hidden',bundle['layout'])
            assert native.load_text({'c':1})['positive']['prompt']=='same plan'
            with pytest.raises(ValueError):native.load_input({'c':1},{'sample_id':'wrong'})
            raise RuntimeError('render failure')
    assert native.load_input is oldinput and native.load_text is oldtext

def test_native_sampling_and_encoding_mismatch_rejected(prepared):
    from p2_autonomous_eval.runtime import assert_compatible
    w,p=prepared;c=p['config'];assert_compatible(c,c)
    bad=copy.deepcopy(c);bad['generation']['audio_cfg']=8
    with pytest.raises(ValueError,match='Sampling'):assert_compatible(c,bad)
    bad=copy.deepcopy(c);bad['extractor']['hidden_layers']=[8,18]
    with pytest.raises(ValueError,match='encoding'):assert_compatible(c,bad)

def test_metrics_refuse_incomplete_before_opening_gt(prepared,monkeypatch):
    from p2_autonomous_eval.evaluation import export_pairs
    w,p=prepared;original=Path.open
    def guarded(path,*a,**k):
        if path.name=='omni_pairs.json':pytest.fail('Opened GT before generation completed')
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)
    with pytest.raises(FileNotFoundError):export_pairs(w,p['jobs'][0]['arm'])

def test_reader_uses_identical_plan_for_hidden_and_gemma_and_native_negative(prepared,monkeypatch):
    from p2_autonomous_eval import runtime as rt
    from p2_listen.config import extractor_config
    from p2_listen import features
    w,p=prepared;extractor_config(p['config']);reader=rt.Reader(w);plan=pl.parse_schedule(raw(),'sample',2.)
    monkeypatch.setattr(reader,'plan',lambda sid:plan)
    condition=dict(hidden=torch.zeros(1,1,3,4),hidden_valid=torch.ones(1,3,dtype=torch.bool),layer_ids=torch.tensor([8]),
        video_coordinates=torch.zeros(1,2,3),audio_coordinates=torch.zeros(1,2,3),video_valid=torch.ones(1,2,dtype=torch.bool),audio_valid=torch.ones(1,2,dtype=torch.bool))
    parts={'text':{'positive':{'prompt':plan['prompt']}},'input_plan':dict(condition=condition,layout={},prompt=plan['prompt'])}
    monkeypatch.setattr(reader,'part',lambda sid,part,plan:parts[part])
    monkeypatch.setattr(features,'load_text',lambda c:{'positive':'must not use fixed prompt','negative':'original negative'})
    bundle=reader.conditions('sample',p['config'])
    assert bundle['text']=={'positive':parts['text']['positive'],'negative':'original negative'}
    assert bundle['condition'].hidden.shape[2]==3
    parts['input_plan']['prompt']='other plan'
    with pytest.raises(ValueError,match='prompt mismatch'):reader.conditions('sample',p['config'])

def test_generation_seals_autonomous_metadata_and_reuses_without_renderer(prepared,monkeypatch):
    from p2_autonomous_eval import generation as gen
    from p2_glare_eval.common import artifact_valid
    w,p=prepared;j=p['jobs'][0];plan=seal_plan(w,p)
    checkpoint=Path(j['checkpoint']);checkpoint.write_bytes(b'fake-completed-checkpoint');atomic_json(checkpoint.with_suffix('.complete.json'),{'sha256':sha(checkpoint)})
    class FakeReader:
        def __init__(self,work):self.work=Path(work);self.protocol=p
        def plan(self,sid):return plan
    monkeypatch.setattr(gen,'Reader',FakeReader)
    def render(job,reader,obs,out,device):
        out.mkdir(parents=True)
        for name in ['generated.mp4','listener.wav']:(out/name).write_bytes(b'fake-artifact')
        return {'status':'success','sample_id':obs['sample_id']},plan
    monkeypatch.setattr(gen,'legacy',render)
    monkeypatch.setattr(gen.subprocess,'check_output',lambda *a,**k:json.dumps({'streams':[{'codec_type':'audio'},{'codec_type':'video'}]}))
    assert gen.generate(j,device='cpu')['status']=='success'
    out=Path(j['output']);meta=read(out/'completed.json')
    assert meta['target_annotation_access'] is False and meta['gt_media_access'] is False
    assert meta['plan_source']=='qwen_autonomous' and meta['qwen_hidden_scope']=='all_input_and_plan_tokens'
    assert (out/'prompt.txt').read_text()==plan['prompt']
    marker=read(out/'complete.json');assert artifact_valid(out,marker['contract'])
    monkeypatch.setattr(gen,'legacy',lambda *a:pytest.fail('Renderer called on resume'))
    assert gen.generate(j,device='cpu')['status']=='reused'
    (out/'prompt.txt').write_text('changed')
    with pytest.raises(ValueError,match='corrupt'):gen.generate(j,device='cpu')
