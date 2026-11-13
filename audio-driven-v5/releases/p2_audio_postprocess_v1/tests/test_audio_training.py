import copy,sys
from pathlib import Path
import pytest,torch
RELEASE=Path(__file__).resolve().parents[1];ROOT=RELEASE.parents[1];sys.path[:0]=[str(RELEASE/'src'),str(ROOT/'src'),str(ROOT/'tests')]
from p2_listen import training_data as td,preflight
from p2_listen.contracts import digest,read
from p2_listen.config import load,source_identity
import test_listener_audio_virtual as fixtures


def identity():
 return dict(part='audio_postprocess',recipe={'test':1},base_payload_sha256='base',code={str(Path(td.overlay.__file__)):td.CURRENT_OVERLAY_SHA,'encoder':'immutable'})

def test_accept_only_exact_legacy_code_change():
 expected=identity();legacy=copy.deepcopy(expected);legacy['code'][str(Path(td.overlay.__file__))]=td.TRAIN50H_OVERLAY_SHA
 assert td.audio_cache_identity(expected,dict(identity=digest(legacy),contract=legacy))==legacy
 assert expected['code'][str(Path(td.overlay.__file__))]==td.CURRENT_OVERLAY_SHA
 for field in ('recipe','base_payload_sha256','code'):
  changed=copy.deepcopy(legacy);changed[field]={'tampered':True}
  with pytest.raises(ValueError):td.audio_cache_identity(expected,dict(identity=digest(changed),contract=changed))

@pytest.mark.parametrize('reuse',[False,True])
def test_training_combines_audio_with_new_events_and_same_video(tmp_path,monkeypatch,reuse):
 base=fixtures.Composition().base();patch={'reuse_base_audio':True} if reuse else fixtures.Composition().audio();new_events={'new_filtered':torch.ones(3)};expected=identity()
 monkeypatch.setattr(td.overlay,'sample_contract',lambda *a:(tmp_path,tmp_path/'base',{'base':True},{},{'new':'label'},expected))
 monkeypatch.setattr(td,'read',lambda p:dict(identity=digest(expected),contract=expected))
 monkeypatch.setattr(td.features,'load_input',lambda *a:('input','layout'))
 monkeypatch.setattr(td.features,'identity',lambda *a,**kw:({'new':'events'} if kw.get('label')=={'new':'label'} else None))
 monkeypatch.setattr(td.features,'folder',lambda c,p,s:tmp_path/p/s)
 def load_part(path,ident,device):
  if path==tmp_path/'base':return base
  if 'audio_postprocess' in path.parts:return patch
  assert 'events' in path.parts and ident=={'new':'events'};return new_events
 monkeypatch.setattr(td,'load_part',load_part)
 result=td.load_training_sample({},dict(sample_id='s'),{})
 assert result[2] is None and result[4] is new_events
 assert result[3]['av']['latents']['latents'] is base['av']['latents']['latents']
 assert result[3]['av']['audio_latents'] is (base['av']['audio_latents'] if reuse else patch['audio_latents'])


def test_dependency_receipt_binds_patch_recipe_and_actual_base(tmp_path,monkeypatch):
 root=tmp_path/'view';base=tmp_path/'base';monkeypatch.setattr(td.overlay,'settings',lambda c:(root,base))
 c={'output':str(root),'training':{'feature_loss_enabled':False}}
 o=dict(sample_id='s',speaker_audio='speaker.wav',listener_image='first.png');t=dict(target_video='v.mp4',target_audio='a.wav')
 meta=dict(manifest_states={},binding={'source':{},'assets':{}},rows=[o],targets={'s':t})
 paths=preflight.dependencies(c,meta)
 for file in [root/'audio_postprocess/policy_manifest.json',root/'audio_postprocess/recipes/s.json',root/'cache/audio_postprocess/s/complete.json',root/'cache/audio_postprocess/s/payload.pt',base/'cache/vae/s/payload.pt',root/'cache/events/s/payload.pt']:assert str(file) in paths
 assert not any('/teacher/' in p for p in paths)
 assert str(root/'cache/vae/s/payload.pt') not in paths


def test_no_stale_teacher_and_source_identity_binds_reader(tmp_path):
 path=ROOT/'experiments/p2_train50h_audio_postprocess_20261026/view/config.json';c=load(path)
 files=source_identity(c)
 assert str(Path(td.__file__)) in files and str(Path(td.overlay.__file__)) in files and str(td.LEGACY_SOURCE) in files
 changed=copy.deepcopy(c);changed['training'].update(feature_loss_enabled=True,feature_lr=1e-5)
 import json
 q=tmp_path/'config.json';q.write_text(json.dumps(changed))
 with pytest.raises(ValueError,match='stale teacher'):load(q)
 del changed['data']['audio_postprocess'];changed['training']['feature_loss_enabled']=False;q.write_text(json.dumps(changed))
 with pytest.raises(ValueError,match='Invalid audio postprocess'):load(q)
