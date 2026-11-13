import copy,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'releases/p2_feature_v1/src'),str(ROOT/'src'),str(ROOT/'tests')]
from p2_listen.config import load,activate,feature_loss_weights
from p2_listen.model import ListeningExtractor
from p2_listen.audit import fixture,TinyTextRenderer,audit_callback
from p2_listen.training import optimization_step
from p2_listen.contracts import EVENTS

def config():return load(ROOT/'experiments/p2_feature01_train50h_20261026/production/config.json')

@pytest.mark.parametrize('weights',[{},dict(video=-1,audio=1),dict(video=0,audio=1),dict(video=float('nan'),audio=1),dict(video=True,audio=1),dict(video=.1,audio=.1,other=1)])
def test_bad_weights(weights):
 with pytest.raises((ValueError,TypeError)):feature_loss_weights(dict(feature_loss_weights=weights))

def test_shared_initialization_and_rng_exactly_match_no_decoder():
 c=config();cfg,sample=fixture(c);torch.manual_seed(24);a=ListeningExtractor(cfg,8,feature_loss_enabled=False);state=torch.get_rng_state()
 torch.manual_seed(24);b=ListeningExtractor(cfg,8,feature_loss_enabled=True)
 assert torch.equal(torch.get_rng_state(),state)
 for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k]),k
 assert hasattr(b.extractor,'video_decoder') and hasattr(b.extractor,'audio_decoder')

def test_lora_shared_initialization_is_preserved():
 c=config();cfg,sample=fixture(c);spec=dict(rank=2,alpha=2.,learning_rate=1e-5)
 renderer=TinyTextRenderer()
 torch.manual_seed(24);a=ListeningExtractor(cfg,8,feature_loss_enabled=False,text_lora=spec,renderer=copy.deepcopy(renderer));state=torch.get_rng_state()
 torch.manual_seed(24);b=ListeningExtractor(cfg,8,feature_loss_enabled=True,text_lora=spec,renderer=copy.deepcopy(renderer))
 assert torch.equal(torch.get_rng_state(),state)
 for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k]),k
 b.assert_parameter_ownership()

def test_loss_coefficient_scales_decoder_updates_and_keeps_av_rng():
 c=config();cfg,sample=fixture(c);torch.manual_seed(24);base=ListeningExtractor(cfg,8,feature_loss_enabled=True)
 outputs=[]
 for weight in (1.,.1):
  m=copy.deepcopy(base);t=dict(c['training'],clip_norm=1e9,feature_loss_weights=dict(video=weight,audio=weight))
  opt=torch.optim.SGD(m.optimizer_groups(t),lr=1.)
  def av(u,pack):return u.video.float().square().mean()+u.audio.float().square().mean()
  rec=optimization_step(m,opt,[sample],av,mode='probe_only',pos_weights={k:1. for k in EVENTS},event_weight=.05,seed=41,ordinals=[0],config=t)
  assert rec['base']==pytest.approx(rec['av']+sum(rec['feature_weighted_losses'].values()),rel=1e-6)
  outputs.append((m,rec))
 assert outputs[0][1]['av']==outputs[1][1]['av']
 assert outputs[0][1]['feature_losses']==outputs[1][1]['feature_losses']
 key='extractor.video_decoder.net.final_layer.linear.weight'
 d1=outputs[0][0].state_dict()[key]-base.state_dict()[key];d2=outputs[1][0].state_dict()[key]-base.state_dict()[key]
 assert d1.abs().sum()>0 and torch.allclose(d2,d1*.1,rtol=2e-5,atol=1e-9)

def test_loss_weight_does_not_change_cached_extractor_config():
 from p2_listen.config import extractor_config
 c=config();prior=load(ROOT/'experiments/p2_e1_train50h_lora_tb_20261026/config.json')
 assert extractor_config(c).to_dict()==extractor_config(prior).to_dict()

def test_preclip_gradients_have_exact_feature_scale_for_both_decoders(monkeypatch):
 import p2_listen.training as train
 c=config();cfg,sample=fixture(c);torch.manual_seed(41);base=ListeningExtractor(cfg,8,feature_loss_enabled=True)
 captured=[]
 original=train.clip_groups
 def capture(model,limit):
  captured.append({k:p.grad.clone() for k,p in model.named_parameters() if 'decoder.net.final_layer.linear.weight' in k})
  return original(model,limit)
 monkeypatch.setattr(train,'clip_groups',capture)
 for weight in (1.,.1):
  m=copy.deepcopy(base);t=dict(c['training'],feature_loss_weights=dict(video=weight,audio=weight))
  opt=torch.optim.SGD(m.optimizer_groups(t),lr=1.)
  optimization_step(m,opt,[sample],lambda u,p:u.video.square().mean()+u.audio.square().mean(),mode='probe_only',
   pos_weights={k:1. for k in EVENTS},event_weight=.05,seed=41,ordinals=[0],config=t)
 assert len(captured[0])==2
 for k,g in captured[0].items():
  assert g.abs().sum()>0
  torch.testing.assert_close(captured[1][k],g*.1,rtol=2e-5,atol=1e-8)

def test_tensorboard_separates_weighted_feature_and_generator_objective():
 from p2_listen.monitoring import global_scalars
 c=config();cfg,sample=fixture(c);m=ListeningExtractor(cfg,8,feature_loss_enabled=True)
 opt=torch.optim.SGD(m.optimizer_groups(c['training']),lr=1.)
 rec=optimization_step(m,opt,[sample],lambda u,p:1+u.video.square().mean()+u.audio.square().mean(),mode='probe_only',
  pos_weights={k:1. for k in EVENTS},event_weight=.05,seed=41,ordinals=[0],config=c['training'])
 timing=dict.fromkeys(('step_seconds','data_seconds','compute_seconds','checkpoint_seconds','allocated_gib','reserved_gib','peak_allocated_gib','peak_reserved_gib'),1.)
 values=global_scalars(rec,local_samples=1,media_seconds=1.,timings=timing,optimizer=opt,event_weight=.05,arm='E0',step=1,dataset_size=1,device='cpu')
 for mod in ('video','audio'):
  assert values['loss/feature_weighted_'+mod]==pytest.approx(.1*values['loss/feature_'+mod])
 assert values['loss/generator_objective']==rec['base']
 assert values['loss/feature_to_av_ratio']==pytest.approx(values['loss/feature_weighted_total']/values['loss/av'])
