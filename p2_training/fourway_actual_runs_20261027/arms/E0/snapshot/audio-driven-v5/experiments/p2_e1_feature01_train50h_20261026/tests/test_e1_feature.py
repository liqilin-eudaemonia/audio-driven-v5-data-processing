import copy,json,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(OUT),str(ROOT/'releases/p2_feature_v1/src'),str(ROOT/'src')]
from p2_listen.config import load
from p2_listen.audit import fixture,audit_model,audit_callback
from p2_listen.training import MODES,optimization_step,save_checkpoint,checked_checkpoint,restore_checkpoint,rng_state
from p2_listen.events import normalized_event_loss
from p2_listen.contracts import EVENTS
from p2_listen.monitoring import global_scalars
import control
from run_node import command

@pytest.fixture
def setup():
 torch.manual_seed(17);c=load(OUT/'config.json');cfg,sample=fixture(c);model=audit_model(c,cfg)
 opt=torch.optim.AdamW(model.optimizer_groups(c['training']),weight_decay=0)
 return c,sample,model,opt

def update(c,sample,model,opt,step,arm='E1',audit=True):
 return optimization_step(model,opt,[sample],audit_callback(model),mode=MODES[arm],pos_weights={k:1. for k in EVENTS},
  event_weight=c['events']['weight'],seed=20261023,ordinals=[step],config=c['training'],audit=audit)

def exact(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:exact(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):exact(x,y)
 else:assert a==b

def test_e1_feature_av_event_gradients_and_monitoring(setup):
 c,sample,m,opt=setup
 for step in range(5):rec=update(c,sample,m,opt,step)
 paths=rec['paths_first_micro']
 assert all(v>0 for key in ('av','feature','event') for v in paths[key])
 assert paths['text_lora']['av']>0 and paths['text_lora']['event']==0
 assert rec['feature_loss_weights']=={'video':.1,'audio':.1}
 assert rec['base']==pytest.approx(rec['av']+.1*sum(rec['feature_losses'].values()),rel=1e-6)
 timing=dict.fromkeys(('step_seconds','data_seconds','compute_seconds','checkpoint_seconds','allocated_gib','reserved_gib','peak_allocated_gib','peak_reserved_gib'),1.)
 scalars=global_scalars(rec,local_samples=1,media_seconds=1.,timings=timing,optimizer=opt,event_weight=.05,
  arm='E1',step=5,dataset_size=1,device='cpu')
 assert scalars['loss/generator_objective']==pytest.approx(rec['base']+.05*rec['event'])
 assert scalars['loss/optimization_objective']==scalars['loss/generator_objective']

@pytest.mark.parametrize('mode',['probe_only','joint_event'])
def test_event_gradient_ownership_with_feature_and_lora(setup,mode):
 c,sample,m,opt=setup
 result=m(sample[0],sample[1],sample[2],sample[4],mode,{k:1. for k in EVENTS},torch.Generator().manual_seed(7))
 event=normalized_event_loss(result['event_sums'],result['event_counts'])
 groups={'connector':list(m.extractor.connector.parameters()),'video_decoder':list(m.extractor.video_decoder.parameters()),
  'audio_decoder':list(m.extractor.audio_decoder.parameters()),'text_lora':list(m.text_lora.parameters()),'events':list(m.event_heads.parameters())}
 norms={}
 for name,params in groups.items():
  grads=torch.autograd.grad(event,params,allow_unused=True,retain_graph=True)
  norms[name]=sum(float(g.detach().abs().sum()) for g in grads if g is not None)
 assert norms['video_decoder']==norms['audio_decoder']==norms['text_lora']==0
 assert norms['events']>0
 assert (norms['connector']>0) if mode=='joint_event' else norms['connector']==0


def test_unknown_event_labels_leave_e1_identical_to_e0(setup):
 c,sample,m,opt=setup
 sample=copy.deepcopy(sample)
 for label in sample[4].values():label['known'].zero_()
 states=[]
 for arm in ('E0','E1'):
  clone=copy.deepcopy(m);optimizer=torch.optim.AdamW(clone.optimizer_groups(c['training']),weight_decay=0)
  rec=update(c,sample,clone,optimizer,0,arm,audit=False);assert rec['event']==0
  states.append((clone.state_dict(),optimizer.state_dict()))
 exact(*states)


def test_e1_feature_checkpoint_resume_exact_and_wrong_arm_rejected(setup,tmp_path):
 c,sample,m,opt=setup
 for step in range(3):update(c,sample,m,opt,step)
 identity={'arm':'E1','config':c};path=tmp_path/'step_000003.pt'
 save_checkpoint(path,m,opt,3,identity,[rng_state()]);saved=checked_checkpoint(path)
 other=copy.deepcopy(m);opt2=torch.optim.AdamW(other.optimizer_groups(c['training']),weight_decay=0)
 with pytest.raises(ValueError,match='identity'):restore_checkpoint(saved,other,opt2,dict(identity,arm='E0'))
 restore_checkpoint(saved,other,opt2,identity)
 a=update(c,sample,m,opt,3);b=update(c,sample,other,opt2,3)
 exact(m.state_dict(),other.state_dict());exact(opt.state_dict(),opt2.state_dict());exact(a,b)


def test_pair_is_only_arm_change():
 a=control.read(control.BASELINE/'run.json');b=dict(a,arm='E1')
 assert control.compare_pair(a,b)['status']=='passed'
 for key in ('initialization','shared_initialization','seed','world','source'):
  with pytest.raises(ValueError,match='differs'):control.compare_pair(a,dict(b,**{key:'changed'}))


def test_launcher_uses_separate_e1_output_and_same_config():
 assert control.read(OUT/'config.json')==control.read(control.TRAIN/'config.json')
 for rank in range(3):
  cmd=command(rank)
  assert cmd[cmd.index('--arm')+1]=='E1'
  assert cmd[cmd.index('--config')+1]==str(OUT/'config.json')
  assert '--nproc-per-node=8' in cmd and '--nnodes=3' in cmd and '--master-port=29960' in cmd
  assert '--resume' not in cmd
 assert control.E1!=control.BASELINE
 with pytest.raises(ValueError,match='E1'):control.validate_resume(control.BASELINE/'step_000100.pt')


def test_release_preserved():
 assert control.verify_release()['arm']=='E1'
 prior=control.read(control.REFERENCE/'release.json')
 assert all(control.sha(p)==h for p,h in prior['files'].items())
