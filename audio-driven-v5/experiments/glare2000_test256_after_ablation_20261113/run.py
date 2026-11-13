"""Reuse the accepted three-node GLARE256 evaluator for early batch8 step2000."""
import argparse,copy,fcntl,importlib.util,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
SELF=Path(__file__).resolve();BASE=SELF.parent;ROOT=BASE.parents[1]
PARENT=ROOT/'experiments/glare_test256_3nodes_20261113'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
spec=importlib.util.spec_from_file_location('accepted_glare256',PARENT/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
read,save,sha=m.read,m.save,m.sha
WORK=BASE/'step2000';SOURCE_CHECKPOINT=ROOT/'experiments/glare_500h_training_node11_gpu7_20261111/best.pt';CHECKPOINT=BASE/'checkpoint_step2000.pt'

def configure():
 m.BASE=BASE;m.SELF=SELF;m.STEP=2000;m.WORK=WORK
 m.native.WORK=WORK;m.native.SELF=SELF;m.core.WORK=WORK;m.core.SELF=SELF
 m.core.CHECKPOINT=CHECKPOINT;m.core.ARM='glare_step2000';m.core.MODEL='GLARE step2000（早期batch8）'
configure()

def prepare():
 import torch
 torch.set_num_threads(2);WORK.mkdir(parents=True,exist_ok=True)
 if (WORK/'PREPARED.json').exists():verify();return
 if not CHECKPOINT.exists():os.link(SOURCE_CHECKPOINT,CHECKPOINT)
 v=torch.load(CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);assert v['step']==2000
 # Compare parameter names, shapes and frozen assets with an already evaluated checkpoint.
 old=torch.load(ROOT/'experiments/glare_paper_settings_node11_gpu7_20261111/before_extension20epochs.pt',map_location='cpu',weights_only=False,mmap=True)
 assert set(v['trainable'])==set(old['trainable'])
 assert all(v['trainable'][k].shape==old['trainable'][k].shape and torch.isfinite(v['trainable'][k]).all() for k in v['trainable'])
 assert v['identity']['implementation']['assets_sha256']==old['identity']['implementation']['assets_sha256']
 save(BASE/'CHECKPOINT.json',dict(step=2000,source=str(SOURCE_CHECKPOINT),pinned=str(CHECKPOINT),sha256=sha(CHECKPOINT),snapshot='Hardlink preserves the original inode if latest.pt is atomically replaced; SHA checked before run and by each native worker',trainable_tensors=len(v['trainable']),training_identity=v['identity'],same_native_parameter_shapes_and_assets_as_10618=True))
 del v,old
 source=PARENT/'step10618';prior=read(source/'protocol.json');assert read(source/'COMPLETE.json')['success']
 p=copy.deepcopy(prior);m.link(source/'prosody',WORK/'prosody');_,cachekey=m.core.prosody_id()
 for s in p['samples']:
  assert sha(s['reference'])==s['reference_sha256'] and sha(s['audio'])==s['audio_sha256']
  assert m.core.cache_path(s,cachekey).exists()
  s['ready']=str(WORK/'inputs'/s['id']/'READY.json')
 c=read(PARENT/'cluster.json');c['nodes']=[n for n in c['nodes'] if n['name'] in ['node12','node11']];c.update(training_script=str(SELF),purpose='Queued GLARE early batch8 step2000 / matched test256 seed137 / after ablation6 test256 completion');save(BASE/'cluster.json',c)
 assert {n['name']:n['gpus'] for n in c['nodes']}=={n:list(range(8)) for n in ['node12','node11']}
 p.update(schema='glare-step2000-early-batch8-test256-v1',created=time.time(),step=2000,model=m.core.MODEL,checkpoint=str(CHECKPOINT),original_checkpoint=str(SOURCE_CHECKPOINT),checkpoint_sha256=read(BASE/'CHECKPOINT.json')['sha256'],entry_sha256=sha(SELF),reused_generated=0,total=256,seed=137,nfe=10,audio_guidance=2.,nodes=['node12','node11'],physical_gpu_count=16,training_variant='Early effective-batch8 independent training, not an intermediate checkpoint of the batch256 training line')
 p['code']={str(f):sha(f) for f in [SELF,PARENT/'run.py',ROOT/'scripts/eval_glare_seamless_test_20261112.py',ROOT/'scripts/eval_glare_final_20261112.py',ROOT/'scripts/glare_resident_inference.py',ROOT/'scripts/infer_glare_checkpoint.py',ROOT/'scripts/p2_pipelined_video_metrics.py']}
 assert len(p['samples'])==len({s['id'] for s in p['samples']})==256
 save(WORK/'protocol.json',p)
 for s in p['samples']:save(Path(s['ready']),dict(sample=s,source_only=True,protocol_sha256=sha(WORK/'protocol.json')))
 for i in range(24):
  part=p['samples'][i::24];save(WORK/'generation_shards'/f'{i}.json',part);save(WORK/'blocks'/f'{i:03d}.json',part)
 save(WORK/'PROSODY_COMPLETE.json',dict(clips=256,reused=True))
 save(WORK/'PREPARED.json',dict(success=True,at=time.time(),step=2000,total=256,new=256,reused=0,checkpoint_sha256=p['checkpoint_sha256'],same256_manifest_sha256=p['shared_items_sha256'],files={str(f):sha(f) for f in [BASE/'cluster.json',WORK/'protocol.json',BASE/'handoff.py',BASE/'run_two_nodes.py',*map(Path,p['code'])]}))
 print('Prepared step2000, exact256 seed137, cached sources/prosody reused; no GPU run',flush=True)

def verify():
 r=read(WORK/'PREPARED.json');assert r['success']
 assert all(sha(k)==v for k,v in r['files'].items())
 assert sha(CHECKPOINT)==r['checkpoint_sha256'];return r

def run():
 from run_two_nodes import run as two_nodes
 two_nodes()

def main():
 if len(sys.argv)>1 and sys.argv[1]=='metric_cli':m.metric_cli();return
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','worker','merge']);ap.add_argument('--step',type=int,choices=[2000],default=2000);ap.add_argument('--stage',choices=['generate','score']);ap.add_argument('--rank',type=int,default=0);ap.add_argument('--slot',type=int,default=0);a=ap.parse_args()
 try:
  if a.action=='prepare':prepare()
  elif a.action=='run':run()
  elif a.action=='worker':m.worker(a.stage,a.rank,a.slot)
  else:m.merge()
 except BaseException:
  save(WORK/f'FAILURE_{a.action}_{a.stage}_{a.slot}.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
