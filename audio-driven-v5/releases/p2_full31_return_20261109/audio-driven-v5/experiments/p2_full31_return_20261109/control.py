from pathlib import Path
import sys,copy,time,math,argparse,subprocess,concurrent.futures
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=W.parent/'p2_ltx_split24_8_batch31_20261109';sys.path[:0]=[str(W),str(ROOT/'src'),str(ROOT/'scripts')]
# Load the old helper under another name; never shadow its control module.
import importlib.util
spec=importlib.util.spec_from_file_location('split_control_helpers',S/'control.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
import runtime31 as R
OUT=W.parent/'p2_ltx_all_lora_full31_resumed_20261109';PARENT=W.parent/'p2_ltx_all_lora_full8_after6000_20261109'
def prepare_migrate():
 import torch
 torch.set_num_threads(8);C.activate()
 from p2_listen.contracts import digest
 from p2_listen.cache import atomic_torch
 old_controller=C.read(PARENT/'CONTROLLER.json')
 for _ in range(180):
  if not C.alive(old_controller):break
  time.sleep(5)
 else:raise TimeoutError('Wait for saved full8 exit; do not kill unsaved training')
 status=C.read(PARENT/'status.json');start=status['step'];path=PARENT/f'step_{start:06d}.pt';assert path.with_suffix('.complete.json').exists()
 assert not OUT.exists();OUT.mkdir();(OUT/'rank_logs').mkdir()
 old=C.read(PARENT/'run.json');c=copy.deepcopy(old['experiment']);c['output']=str(OUT);C.native('full','config').validate(c)
 C.write(OUT/'config.json',c);pre=copy.deepcopy(C.read(PARENT/'preflight.json'));pre.update(configuration_sha256=digest(c),source=R.source());C.write(OUT/'preflight.json',pre)
 # Reuse the previously stable full31 allocation and single-NIC fabric.
 cl=copy.deepcopy(C.read(W.parent/'p2_ltx_all_attention_ffn_lora31_20261108/cluster.json'));assert [n['gpus']for n in cl['nodes']]==[list(range(8))]*3+[list(range(1,8))]
 cl.update(master_port=29953,training_script=str(W/'worker.py'),training_config=str(OUT/'config.json'));cl['network']['NCCL_IB_HCA']='=mlx5_0:1';C.write(OUT/'cluster.json',cl)
 schedule=C.native('full','schedule').Schedule(c)
 import runtime as old_runtime
 for step in range(1,c['training']['steps']+1):
  pairs=schedule.indices(schedule.size,step,31,c['training']['seed'],0,1)
  before=[p for rank in range(8)for p in old_runtime.partition(pairs,rank,8)]
  after=[p for rank in range(31)for p in schedule.indices(schedule.size,step,31,c['training']['seed'],rank,31)]
  assert before==after==pairs
 identity=copy.deepcopy(old);identity.update(experiment=c,source=R.source(),world=31)
 normal=copy.deepcopy(identity);normal.update(source=old['source'],world=8);normal['experiment']['output']=old['experiment']['output'];assert normal==old
 cp=C.native('full','checkpoint');v=cp.checked(path);assert v['identity']==old and len(v['rank_rng'])==8
 payload=dict(v,identity=identity,rank_rng=[copy.deepcopy(v['rank_rng'][0])for _ in range(31)])
 dest=OUT/path.name;atomic_torch(dest,payload);C.write(dest.with_suffix('.complete.json'),dict(step=start,sha256=C.sha(dest)));new=cp.checked(dest)
 assert C.equal_tree(v['model'],new['model'])and C.equal_tree(v['optimizer'],new['optimizer'])and all(C.equal_tree(r,v['rank_rng'][0])for r in new['rank_rng'])
 C.write(OUT/'run.json',identity);C.write(OUT/'MIGRATION.json',dict(success=True,step=start,source=str(path),destination=str(dest),parent_world=8,world=31,model_and_adam_equal=True,rng_policy='Parent rank0 startup RNG cloned to31; per-example ordinal seeds and entire schedule unchanged',original_8_rng_preserved_at_source=True,all_schedule_pairs_equal=True,source_sha256=C.read(path.with_suffix('.complete.json'))['sha256'],destination_sha256=C.sha(dest)))
 (OUT/'metrics.jsonl').write_text('\n'.join(l for l in(PARENT/'metrics.jsonl').read_text().splitlines()if __import__('json').loads(l)['step']<=start)+'\n');C.write(OUT/'status.json',dict(step=start,budget=c['training']['steps'],complete=False))
 C.write(W/'READY.json',dict(success=True,source=R.source(),config_sha256=C.sha(OUT/'config.json'),cluster_sha256=C.sha(OUT/'cluster.json'),parent=str(PARENT),output=str(OUT),step=start,global_batch=31,all_schedule_pairs_equal=True))
 C.write(W/'STATE.json',dict(stage='migration_verified',step=start,at=time.time()));print('Migration verified',start,flush=True)
def launch():
 from listen_cluster import launch
 ready=C.read(W/'READY.json');assert ready['source']==R.source()and ready['config_sha256']==C.sha(OUT/'config.json')and ready['cluster_sha256']==C.sha(OUT/'cluster.json')
 cl=C.read(OUT/'cluster.json');commands=[]
 for i,n in enumerate(cl['nodes']):
  commands.append(['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v)for k,v in cl['network'].items()],cl['python'],'-u',str(W/'worker.py'),'node','--index',str(i),'--config',str(OUT/'config.json'),'--resume',C.read(OUT/'MIGRATION.json')['destination']])
 C.write(OUT/'CONTROLLER.json',C.record());C.write(W/'COMMANDS.json',commands);C.write(W/'STATE.json',dict(stage='starting_full31',step=ready['step'],at=time.time()))
 launch(cl,commands,OUT/'nodes')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare-migrate','launch']);a=p.parse_args();prepare_migrate()if a.action=='prepare-migrate'else launch()
