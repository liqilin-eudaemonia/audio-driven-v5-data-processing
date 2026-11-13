from pathlib import Path
import sys,copy,time,math,argparse,importlib.util
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=W.parent/'p2_ltx_split24_8_batch31_20261109';sys.path[:0]=[str(W),str(ROOT/'src'),str(ROOT/'scripts')]
spec=importlib.util.spec_from_file_location('noevent_split_helpers28',S/'control.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
import runtime28 as R
PARENT=W.parent/'p2_ltx_all_lora_noevent24_after6000_20261109';OUT=W.parent/'p2_ltx_all_lora_noevent28_batch28_20261109'
def prepare():
 from p2_listen.contracts import digest
 assert C.read(PARENT/'status.json')['step']==660 and (PARENT/'STOP').exists()
 c=C.read(PARENT/'config.json');C.activate();schedule=C.native('noevent','schedule').Schedule(c)
 p=dict(start_step=660,prefix_pairs=660*31,total_pairs=len(schedule.pairs),original_steps=c['training']['steps'],new_end_step=660+math.ceil((len(schedule.pairs)-660*31)/28),parent_config=str(PARENT/'config.json'),parent_config_sha256=C.sha(PARENT/'config.json'))
 C.write(W/'PLAN.json',p);c['batch28_continuation']=p;c['training']['global_batch']=28;c['training']['steps']=p['new_end_step'];c['output']=str(OUT);R.validate(c)
 assert not OUT.exists();OUT.mkdir();(OUT/'rank_logs').mkdir();C.write(OUT/'config.json',c)
 pre=copy.deepcopy(C.read(PARENT/'preflight.json'));pre.update(configuration_sha256=digest(c),source=R.source());C.write(OUT/'preflight.json',pre)
 cl=copy.deepcopy(C.read(W.parent/'p2_ltx_all_lora_full28_batch28_20261109/cluster.json'));cl.update(master_port=29981,training_script=str(W/'worker.py'),training_config=str(OUT/'config.json'));cl['nodes'][-1]['gpus']=[0,1,2,3]
 cl['nodes'][-1]['gpus']=[0,1,2,3]
 assert [n['name']for n in cl['nodes']]==['node12','node8','node4','node11'];assert sum(len(n['gpus'])for n in cl['nodes'])==28
 C.write(OUT/'cluster.json',cl);C.write(W/'READY.json',dict(source=R.source(),config_sha256=C.sha(OUT/'config.json'),cluster_sha256=C.sha(OUT/'cluster.json'),step=660,world=28,global_batch=28))
 print(p,flush=True)
def migrate():
 import torch
 torch.set_num_threads(8);C.activate()
 from p2_listen.cache import atomic_torch
 c=R.validate(C.read(OUT/'config.json'));old=C.read(PARENT/'run.json');identity=copy.deepcopy(old);identity.update(experiment=c,source=R.source(),world=28)
 normal=copy.deepcopy(identity);normal.update(source=old['source'],world=24);normal['experiment']=R.normalized(c);normal['experiment']['output']=old['experiment']['output'];assert normal==old
 assert all(R.source().get(k)==v for k,v in old['source'].items())
 cp=C.native('noevent','checkpoint');path=PARENT/'step_000660.pt';v=cp.checked(path);assert v['identity']==old and v['step']==660 and len(v['rank_rng'])==24
 payload=dict(v,identity=identity,rank_rng=[copy.deepcopy(v['rank_rng'][0])for _ in range(28)])
 dest=OUT/path.name;atomic_torch(dest,payload);C.write(dest.with_suffix('.complete.json'),dict(step=660,sha256=C.sha(dest)));new=cp.checked(dest)
 assert C.equal_tree(v['model'],new['model'])and C.equal_tree(v['optimizer'],new['optimizer'])and all(C.equal_tree(r,v['rank_rng'][0])for r in new['rank_rng'])
 C.write(OUT/'run.json',identity);C.write(OUT/'MIGRATION.json',dict(success=True,step=660,source=str(path),destination=str(dest),model_and_adam_equal=True,parent_world=24,world=28,rng_policy='Parent rank0 startup RNG cloned to28; original24 RNG preserved at source; same per-sample ordinal seeds',source_sha256=C.read(path.with_suffix('.complete.json'))['sha256'],destination_sha256=C.sha(dest)))
 (OUT/'metrics.jsonl').write_text('\n'.join(l for l in(PARENT/'metrics.jsonl').read_text().splitlines()if __import__('json').loads(l)['step']<=660)+'\n');C.write(OUT/'status.json',dict(step=660,budget=c['training']['steps'],complete=False))
 C.write(W/'STATE.json',dict(stage='migration_verified',step=660,at=time.time()));print('Model/Adam migration verified',flush=True)
def launch():
 from listen_cluster import launch
 ready=C.read(W/'READY.json');assert ready['source']==R.source()and ready['config_sha256']==C.sha(OUT/'config.json')and ready['cluster_sha256']==C.sha(OUT/'cluster.json')
 cl=C.read(OUT/'cluster.json');commands=[]
 for i,n in enumerate(cl['nodes']):commands.append(['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v)for k,v in cl['network'].items()],cl['python'],'-u',str(W/'worker.py'),'node','--index',str(i),'--config',str(OUT/'config.json'),'--resume',C.read(OUT/'MIGRATION.json')['destination']])
 C.write(OUT/'CONTROLLER.json',C.record());C.write(W/'COMMANDS.json',commands);C.write(W/'STATE.json',dict(stage='starting28',at=time.time(),step=660));launch(cl,commands,OUT/'nodes')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','migrate','launch']);a=p.parse_args();globals()[a.action]()
