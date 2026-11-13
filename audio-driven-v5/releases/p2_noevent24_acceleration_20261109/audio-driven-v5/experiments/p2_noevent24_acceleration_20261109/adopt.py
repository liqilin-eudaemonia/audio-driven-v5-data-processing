from pathlib import Path
import sys,json,copy,os,time,subprocess,argparse
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=ROOT/'experiments/p2_ltx_split24_8_batch31_20261109'
sys.path[:0]=[str(W),str(S),str(ROOT/'src'),str(ROOT/'scripts')]
import control as C
OUT=ROOT/'experiments/p2_ltx_all_lora_noevent24_fast_20261109'

def prepare(policy):
 import adapt_runtime as R
 from p2_listen.contracts import digest
 C.activate();parent=C.path('noevent',True);old=C.read(parent/'run.json');c=copy.deepcopy(old['experiment']);c['output']=str(OUT)
 assert C.read(parent/'status.json')['step']==224
 assert not OUT.exists(),'Independent adoption output must be new'
 OUT.mkdir();(OUT/'rank_logs').mkdir();C.write(W/'ADOPTED_POLICY.json',policy)
 original=(S/'training_noevent.py').read_text();text=original
 line="optimizer=torch.optim.AdamW(groups,weight_decay=0.)"
 assert text.count(line)==1
 text=text.replace(line,"optimizer=torch.optim.AdamW(groups,weight_decay=0.,fused="+repr(True if policy['fused']else None)+")")
 line="start=checkpoint.resume(resume_path,model,optimizer,identity,rank) if resume_path else 0"
 assert text.count(line)==1
 text=text.replace(line,line+'\n        optimizer_flags(optimizer)')
 oldnorm="norms={g['name']:sum(float(p.grad.double().square().sum()) for p in g['params'] if p.grad is not None)**.5 for g in optimizer.param_groups}"
 assert text.count(oldnorm)==1
 if policy['diagnostics']:text=text.replace(oldnorm,"norms=fast_norms(optimizer.param_groups)")
 compile(text,'training_adopted.py','exec');(W/'training_adopted.py').write_text(text)
 C.native('noevent','config').validate(c);C.write(OUT/'config.json',c)
 pre=copy.deepcopy(C.read(parent/'preflight.json'));pre.update(configuration_sha256=digest(c),source=R.source());C.write(OUT/'preflight.json',pre)
 identity=copy.deepcopy(old);identity.update(experiment=c,source=R.source())
 assert all(identity['source'].get(k)==v for k,v in old['source'].items())
 C.write(OUT/'run.json',identity)
 cl=copy.deepcopy(C.read(parent/'cluster.json'));cl.update(training_script=str(W/'worker.py'),training_config=str(OUT/'config.json'),master_port=29941);cl['network']['NCCL_IB_HCA']=policy['hca'];C.write(OUT/'cluster.json',cl)
 C.write(W/'ADOPTION_READY.json',dict(source=R.source(),config_sha256=C.sha(OUT/'config.json'),cluster_sha256=C.sha(OUT/'cluster.json'),parent=str(parent),output=str(OUT),policy=policy))
 return identity

def migrate():
 import torch
 torch.set_num_threads(8)
 from p2_listen.cache import atomic_torch
 C.activate();parent=C.path('noevent',True);source=parent/'step_000224.pt';dest=OUT/source.name
 cp=C.native('noevent','checkpoint');v=cp.checked(source);old=C.read(parent/'run.json');identity=C.read(OUT/'run.json')
 assert v['identity']==old and v['step']==224 and len(v['rank_rng'])==24
 normalized=copy.deepcopy(identity);normalized['source']=old['source'];normalized['experiment']['output']=old['experiment']['output'];assert normalized==old
 atomic_torch(dest,dict(v,identity=identity));C.write(dest.with_suffix('.complete.json'),dict(step=224,sha256=C.sha(dest)));new=cp.checked(dest)
 assert C.equal_tree(v['model'],new['model'])and C.equal_tree(v['optimizer'],new['optimizer'])and C.equal_tree(v['rank_rng'],new['rank_rng'])
 C.write(OUT/'MIGRATION.json',dict(success=True,step=224,world=24,source=str(source),destination=str(dest),model_and_adam_equal=True,all24_rng_equal=True,source_sha256=C.sha(source),destination_sha256=C.sha(dest)))
 C.write(OUT/'status.json',dict(step=224,budget=9169,complete=False))
 (OUT/'metrics.jsonl').write_text('\n'.join(l for l in(parent/'metrics.jsonl').read_text().splitlines()if json.loads(l)['step']<=224)+'\n')

def launch():
 from listen_cluster import launch as go,distributed_commands
 import adapt_runtime as R
 ready=C.read(W/'ADOPTION_READY.json');assert ready['source']==R.source()and ready['config_sha256']==C.sha(OUT/'config.json')and ready['cluster_sha256']==C.sha(OUT/'cluster.json')
 cl=C.read(OUT/'cluster.json');cmds=distributed_commands(cl,W/'worker.py',['--config',str(OUT/'config.json'),'--resume',str(OUT/'step_000224.pt')])
 C.write(OUT/'CONTROLLER.json',C.record());C.write(OUT/'COMMANDS.json',cmds)
 go(cl,cmds,OUT/'nodes')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','migrate','launch']);p.add_argument('--policy');a=p.parse_args()
 if a.action=='prepare':prepare(C.read(a.policy))
 elif a.action=='migrate':migrate()
 else:launch()
