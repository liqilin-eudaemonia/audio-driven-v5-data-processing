from pathlib import Path
import sys,copy,time,argparse
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=W.parent/'p2_ltx_split24_8_batch31_20261109';sys.path[:0]=[str(W),str(S),str(ROOT/'src'),str(ROOT/'scripts')]
import control as C,batch_runtime as R
OUT=W.parent/'p2_ltx_all_lora_noevent_batch24_20261109'
def migrate():
 import torch
 torch.set_num_threads(8);C.activate()
 from p2_listen.cache import atomic_torch
 results=C.read(W/'matched744/RESULTS.json')['results'];a,b=results
 assert a['batch']==31 and b['batch']==24 and a['samples']==b['samples']==744
 assert b['samples_per_second']>a['samples_per_second']*1.05 and b['finite_updates']
 boundary=C.read(W/'BOUNDARY.json');parent=Path(boundary['parent']);old=C.read(parent/'run.json');c=R.validate(C.read(OUT/'config.json'));identity=copy.deepcopy(old);identity.update(experiment=c,source=R.source())
 normal=copy.deepcopy(identity);normal['source']=old['source'];normal['experiment']=R.normalized(c);normal['experiment']['output']=old['experiment']['output'];assert normal==old
 assert C.read(W/'READY.json')['source']==R.source()
 cp=C.native('noevent','checkpoint');v=cp.checked(boundary['checkpoint']);assert v['identity']==old and v['step']==boundary['step']and len(v['rank_rng'])==24
 C.write(OUT/'run.json',identity);dest=OUT/Path(boundary['checkpoint']).name
 atomic_torch(dest,dict(v,identity=identity));C.write(dest.with_suffix('.complete.json'),dict(step=v['step'],sha256=C.sha(dest)));new=cp.checked(dest)
 assert C.equal_tree(v['model'],new['model'])and C.equal_tree(v['optimizer'],new['optimizer'])and C.equal_tree(v['rank_rng'],new['rank_rng'])
 C.write(OUT/'MIGRATION.json',dict(success=True,step=v['step'],source=boundary['checkpoint'],destination=str(dest),model_adam_all24_rng_equal=True,source_sha256=boundary['sha256'],destination_sha256=C.sha(dest),sample_prefix_pairs=c['batch24_continuation']['prefix_pairs']))
 (OUT/'metrics.jsonl').write_text('\n'.join(l for l in(parent/'metrics.jsonl').read_text().splitlines()if __import__('json').loads(l)['step']<=v['step'])+'\n')
 C.write(OUT/'status.json',dict(step=v['step'],budget=c['training']['steps'],complete=False))
 C.write(W/'STATE.json',dict(at=time.time(),stage='migration_verified',step=v['step']))
 print('Migration verified',v['step'],flush=True)
def launch():
 from listen_cluster import launch,distributed_commands
 ready=C.read(W/'READY.json');assert ready['source']==R.source()and ready['config_sha256']==C.sha(OUT/'config.json')and ready['cluster_sha256']==C.sha(OUT/'cluster.json')
 migration=C.read(OUT/'MIGRATION.json');assert migration['success']
 cl=C.read(OUT/'cluster.json');C.write(OUT/'CONTROLLER.json',C.record());C.write(W/'PRODUCTION_CONTROLLER.json',C.record())
 cmds=distributed_commands(cl,W/'worker.py',['--config',str(OUT/'config.json'),'--resume',migration['destination']]);C.write(W/'COMMANDS.json',cmds)
 C.write(W/'STATE.json',dict(at=time.time(),stage='batch24_starting',step=migration['step']))
 launch(cl,cmds,OUT/'nodes')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['migrate','launch']);a=p.parse_args()
 migrate()if a.action=='migrate'else launch()
