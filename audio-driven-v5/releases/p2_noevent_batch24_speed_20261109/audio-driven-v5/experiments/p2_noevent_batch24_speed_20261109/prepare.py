from pathlib import Path
import sys,copy,math,argparse
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=W.parent/'p2_ltx_split24_8_batch31_20261109';sys.path[:0]=[str(W),str(S),str(ROOT/'src'),str(ROOT/'scripts')]
import control as C
OUT=W.parent/'p2_ltx_all_lora_noevent_batch24_20261109'
def prepare(start):
 from p2_listen.contracts import digest
 import batch_runtime as R
 parent=C.path('noevent',True);c=C.read(parent/'config.json');C.activate();original=C.native('noevent','schedule').Schedule(c);total=len(original.pairs)
 plan=dict(start_step=start,prefix_pairs=start*31,total_pairs=total,original_steps=c['training']['steps'],new_end_step=start+math.ceil((total-start*31)/24),equivalent_old6000_step=start+math.ceil((6000-start)*31/24),parent_config=str(parent/'config.json'),parent_config_sha256=C.sha(parent/'config.json'))
 C.write(W/'PLAN.json',plan);c['batch24_continuation']=plan;c['training']['global_batch']=24;c['training']['steps']=plan['new_end_step'];c['output']=str(OUT);R.validate(c)
 OUT.mkdir(exist_ok=True);(OUT/'rank_logs').mkdir(exist_ok=True);assert not(OUT/'run.json').exists();C.write(OUT/'config.json',c)
 pre=copy.deepcopy(C.read(parent/'preflight.json'));pre.update(configuration_sha256=digest(c),source=R.source());C.write(OUT/'preflight.json',pre)
 cl=C.read(parent/'ACTIVE_CLUSTER.json');cl=copy.deepcopy(C.read(cl['path']));cl.update(master_port=29947,training_config=str(OUT/'config.json'),training_script=str(W/'worker.py'));C.write(OUT/'cluster.json',cl)
 C.write(W/'READY.json',dict(source=R.source(),config_sha256=C.sha(OUT/'config.json'),cluster_sha256=C.sha(OUT/'cluster.json'),parent=str(parent),output=str(OUT)))
 return c
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);a=p.parse_args();print(prepare(a.step)['batch24_continuation'])
