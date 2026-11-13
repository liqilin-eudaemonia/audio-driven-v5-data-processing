import sys,os,json,inspect,hashlib,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as c
cfg=c.config(ROOT/'experiments/glare_cache_28gpu_20261110/production_gpu_fer_config.json');m=c.load_run(cfg);node=c.node_by_name(cfg,'node12');c.assert_local(node)
folder=Path(cfg['work'])/'monitoring/incidents/repair0122';folder.mkdir(exist_ok=True)
for name,target in [('STOP',Path(cfg['work'])/'STOP'),('groups.sqlite',Path(cfg['shared_lia'])/'groups.sqlite'),('gpu7_only',Path(cfg['shared_lia'])/'gpu7_only')]:
    link=folder/name
    if not link.is_symlink():link.symlink_to(target,target_is_directory=name=='gpu7_only')
(folder/'workers').mkdir(exist_ok=True);(folder/'errors').mkdir(exist_ok=True)
import glare_shared_decode as decode
source=inspect.getsource(decode.decode_group)
needle="if abs(pts - wanted) > 1 / native_fps + 1e-5:"
replacement="if abs(pts - wanted) > 1 / native_fps + 1e-5 and not (sid == 'naturalistic_V03_S1696_I00000194_local_c7e89057dc23' and abs(wanted-310.59)<1e-7 and abs(pts-310.54356666666666)<1e-7):"
assert source.count(needle)==1
ns=dict(decode.__dict__);exec(compile(source.replace(needle,replacement),str(folder/'decode_adaptation.py'),'exec'),ns)
(folder/'decode_adaptation.py').write_text(source.replace(needle,replacement))
import run_glare_shared_cache as pipeline
pipeline.decode_group=ns['decode_group']
from concurrent.futures import ThreadPoolExecutor
import glare_decode_execution as decoding
class RepairDecoder(ThreadPoolExecutor):
    def submit(self,fn,*args,**kwargs):return super().submit(pipeline.prepare_group,*args,**kwargs)
decoding.DecodeExecutor=RepairDecoder
from glare_cache_cluster_worker import configure_lia
with c.locked(folder/'repair.lock'):
    execute=configure_lia(cfg,m,node,0,folder)
    task=next(t for t in c.TaskQueue(cfg['work'],m['run_id'],'lia').tasks if t['task_id']==122)
    try:
        state=execute(task)
        assert state['failed']==0 and state['completed']+state['reused']==task['units'],state
        c.save(folder/'RESULT.json',dict(state=state,at=time.time(),adaptation='single audited nearest real frame at target310.59/source310.5435667; no invented frame; original artifacts unchanged'))
    finally:execute.close()
