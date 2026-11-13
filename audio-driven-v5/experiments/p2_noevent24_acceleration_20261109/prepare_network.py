from pathlib import Path
import sys,time,copy,statistics,concurrent.futures,subprocess,json
W=Path(__file__).resolve().parent;S=W.parent/'p2_ltx_split24_8_batch31_20261109';sys.path.insert(0,str(S))
import control as C,runtime
assert C.read(W/'STATE.json')['stage']=='eight_only_complete'
a=[r for r in C.read(W/'single_exact/RESULTS.json')['results']if r['kind']=='baseline'];b=[r for r in C.read(W/'eight_only/RESULTS.json')['results']if r['kind']=='baseline'];assert len(a)==len(b)==2
assert all(r['accepted_numerics']and all(x['bitwise_equal']for x in r['comparison'].values())for r in b)
assert max(r['mean_seconds']for r in b)<min(r['mean_seconds']for r in a)
assert all(abs(x-y)<1e-12 for r in b for x,y in zip(r['losses'],a[0]['losses']))
old=statistics.mean(r['mean_seconds']for r in a);new=statistics.mean(r['mean_seconds']for r in b);assert new<old*.98
out=C.path('noevent',True);cl=copy.deepcopy(C.read(out/'cluster.json'));cl['master_port']=29941;cl['network']['NCCL_IB_HCA']=C.read(W/'EIGHT_CLUSTER.json')['network']['NCCL_IB_HCA']
code="""import json,subprocess
print(json.dumps(dict(free_mib=[float(x)for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()],apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip())))
"""
with concurrent.futures.ThreadPoolExecutor(3)as pool:obs=list(pool.map(lambda n:dict(node=n['name'],**C.remote(cl,n,code)),cl['nodes']))
C.write(W/'PRE_NETWORK_LAUNCH_GPU.json',obs)
assert all(len(r['free_mib'])==8 and min(r['free_mib'])>135*1024 for r in obs),'Insufficient free memory; no foreign processes killed'
C.write(W/'NETWORK_PRODUCTION_CLUSTER.json',cl)
ready=dict(success=True,at=time.time(),single_means=[r['mean_seconds']for r in a],eight_means=[r['mean_seconds']for r in b],single_mean=old,eight_mean=new,time_reduction_percent=100*(1-new/old),throughput_gain_percent=100*(old/new-1),peak_gib=max(r['peak_gib']for r in b),config_sha256=C.sha(out/'config.json'),source=runtime.source_identity('noevent'),cluster_sha256=C.sha(W/'NETWORK_PRODUCTION_CLUSTER.json'),eight_results_sha256=C.sha(W/'eight_only/RESULTS.json'),single_results_sha256=C.sha(W/'single_exact/RESULTS.json'),checkpoint_sha256=C.read(out/'step_000224.complete.json')['sha256'],only_network_changed=True,training_code_unchanged=True,global_batch=31,numerics='FP64 SUM unchanged; within-network parameters and Adam bitwise equal; cross-network AV losses agree within1e-12')
C.write(W/'NETWORK_ADOPTION_READY.json',ready)
C.write(out/'ACTIVE_CLUSTER.json',dict(path=str(W/'NETWORK_PRODUCTION_CLUSTER.json'),sha256=ready['cluster_sha256'],note='Network-only eight-HCA adoption; original cluster.json retained as fallback reference'))
with(W/'network_resume.log').open('a')as f:p=subprocess.Popen([C.PY,'-u',str(W/'resume_network.py')],cwd=C.ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
C.write(W/'STATE.json',dict(at=time.time(),stage='resuming_eight_network_from224',launcher_pid=p.pid,training_code_unchanged=True))
print(json.dumps(dict(pid=p.pid,old_seconds=old,new_seconds=new,time_reduction_percent=ready['time_reduction_percent']),indent=2))
