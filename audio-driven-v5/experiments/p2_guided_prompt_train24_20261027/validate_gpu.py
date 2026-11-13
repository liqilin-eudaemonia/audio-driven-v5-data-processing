import sys,json,subprocess,os,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
exps=json.loads((ROOT/'experiments/p2_train24_fourway_eval_20261027/experiments.json').read_text());ids=['naturalistic_V01_S0047_I00000498_local_24572dbe1845','naturalistic_V03_S1897_I00000023_local_14bd5941f383'];long_id='naturalistic_V03_S0261_I00000374_local_73a56e6ea9f8'
jobs=[dict(arm=a,sample_id=s,variant='guided',seed=137,name=a+'-'+str(i)) for a in ['E0','E1','E0F','E1F'] for i,s in enumerate(ids)]
jobs += [dict(arm='E0F',sample_id=long_id,variant='guided',seed=137,name='long-4096'),dict(arm='E0',sample_id=ids[0],variant='fixed',seed=137,name='fixed-replay')]
(OUT/'logs').mkdir(exist_ok=True)
if len(sys.argv)>1:
 rank=int(sys.argv[1]);results=[]
 for job in jobs[rank::8]:
  e=exps[job['arm']];command=[str(PY),str(ROOT/'scripts/p2_guided_prompt.py'),'generate','--work',str(OUT),'--run',e['run'],'--checkpoint',e['checkpoint'],'--sample-id',job['sample_id'],'--output',str(OUT/'validation_outputs'/job['name']),'--seed',str(job['seed']),'--device','cuda:0','--variant',job['variant'],'--restoration',str(ROOT/'experiments/p2_train24_fourway_eval_20261027/source_restoration.json')]
  with (OUT/'logs'/(job['name']+'.log')).open('a') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
  results.append(dict(**job,returncode=r.returncode));(OUT/f'worker-{rank}.json').write_text(json.dumps(dict(results=results,finished=False),indent=2))
  if r.returncode:break
 (OUT/f'worker-{rank}.json').write_text(json.dumps(dict(results=results,finished=True),indent=2));raise SystemExit(any(r['returncode'] for r in results))
else:
 (OUT/'validation_jobs.json').write_text(json.dumps(jobs,indent=2));children=[]
 for rank in range(8):
  used=int(subprocess.check_output(['nvidia-smi','-i',str(rank),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True));assert used<2048,(rank,used)
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(rank),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ALLOC_CONF':'expandable_segments:True'}
  with (OUT/'logs'/f'worker-{rank}.log').open('a') as log:p=subprocess.Popen([str(PY),str(__file__),str(rank)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,cwd=ROOT)
  children.append(dict(rank=rank,pid=p.pid))
 print(json.dumps(children))
