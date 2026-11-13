"""Durable generation -> metrics -> full HTML publication, retaining failure state."""
import os,sys,time,subprocess,traceback,concurrent.futures
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';MP=read(ROOT/'configs/p2_omni_runtime.lock.json')['paths']['metrics_python'];ENV={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
def run(args,name,python=PY):
 with (OUT/(name+'.log')).open('a') as log:subprocess.run([str(python),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,env=ENV,check=True)
def state(phase,**kw):atomic_json(OUT/'status.json',dict(phase=phase,time=time.time(),**kw))
def main():
 while True:
  workers=[read(OUT/'workers'/f'{r:02d}.json') if (OUT/'workers'/f'{r:02d}.json').exists() else {} for r in range(24)]
  state('generating',completed=sum(len(w.get('results',[])) for w in workers),planned=384)
  if all(w.get('finished') for w in workers):break
  for r in range(3):
   if (OUT/f'node-{r}.json').exists() and any(read(OUT/f'node-{r}.json')['codes']):raise RuntimeError('Generation node failed '+str(r))
  time.sleep(5)
 state('preparing_metrics');run([OUT/'evaluate_cluster.py','prepare'],'prepare_metrics')
 state('evaluating');run([OUT/'evaluate_cluster.py','launch'],'launch_metrics')
 while not all((OUT/f'metric-node-{r}.json').exists() for r in range(3)):time.sleep(5)
 for r in range(3):
  codes=read(OUT/f'metric-node-{r}.json')['codes'];assert all(x['code']==0 if x['type']=='omni' else x['code'] in (0,2) for x in codes),codes
 state('aggregating');run([OUT/'evaluate_cluster.py','aggregate'],'aggregate',MP)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda kind:run([OUT/'evaluate_cluster.py','compare',kind],'compare_'+kind,MP),['train','test']))
 state('building_html');run([ROOT/'scripts/build_p2_av_review.py'],'build_html')
 state('complete',planned=384,completed=384)
if __name__=='__main__':
 if len(sys.argv)>1:
  with (OUT/'finish.log').open('a') as log:p=subprocess.Popen([str(PY),str(__file__)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
  print(p.pid)
 else:
  try:main()
  except Exception as error:state('failed',error=str(error));traceback.print_exc();raise
