"""Cache -> generation -> metrics with durable state; all English guided outputs."""
import os,sys,time,subprocess,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
MP=read(ROOT/'configs/p2_omni_runtime.lock.json')['paths']['metrics_python']
ENV={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
def run(args,name,python=PY):
 with (OUT/(name+'.log')).open('a') as log:subprocess.run([str(python),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,env=ENV,check=True)
def state(phase,**kw):atomic_json(OUT/'status.json',dict(phase=phase,time=time.time(),**kw))
def wait_nodes(prefix):
 while True:
  nodes=[OUT/f'{prefix}{i}.json' for i in range(2)]
  for p in nodes:
   if p.exists():
    codes=read(p)['codes'];assert all((c['code'] if isinstance(c,dict) else c)==0 for c in codes),(p,codes)
  if all(p.exists() for p in nodes):return
  time.sleep(5)
def main():
 state('encoding_english_text');wait_nodes('cache-node-')
 state('generating',planned=384);run([OUT/'generate_cluster.py','launch'],'launch_generation')
 while True:
  workers=[read(OUT/'workers'/f'{r:02d}.json') if (OUT/'workers'/f'{r:02d}.json').exists() else {} for r in range(16)]
  state('generating',completed=sum(len(w.get('results',[])) for w in workers),planned=384)
  if all(w.get('finished') for w in workers):break
  for r in range(2):
   if (OUT/f'node-{r}.json').exists():assert not any(read(OUT/f'node-{r}.json')['codes'])
  time.sleep(5)
 state('preparing_metrics');run([OUT/'evaluate_cluster.py','prepare'],'prepare_metrics')
 state('evaluating');run([OUT/'evaluate_cluster.py','launch'],'launch_metrics')
 run([OUT/'reaction_eval.py','launch'],'launch_reactions')
 wait_nodes('metric-node-')
 state('aggregating_av');run([OUT/'evaluate_cluster.py','aggregate'],'aggregate',MP)
 atomic_json(OUT/'av_complete.json',dict(complete=True))
 while not (OUT/'reactions_complete.json').exists():
  p=OUT/'reactions_status.json'
  if p.exists() and read(p)['phase']=='failed':raise RuntimeError(read(p)['error'])
  time.sleep(5)
 state('publishing');run([OUT/'publish.py'],'publish',MP)
 run([ROOT/'scripts/build_p2_av_review.py'],'build_html')
 state('validating_published_page')
 view=ROOT/'experiments/p2_av_review_20261027'
 browser_env={**ENV,'LD_LIBRARY_PATH':'/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu','FONTCONFIG_FILE':'/tmp/p2_review_fonts.conf'}
 for script in ['browser_english_check.py','browser_full_guided_check.py']:
  with (OUT/(script+'.log')).open('a') as log:subprocess.run(['/tmp/p2-review-qa/bin/python',str(view/script)],env=browser_env,stdout=log,stderr=subprocess.STDOUT,check=True)
 run([view/'validate_full_media.py'],'validate_media')
 state('complete',outputs=384,english_only=True)
if __name__=='__main__':
 if len(sys.argv)>1:
  with (OUT/'finish.log').open('a') as log:p=subprocess.Popen([str(PY),str(__file__)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True,env=ENV)
  print(p.pid)
 else:
  try:main()
  except Exception as error:state('failed',error=str(error));traceback.print_exc();raise
