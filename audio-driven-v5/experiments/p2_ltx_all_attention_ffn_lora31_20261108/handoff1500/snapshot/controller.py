"""Checkpoint-gated handoff, GPU acceptance, strict resume and monitoring."""
from pathlib import Path
import fcntl,importlib.util,json,math,os,statistics,subprocess,sys,time,traceback
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
OUT=ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108';HERE=OUT/'handoff1500'
OLD=ROOT/'experiments/p2_current_component_ablations31_20261108/no_event'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
CLI=ROOT/'scripts/p2_ltx_all_lora_node11shift_20261108.py'

def state(stage,**kw):
 v=dict(stage=stage,at=time.time(),pid=os.getpid(),**kw)
 atomic_json(HERE/'STATE.json',v);print(json.dumps(v),flush=True)

def cancel():
 if (HERE/'CANCEL').exists():raise RuntimeError('Handoff explicitly cancelled')

def helpers():
 p=OLD.parent/'sequence_v2.py';s=importlib.util.spec_from_file_location('handoff_helpers',p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def verify(folder,step,old):
 cmd=[sys.executable,str(Path(__file__).resolve()),'verify',str(folder),str(step),'old'if old else'new']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 with (HERE/f'verify_{folder.name}_{step}.log').open('w')as f:
  p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 if p.returncode:raise RuntimeError('Full checkpoint verification failed')
 return read(HERE/f'CHECKPOINT_{folder.name}_{step}_VERIFIED.json')

def verify_main(folder,step,kind):
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import torch
 torch.set_num_threads(2)
 if kind=='old':from p2_current_component_ablation.checkpoint import checked
 else:from p2_ltx_all_lora.checkpoint import checked
 p=folder/f'step_{step:06d}.pt';v=checked(p)
 assert v['step']==step and v['identity']==read(folder/'run.json')
 assert v['identity']['world']==31 and len(v['rank_rng'])==31
 assert {int(s['step'])for s in v['optimizer']['state'].values()if'step'in s}=={step}
 for s in v['optimizer']['state'].values():
  for t in s.values():
   if torch.is_tensor(t)and not torch.isfinite(t).all():raise ValueError('Nonfinite Adam state')
 for r in v['rank_rng']:
  assert isinstance(r,dict)and r
 assert not torch.cuda.is_initialized()
 atomic_json(HERE/f'CHECKPOINT_{folder.name}_{step}_VERIFIED.json',dict(success=True,step=step,path=str(p),sha256=sha(p),bytes=p.stat().st_size,model_adam_finite=True,rank_rng=31,at=time.time()))

def wait_idle():
 import p2_ltx_all_lora_node11shift_20261108 as cli
 cli.verify()
 from p2_whisper_event_feature_train31 import selected_resources
 for attempt in range(120):
  cancel();r=selected_resources(read(OUT/'cluster.json'));atomic_json(HERE/'RESOURCES.json',r)
  if r['selected_all_idle']:return
  state('waiting_selected31_idle');time.sleep(10)
 raise RuntimeError('Selected GPUs did not become idle; preserve other tasks')

def launch(debug):
 cancel();wait_idle()
 phase='debug2'if debug else'formal'
 cmd=[sys.executable,str(CLI),'launch','--logs',str(HERE/(phase+'_nodes'))]
 if debug:cmd+=['--stop-after','2']
 else:cmd+=['--resume',str(OUT/'step_000002.pt')]
 f=(HERE/(phase+'.log')).open('a')
 p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 atomic_json(HERE/(phase+'_CONTROLLER.json'),dict(pid=p.pid,command=cmd,at=time.time()))
 return p,f

def report_metrics(min_step):
 path=OUT/'metrics.jsonl'
 if not path.exists():return None
 with path.open('rb')as f:
  f.seek(max(0,path.stat().st_size-250000));lines=f.read().splitlines()
 rows=[]
 for line in lines:
  try:rows.append(json.loads(line))
  except ValueError:pass
 if not rows or rows[-1]['step']<min_step:return None
 for row in rows[-30:]:
  for k in ['av','av_video','av_audio','event','feature_video','feature_audio','weighted_total']:
   if not math.isfinite(row[k]):raise ValueError('Nonfinite '+k)
  expected=row['av']+.05*row['event']+.025*(row['feature_video']+row['feature_audio'])
  if abs(expected-row['weighted_total'])>1e-8:raise ValueError('Incorrect full loss')
  for k in ['connector','events','features','ltx_lora','whisper']:
   if not math.isfinite(row['grad_norm'][k])or row['grad_norm'][k]<=0:raise ValueError('Inactive gradient '+k)
 last=rows[-30:]
 v=dict(success=True,at=time.time(),step=rows[-1]['step'],window=len(last),mean_step_seconds=statistics.mean(r['seconds']for r in last),peak_rank0_reserved_gib=max(r['peak_reserved_gib']for r in last),loss_and_gradients_valid=True)
 atomic_json(HERE/'PERFORMANCE.json',v);return v

def run():
 with (HERE/'controller.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  atomic_json(HERE/'CONTROLLER.json',dict(pid=os.getpid(),start_ticks=Path('/proc',str(os.getpid()),'stat').read_text().split()[21],at=time.time()))
  h=helpers();oldcontroller=read(OLD/'CONTROLLER.json')
  state('waiting_old1500',current_step=read(OLD/'status.json')['step'])
  while not(HERE/'STOP_ARMED.json').exists():
   cancel()
   if(HERE/'STOP_ERROR.json').exists():raise RuntimeError(read(HERE/'STOP_ERROR.json'))
   if not h.alive(oldcontroller):raise RuntimeError('Old controller exited before target')
   time.sleep(1)
  state('waiting_old1500_complete_exit')
  while True:
   cancel()
   if read(OLD/'status.json')['step']!=1500:raise ValueError('Expected exact1500')
   if not h.alive(oldcontroller)and h.workers_exited(OLD):break
   time.sleep(5)
  old=verify(OLD,1500,True);atomic_json(HERE/'OLD_STOPPED_ACCEPTED.json',old)
  # Training code and accepted config must remain unchanged.
  accepted=read(OUT/'CODE_ACCEPTED.json')
  assert sha(OUT/'config.json')==accepted['configuration_sha256']
  assert all(sha(p)==v for p,v in accepted['source'].items())
  state('starting_gpu_debug')
  p,f=launch(True)
  while p.poll()is None:cancel();time.sleep(5)
  f.close()
  if p.returncode:raise RuntimeError('GPU debug failed; formal training blocked')
  longest=read(OUT/'LONGEST_SAMPLE_CHECK.json')
  assert longest['success']and longest['updates_restored']and longest['rng_restored']
  assert longest['max_rank_peak_reserved_gib']<=134
  assert read(OUT/'status.json')['step']==2
  assert h.workers_exited(OUT)
  m=report_metrics(2);assert m and m['step']==2
  checkpoint=verify(OUT,2,False)
  atomic_json(HERE/'GPU_DEBUG_ACCEPTED.json',dict(success=True,longest=longest,metrics=m,checkpoint=checkpoint,world=31,reserved_gpu='node11:0',at=time.time()))
  state('starting_formal_from_verified_step2')
  p,f=launch(False);accepted_formal=False
  while p.poll()is None:
   cancel();m=report_metrics(3)
   if m:
    if not accepted_formal:
     atomic_json(HERE/'FORMAL_STARTED_ACCEPTED.json',dict(success=True,at=time.time(),metrics=m,strict_resume=2,world=31,reserved_gpu='node11:0',budget=9169,save_every=500));accepted_formal=True
    state('formal_training',**{'performance':m})
   time.sleep(30)
  f.close()
  if p.returncode:raise RuntimeError('Formal training exited with error')
  state('formal_exited',status=read(OUT/'status.json'))

if __name__=='__main__':
 if len(sys.argv)>1:verify_main(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
 else:
  try:run()
  except Exception:
   atomic_json(HERE/'ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()));state('blocked');raise
