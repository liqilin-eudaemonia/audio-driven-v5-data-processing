"""Run the existing independent clip-presence detector on released node12GPU7."""
from pathlib import Path
import sys,os,json,subprocess,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json,sha
from reference_media_cache import bind as reuse_reference_media,remember as remember_reference_media

def run(work):
 p=read(work/'protocol.json');cl=read(work/'cluster.json')
 assignment=p.get('scoring_device')
 if assignment:
  from gpu7_score import validate_device
  validate_device(p)
  n=next(n for n in cl['nodes']if n['host']==assignment['host'])
 else:n=cl['nodes'][0]
 if n['name']!=('node11'if assignment else'node12') or n['host']not in subprocess.check_output(['hostname','-I'],text=True).split():raise ValueError('Wrong event detection host')
 if not(work/'FINAL_AUDIT.json').exists():raise ValueError('Finish generation and native scoring before event detection')
 # The inherited detector pins physical7. Rebind only its recorded UUID to the
 # selected training allocation on node12; preserve reserved node11GPU4-7.
 gpu=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
 cfg=read(ROOT/'configs/p2_event_presence_v1.json');cfg.update(node11_gpu_uuid=gpu[0].strip(),physical_gpu=7,authorized_scope='node11GPU7 alongside training'if assignment else'node12GPU7 after all28 generation workers exited; node11 reserved cards untouched')
 atomic_json(work/'event_detector_config.json',cfg,immutable=True)
 arm=p['arms'][0]['name'];manifest=work/'event_pairs_manifest.jsonl'
 rows=[]
 for rank in range(p['workers']):
  f=work/'metric_manifests'/arm/'seamless'/f'{rank:02d}.jsonl'
  rows.extend(json.loads(x)for x in f.read_text().splitlines()if x.strip())
 if len(rows)!=p['total_samples']or len({r['id']for r in rows})!=len(rows):raise ValueError('Missing or duplicate enriched event pairs')
 if any(not r.get('generated_audio')or not Path(r['generated_audio']).is_file()for r in rows):raise ValueError('Generated own WAV must be present; do not score mute video as silence')
 text=''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows)
 if manifest.exists()and manifest.read_text()!=text:raise ValueError('Event input pairs changed')
 manifest.write_text(text)
 entry=ROOT/'scripts/p2_event_presence.py';events=work/'events';env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 def command(args,log,allowed=(0,),python=None,stage=None):
  python=python or sys.executable;runtime_env=dict(env,MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='7'if stage=='infer'else'')
  library=Path(python).parent.parent/'lib'
  if library.is_dir():runtime_env['LD_LIBRARY_PATH']=str(library)+':'+runtime_env.get('LD_LIBRARY_PATH','')
  with log.open('a')as stream:v=subprocess.run([python,str(entry),*args],cwd=ROOT,env=runtime_env,stdout=stream,stderr=subprocess.STDOUT)
  if v.returncode not in allowed:raise RuntimeError('Event evaluation failed: '+str(log))
 if not(events/'contract.json').exists():command(['bind','--manifest',str(manifest),'--work',str(events),'--method',arm,'--config',str(work/'event_detector_config.json')],work/'events_bind.log')
 cache=Path(__file__).resolve().parent/'reference_media_cache'
 if not(events/'summary.json').exists():
  # Same native stages and environments as its run command, with a content-
  # verified GT reuse point between preparation and candidate/detector work.
  command(['prepare','--work',str(events)],events/'prepare.log',python=cfg['media_python'],stage='prepare')
  reuse_reference_media(events,cache)
  for stage,python in [('candidates',sys.executable),('infer',cfg['inference_python']),('score',sys.executable)]:
   atomic_json(events/'RUN_STATUS.json',dict(stage=stage,status='running',pid=os.getpid()))
   command([stage,'--work',str(events)],events/(stage+'.log'),(0,2)if stage=='score'else(0,),python,stage)
 summary=read(events/'summary.json')
 if p['checkpoint_identity']['experiment']['loss_search']['candidate']=='baseline':remember_reference_media(events,cache)
 atomic_json(events/'RUN_STATUS.json',dict(stage='finished',status='execution_complete'if summary['status']=='complete'else'execution_complete_partial',summary_sha256=sha(events/'summary.json')))
 atomic_json(work/'EVENTS_COMPLETE.json',dict(summary_sha256=sha(events/'summary.json'),status=summary['status'],detector_based=True,human_calibrated=False,pilot_only=True,independent_of_training_event_head=True))
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();run(a.work.resolve())
