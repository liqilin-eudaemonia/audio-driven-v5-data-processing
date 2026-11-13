"""Explicit staged64-dev evaluation and frozen trial decisions; no training launch."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
from p2_listen.contracts import ROOT,read,sha,digest,atomic_json
from .config import validate,source_identity
from .policy import decide


def fixed_reference_records(records,frozen):
 refs={r['id']:r for r in frozen}
 if len(refs)!=64 or len(records)!=64 or {r['id']for r in records}!=set(refs):raise ValueError('Fixed reference membership mismatch')
 result=[]
 for row in records:
  ref=refs[row['id']]
  if row['reference_identity']!=ref['reference_identity']or row['protocol_identity']!=ref['protocol_identity']:
   raise ValueError('Reference media or detector protocol changed across checkpoints')
  result.append(dict(row,reference=ref['reference']))
 return result


def freeze_reference_scores(c,step):
 from p2_event_presence.scoring import score
 out=Path(c['output'])/'evaluation'/f'step_{step:06d}'
 records=[json.loads(line)for line in (out/'events/event_pairs.jsonl').read_text().splitlines()if line.strip()]
 baseline=Path(c['output'])/'evaluation/step_000000/REFERENCE_EVENTS.json'
 if step==0:
  frozen=[{k:r[k]for k in ('id','reference_identity','protocol_identity','reference')}for r in records]
  if len(frozen)!=64:raise ValueError('Incomplete baseline reference annotations')
  atomic_json(baseline,frozen,immutable=True)
 else:
  verify_report(c,0)
  frozen=read(baseline)
 fixed=fixed_reference_records(records,frozen)
 (out/'event_pairs_fixed_reference.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in fixed))
 expected=[json.loads(line)for line in (out/'events/frozen_pairs.jsonl').read_text().splitlines()if line.strip()]
 atomic_json(out/'fixed_reference_event_summary.json',score(fixed,expected))
 return sha(baseline)


def run_command(command,log,cwd=None,env=None,allowed=(0,)):
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a')as stream:
  code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=cwd,env=env).returncode
 if code not in allowed:raise RuntimeError('Evaluation phase failed; see '+str(log))
 return code


def stage_contract(c,step):
 out=Path(c['output']);marker=read(out/f'step_{step:06d}.complete.json')
 return dict(step=step,checkpoint_sha256=marker['sha256'],configuration_sha256=digest(c),
  validation_sha256=c['validation_manifest_sha256'],policy_sha256=c['policy_sha256'],source=source_identity(),
  event_config_sha256=sha(c['evaluation']['event_config']),metric_assets_sha256=sha(c['evaluation']['metric_assets']))


def verify_report(c,step):
 out=Path(c['output'])/'evaluation'/f'step_{step:06d}';report=read(out/'REPORT.json')
 expected=stage_contract(c,step)
 if not report['success']or any(report.get(k)!=v for k,v in expected.items()):raise ValueError('Evaluation report identity mismatch')
 if read(out/'CONTRACT.json')!=expected:raise ValueError('Evaluation contract changed')
 for name,h in report['metrics'].items():
  if sha(out/name)!=h:raise ValueError('Scored evaluation evidence changed: '+name)
 if report.get('fixed_reference_sha256')is not None and report['fixed_reference_sha256']!=sha(Path(c['output'])/'evaluation/step_000000/REFERENCE_EVENTS.json'):
  raise ValueError('Frozen reference labels changed')
 if report['summary']!=derive_summary(c,step):raise ValueError('Evaluation summary differs from scored evidence')
 return report


def derive_summary(c,step):
 out=Path(c['output'])/'evaluation'/f'step_{step:06d}';plans=read(out/'planning_metrics.json')
 ids={r['sample_id']for r in read(c['validation_manifest'])}
 if len(plans)!=64 or {r['sample_id']for r in plans}!=ids:raise ValueError('Planning evaluation membership differs')
 video=read(out/'metrics/video/summary.json');audio=read(out/'metrics/audio/summary.json');events=read(out/'fixed_reference_event_summary.json')
 method=events['datasets']['stage2_dev']['methods']['stage2']
 f1=[method['full_dataset_draw_aggregate'][m]['macro_f1']['mean']for m in ('visual','audio')]
 reaction=sum(f1)/2 if all(v is not None for v in f1)else None
 acoustic=audio.get('activity',{}).get('acoustic',{})
 # Native listener suite may nest the audio summary under "audio".
 if not acoustic:acoustic=audio.get('audio',{}).get('activity',{}).get('acoustic',{})
 false_sound=acoustic.get('false_sound_clip_rate')
 lpips=video['metrics'].get('lpips');coverage=video['coverage'].get('lpips',{}).get('pairs',0)
 tokens=sum(r['planning_tokens']for r in plans)
 summary=dict(samples=64,planning_ce=sum(r['planning_sum']for r in plans)/max(tokens,1),planning_tokens=tokens,
  lpips=lpips,false_sound_rate=false_sound,format_error_rate=sum(r['format_error']for r in plans)/64,
  reaction_f1=reaction,reaction_f1_visual=f1[0],reaction_f1_audio=f1[1],
  complete=coverage==64 and not video.get('initialization_errors')and not audio.get('pairs_with_errors')and all(v is not None for v in (lpips,false_sound)),
  reaction_complete=reaction is not None,
  detector_based=True,detector_human_calibration_pending=True,
  interpretation='reaction category agreement with one observed reference, not context appropriateness or plan execution accuracy')
 from .plan_evidence import execution_metrics
 event_records=[json.loads(line)for line in (out/'event_pairs_fixed_reference.jsonl').read_text().splitlines()if line.strip()]
 summary['plan_execution']=execution_metrics(read(out/'plan_labels.json'),event_records)
 return summary


def summarize(c,step):
 out=Path(c['output'])/'evaluation'/f'step_{step:06d}'
 summary=derive_summary(c,step)
 receipt=stage_contract(c,step)
 baseline=Path(c['output'])/'evaluation/step_000000/REFERENCE_EVENTS.json'
 report=dict(success=True,at=time.time(),**receipt,summary=summary,fixed_reference_sha256=sha(baseline),
  metrics={os.path.relpath(p,out):sha(p)for p in [out/'planning_metrics.json',out/'metrics/video/summary.json',out/'metrics/audio/summary.json',out/'events/summary.json',out/'plan_labels.json',out/'events/event_pairs.jsonl',out/'pairs.jsonl',out/'fixed_reference_event_summary.json',out/'event_pairs_fixed_reference.jsonl',baseline]})
 atomic_json(out/'REPORT.json',report);return report


def evaluate(c,step,device='cuda:0'):
 validate(c)
 if step not in (0,100,300):raise ValueError('Only fixed trial milestones may be evaluated')
 out=Path(c['output'])/'evaluation'/f'step_{step:06d}';out.mkdir(parents=True,exist_ok=True)
 contract=stage_contract(c,step)
 # Existing detector configuration is pinned to the reserved node11 GPU7.
 # Use the same physical device for all inference/metric phases.
 event_policy=read(c['evaluation']['event_config'])
 if device!='cuda:0'or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(event_policy['physical_gpu']):
  raise ValueError('Run trial evaluation on node11 with CUDA_VISIBLE_DEVICES=7 and --device cuda:0')
 observed=subprocess.check_output(['nvidia-smi','-i',str(event_policy['physical_gpu']),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
 if observed[0].strip()!=event_policy['node11_gpu_uuid']or float(observed[1])<108*1024:
  raise RuntimeError('Reserved node11GPU7 not available; preserve existing jobs')
 with (out/'evaluation.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  atomic_json(out/'CONTRACT.json',contract,immutable=True)
  worker=ROOT/'scripts/p2_stage2_v2_eval_worker.py';base_env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
  base_env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse'),base_env.get('PYTHONPATH','')])
  for phase in ('qwen','gemma','whisper','render','decode','references'):
   marker=out/'stages'/(phase+'.json')
   if marker.exists():
    if read(marker)['contract_sha256']!=digest(contract):raise ValueError('Evaluation stage identity changed')
    continue
   atomic_json(out/'STATUS.json',dict(stage=phase,at=time.time(),complete=False))
   python=sys.executable
   env=dict(base_env)
   if phase=='references':env['CUDA_VISIBLE_DEVICES']=''
   run_command([python,str(worker),'--config',str(Path(c['output'])/'config.json'),'--step',str(step),'--phase',phase,'--device',device],out/(phase+'.log'),env=env)
   atomic_json(marker,dict(success=True,contract_sha256=digest(contract)))
  metric_python=c['evaluation']['metric_python'];omni=ROOT.parent/'OmniResponse'
  from p2_testsets_eval.cluster import environment
  metric_env=environment()
  metric_env['PYTHONPATH']=os.pathsep.join([metric_env['PYTHONPATH'],str(ROOT/'src'),str(ROOT/'scripts'),str(omni)])
  manifest=out/'pairs.jsonl';assets=c['evaluation']['metric_assets']
  for kind in ('video','audio'):
   dest=out/'metrics'/kind
   if (dest/'summary.json').exists():continue
   command=[metric_python,'-m','evaluation.video.paired_metrics.run'if kind=='video'else'evaluation.listener_metrics.run',
    '--manifest',str(manifest),'--assets',assets,'--output',str(dest)]
   if kind=='video':command+=['--device',device,'--threads','4','--batch-size','16','--metrics','psnr,ssim,lpips','--allow-partial']
   else:command+=['--stages','audio','--quality-kinds','utmos','--audio-device',device]
   run_command(command,out/('metrics_'+kind+'.log'),cwd=omni,env=metric_env,allowed=(0,2))
  events=out/'events';entry=ROOT/'scripts/p2_event_presence.py'
  if not (events/'contract.json').exists():run_command([sys.executable,str(entry),'bind','--manifest',str(manifest),'--work',str(events),'--method','stage2','--config',c['evaluation']['event_config']],out/'events_bind.log',env=base_env)
  if not (events/'summary.json').exists():run_command([sys.executable,str(entry),'run','--work',str(events)],out/'events_run.log',env=base_env,allowed=(0,2))
  if not (out/'plan_labels.json').exists():
   event_policy=read(c['evaluation']['event_config']);env=dict(base_env,CUDA_VISIBLE_DEVICES=str(event_policy['physical_gpu']))
   lib=str(Path(event_policy['inference_python']).parent.parent/'lib');env['LD_LIBRARY_PATH']=lib+':'+env.get('LD_LIBRARY_PATH','')
   run_command([event_policy['inference_python'],str(worker),'--config',str(Path(c['output'])/'config.json'),'--step',str(step),'--phase','plan-evidence'],out/'plan_evidence.log',env=env)
  if contract!=stage_contract(c,step):raise ValueError('Evaluation source/assets changed during execution')
  freeze_reference_scores(c,step)
  report=summarize(c,step)
  atomic_json(out/'STATUS.json',dict(stage='finished',at=time.time(),complete=report['summary']['complete']))
  return report


def decision(c,step):
 if step not in (100,300):raise ValueError('Decision only at100/300')
 validate(c)
 root=Path(c['output'])/'evaluation';base=verify_report(c,0);current=verify_report(c,step)
 for report,s in [(base,0),(current,step)]:
  if {k:report[k]for k in stage_contract(c,s)}!=stage_contract(c,s):raise ValueError('Decision report does not match frozen run')
 result=decide(base['summary'],current['summary'],read(c['policy']),step)
 if base['fixed_reference_sha256']!=current['fixed_reference_sha256']:raise ValueError('Different reference labels across checkpoints')
 result.update(at=time.time(),step=step,policy_sha256=c['policy_sha256'],report_sha256=sha(root/f'step_{step:06d}/REPORT.json'),baseline_report_sha256=sha(root/'step_000000/REPORT.json'))
 atomic_json(root/f'DECISION_{step:06d}.json',result);return result
