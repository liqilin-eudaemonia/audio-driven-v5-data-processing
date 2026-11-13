"""Hourly training record; after complete1000, same64-dev latest-v3 evaluation."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time,statistics
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W));import runtime as R
sys.path.insert(0,str(W));import eval28 as E
from p2_listen.contracts import read,sha,atomic_json
PY='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'

def result():
 from p2_event_presence.scoring import score
 receipt=read(R.RUN/'CONTINUATION_ACCEPTED.json');paths={0:R.PARENT/'run/evaluation/step_000000',100:R.PARENT/'run/evaluation/step_000100',1000:R.RUN/'evaluation/step_001000'}
 rows=[];expected=[];summary={}
 for step,out in paths.items():
  report=read(out/'REPORT.json')
  if step in(0,100)and sha(out/'REPORT.json')!=receipt['parent_latest_reports'][str(step)]:raise ValueError('Parent score changed')
  if step==1000:E.E.verify_report(read(R.RUN/'config.json'),step)
  for name,h in report['metrics'].items():
   if sha(out/name)!=h:raise ValueError('Scoring evidence changed')
  for row in E.values(out/'event_pairs_fixed_reference.jsonl'):rows.append(dict(row,method=f'step{step}'))
  for row in E.values(out/'events/frozen_pairs.jsonl'):expected.append(dict(row,method=f'step{step}'))
  video=read(out/'metrics/video/summary.json')['metrics'];audio=read(out/'metrics/audio/summary.json')['audio']['activity']['acoustic']
  summary[str(step)]=dict(report['summary'],video=video,audio_activity=audio)
 scored=score(rows,expected);dataset=scored['datasets']['stage2_dev']
 for step in paths:
  m=dataset['methods'][f'step{step}']['common_subset_draw_aggregate']
  summary[str(step)]['common_known_visual_f1']=m['visual']['macro_f1']['mean'];summary[str(step)]['common_known_audio_f1']=m['audio']['macro_f1']['mean']
 atomic_json(W/'FINAL_COMPARISON.json',dict(success=True,at=time.time(),steps=summary,common_support_events=scored,
  comparison='Same64 independentdev, seed137, latest-v3 scorer; baseline is reused generation, not retrained',
  limitation='Common-known event F1 excludes unknown/error; one generation seed, no human calibration or statistical-significance claim'))
 lines=['# 联合微调：0／100／1000步比较','','同64条独立dev、seed137、最新版v3评分；训练数据、损失与原始100步接续一致。','','| 联合步数 | LPIPS↓ | SSIM↑ | PSNR↑ | 共同可判定视觉F1↑ | 发声F1↑ | 规划CE↓ |','|---|---|---|---|---|---|---|']
 def fmt(x):return 'N/A'if x is None else f'{x:.4f}'
 for step,r in summary.items():lines.append('| '+' | '.join([step,fmt(r['lpips']),fmt(r['video']['ssim']),fmt(r['video']['psnr']),fmt(r['common_known_visual_f1']),fmt(r['common_known_audio_f1']),fmt(r['planning_ce'])])+' |')
 lines+=['','事件F1使用三组完全相同的可判定样本／类别交集，unknown/error不作负例；此小规模单seed比较不能证明长期最优。完整计数、覆盖与误发声／漏回应见FINAL_COMPARISON.json。']
 (W/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
 last=0
 while True:
  if(R.RUN/'STOP').exists():atomic_json(W/'FINISH_STATUS.json',dict(stage='user_stopped',at=time.time()));return
  status=read(R.RUN/'status.json')if(R.RUN/'status.json').exists()else{}
  if time.time()-last>=3600:
   last=time.time();record=dict(at=last,status=status)
   if(R.RUN/'metrics.jsonl').exists():
    rows=E.values(R.RUN/'metrics.jsonl')[-30:];record['last30_mean_seconds']=statistics.mean(r['seconds']for r in rows)
   with(W/'hourly.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
  if status.get('step')==1000 and(R.RUN/'step_001000.complete.json').exists():break
  pid=read(W/'LAUNCHER.json')['pid'];proc=Path(f'/proc/{pid}/stat')
  if not proc.exists()or proc.read_text().split()[2]=='Z':raise RuntimeError('Training controller exited before1000; preserve complete checkpoints and inspect launch.log')
  time.sleep(30)
 atomic_json(W/'FINISH_STATUS.json',dict(stage='evaluating1000',at=time.time()))
 with(W/'evaluation1000.log').open('ab',buffering=0)as f:
  code=subprocess.run([PY,'-u',str(W/'evaluate_overlap.py'),'--step','1000'],stdout=f,stderr=subprocess.STDOUT).returncode
 if code:raise RuntimeError('Final1000 evaluation failed; retain full checkpoint/media and inspect evaluation1000.log')
 result();atomic_json(W/'FINISH_STATUS.json',dict(stage='complete',at=time.time(),comparison_sha256=sha(W/'FINAL_COMPARISON.json')))

if __name__=='__main__':
 with(W/'finish.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main()
  except BaseException as e:atomic_json(W/'FINISH_STATUS.json',dict(stage='failed',at=time.time(),error=repr(e)));raise
