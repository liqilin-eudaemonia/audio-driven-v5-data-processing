"""Four independent paired smoke tests; publish readiness only after all eight arms."""
from pathlib import Path
from p2_listen.contracts import read,atomic_json,sha
from .data import publish_task
LOSSES=('av','av_event','av_feature','av_event_feature')


def schedule(root):
 root=Path(root);reports=[]
 for loss in LOSSES:
  tid='s0_final_'+loss;out=root/'s0'/loss
  publish_task(root,dict(id=tid,kind='s0',chunk='final:'+loss,output=str(out),priority=0))
  state=root/'pool/tasks'/(tid+'.json')
  if not state.exists() or read(state)['status']!='succeeded':continue
  report=read(out/'COMPLETE.json')
  if report.get('success') is not True or [r['id'] for r in report['runs']]!=[f'guided_{loss}_s1',f'guided_{loss}_s2']:raise ValueError('Incomplete or mismatched paired smoke test')
  reports.append(report)
 if len(reports)!=4:return False
 from .backend import source_identity
 if any(r['source']!=source_identity() for r in reports):raise ValueError('Smoke source changed')
 atomic_json(root/'s0/COMPLETE.json',dict(success=True,runs=[r for report in reports for r in report['runs']],paired_parallel=True,source=source_identity(),parts={loss:sha(root/'s0'/loss/'COMPLETE.json') for loss in LOSSES}),immutable=True)
 atomic_json(root/'TRAINING_READY.json',dict(samples=read(root/'view/COMPLETE.json')['samples'],s0_sha256=sha(root/'s0/COMPLETE.json'),configs_sha256=sha(root/'configs/matrix.json')),immutable=True)
 return True
