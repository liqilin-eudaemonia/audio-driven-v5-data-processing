import json,time,os,signal,subprocess
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');out=root/'experiments/p2_joint_qwen_ltx_test64_16gpu_20261113';out.mkdir(exist_ok=True)
roots=[root/'experiments'/name for name in ['p2_ltx15_test64_sft2752_20261113','p2_planning_ckpt_seamless200_20261113','p2_planning_ckpt_extended_seamless200_20261113']]
records=[]
for w in roots:
 paths=[w/'CONTROLLER.json',w/'LAUNCH.json',w/'DEVICE.json',w/'cluster29/CONTROLLER.json',w/'node11_attempt/LAUNCH.json']
 paths+=list((w/'node11_attempt').glob('process_*.json'))
 paths+=list((w/'cluster29/processes').glob('*.json'))
 paths+=list((w/'processes').glob('*.json'))
 for f in paths:
  if not f.exists():continue
  r=json.loads(f.read_text());pid=r.get('pid')
  if not pid:continue
  if r.get('node') not in (None,'node11'):continue
  p=Path('/proc')/str(pid)
  try:state=p.joinpath('stat').read_text().split(') ',1)[1][0];cmd=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
  except (FileNotFoundError,ProcessLookupError):state='absent';cmd=''
  match=state not in ['absent','Z'] and str(w) in cmd
  records.append(dict(receipt=str(f),pid=pid,state=state,live_owned=match))
  if match:os.kill(pid,signal.SIGTERM)
for _ in range(15):
 live=[]
 for r in records:
  if not r['live_owned']:continue
  p=Path('/proc')/str(r['pid'])
  try:state=p.joinpath('stat').read_text().split(') ',1)[1][0]
  except (FileNotFoundError,ProcessLookupError):state='absent'
  if state not in ['absent','Z']:live.append(r['pid'])
 if not live:break
 time.sleep(1)
assert not live,live
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus=[]
for line in raw.splitlines():
 a=[x.strip() for x in line.split(',')];gpus.append(dict(index=int(a[0]),uuid=a[1],memory_mib=float(a[2]),util=float(a[3])))
report=dict(at=time.time(),node='node11',host='172.29.189.119',scope='Current Qwen/LTX checkpoint-selection tasks; independent reference-method queue is outside this run',success=True,matched_live_owned_before=sum(r['live_owned'] for r in records),live_owned_after=0,checked_process_receipts=records,gpus=gpus,independent_reference_queue='GPU7 p2_reference_methods_seamless6624_20261112 remains running',results_preserved=True,no_new_evaluation_launched_on_node11=True)
(out/'NODE11_RELEASED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['checked_process_receipts','gpus']}));print('gpu0to4',[g for g in gpus if g['index']<=4])
