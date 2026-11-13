import json,os,signal,subprocess,time
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');out=root/'experiments/listen_labeling_node8_models'
p=out/'joy.process.json';r=json.loads(p.read_text());pid=r['pid'];proc=Path('/proc')/str(pid)
if proc.exists():
 args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 if str(root/'scripts/serve_listen_labeling_models.py') not in args:raise RuntimeError('PID ownership mismatch')
 os.kill(pid,signal.SIGTERM)
 for _ in range(50):
  if not proc.exists():break
  time.sleep(.1)
r['command'] += ['--max-generation-seconds','600']
log=open(out/'joy.log','a');log.write('\nRetry shortest clip with 600 second generation limit.\n');log.flush()
child=subprocess.Popen(r['command'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(r['gpu']),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1'))
r['previous_pid']=pid;r['pid']=child.pid;p.write_text(json.dumps(r,indent=2)+'\n');print('joy',child.pid)
cfg=json.loads((out/'joy_qwen_r3.json').read_text())
for ep in cfg['endpoints'].values():ep['revision'] += '-joy600-r4'
(out/'joy_qwen_r4.json').write_text(json.dumps(cfg,indent=2)+'\n')
