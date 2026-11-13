import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');OUT=ROOT/'experiments/listen_labeling_node8_models'
for name in ['qwen3-omni','joy']:
 p=OUT/(name+'.process.json');r=json.loads(p.read_text());pid=r['pid']
 proc=Path('/proc')/str(pid)
 if proc.exists():
  args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if str(ROOT/'scripts/serve_listen_labeling_models.py') not in args:raise RuntimeError('PID ownership mismatch')
  os.kill(pid,signal.SIGTERM)
  for _ in range(50):
   if not proc.exists():break
   time.sleep(.1)
 log=open(OUT/(name+'.log'),'a')
 log.write('\nRestart: add bounded generation/progress; Qwen thinking explicitly disabled. Previous incomplete requests retained.\n');log.flush()
 child=subprocess.Popen(r['command'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(r['gpu']),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1'))
 r['previous_pid']=pid;r['pid']=child.pid;p.write_text(json.dumps(r,indent=2)+'\n');print(name,child.pid)
