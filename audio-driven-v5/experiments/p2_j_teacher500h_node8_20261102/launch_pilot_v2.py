import os,subprocess,json,time
from pathlib import Path
r=Path('/private/liqi_workspace/project_code/audio-driven-v5');o=r/'experiments/p2_j_teacher500h_node8_20261102';p=o/'pilot_v2';p.joinpath('logs').mkdir(exist_ok=True)
rows=[]
for gpu,batch in enumerate([16,8,16,32,8,16,32,16]):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYAV_LOGGING='off',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
 cmd=['/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python',str(r/'scripts/verify_p2_j_teacher.py'),'--work',str(p),'--gpu',str(gpu),'--batch',str(batch),'--producers','4']
 with (p/'logs'/f'gpu{gpu}.log').open('a') as log:child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
 rows.append(dict(gpu=gpu,batch=batch,pid=child.pid,command=cmd))
(p/'launchers.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows),flush=True)
