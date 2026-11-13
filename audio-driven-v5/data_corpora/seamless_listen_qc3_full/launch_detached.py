import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path('/private/liqi_workspace/project_code/audio-driven-v5')
out=root/'data_corpora/seamless_listen_qc3_full'
node=int(sys.argv[1]);assert node in (0,1,2)
(out/'logs').mkdir(exist_ok=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
cmd=[sys.executable,'-u',str(root/'scripts/run_listen_qc3_bulk.py'),'launch-node','--node',str(node)]
with (out/'logs'/f'node_{node}.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(root.parent),env=env)
record=dict(node=node,pid=p.pid,command=cmd,utc=datetime.now(timezone.utc).isoformat())
(out/f'launch_{node}.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
