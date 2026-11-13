import subprocess,json,time,os
from pathlib import Path
out=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_visual_input50_20261029/native_fps_decision/speed')
args=['/private/software/conda/envs/vllm/bin/python', '/private/liqi_workspace/project_code/audio-driven-v5/scripts/benchmark_listen_native_fps.py', '--gpu', '4', '--memory', '.17', '--batch-size', '1', '--max-model-len', '32768', '--views', 'full', '--shared-gpu']
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6')
with (out/'run.log').open('a') as log:
 p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,cwd='/private/liqi_workspace/project_code/audio-driven-v5')
 (out/'pid').write_text(str(p.pid)+'\n')
 code=p.wait()
if code!=0:
 state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
 state.update(state='interrupted',reason='worker exit '+str(code),updated_at=time.time())
 tmp=out/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'status.json')
