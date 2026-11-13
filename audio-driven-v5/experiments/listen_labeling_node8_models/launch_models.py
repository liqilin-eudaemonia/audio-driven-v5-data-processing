"""Run on authorized node8 only. Starts only this task's loopback smoke servers."""
import json, os
from pathlib import Path
import subprocess
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
OUT=ROOT/'experiments/listen_labeling_node8_models'
models=[('qwen3-omni',7,17062,'/private/software/conda/envs/qwenomni/bin/python','/private/ckpts/Qwen3-Omni/Instruct','Qwen3-Omni-Instruct-listen-test'),
        ('minicpm',6,17063,'/private/codes/full-duplex/code/MiniCPM-o-Demo/.venv/base/bin/python','/private/codes/full-duplex/code/MiniCPM-o-Demo/models/MiniCPM-o-4_5','MiniCPM-o-4.5-listen-test')]
for backend,gpu,port,python,model,name in models:
    import socket
    probe=socket.socket();probe.bind(('127.0.0.1',port));probe.close()
    command=[python,'-u',str(ROOT/'scripts/serve_listen_labeling_models.py'),'--backend',backend,'--model-path',model,'--served-model-name',name,'--port',str(port)]
    log=open(OUT/(backend+'.log'),'a')
    proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1'))
    record=dict(pid=proc.pid,backend=backend,gpu=gpu,port=port,command=command)
    (OUT/(backend+'.process.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
