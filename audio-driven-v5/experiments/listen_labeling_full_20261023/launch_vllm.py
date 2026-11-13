import json,os,subprocess,sys
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');out=root/'experiments/listen_labeling_full_20261023'
name=sys.argv[1]
models={'joy0_fast':('/private/ckpts/JoyAI-VL-Interaction',0,18260,'JoyAI-VL-Interaction-listen-gpu0',.32,{'image':128,'video':0}),
'joy1_fast':('/private/ckpts/JoyAI-VL-Interaction',1,18261,'JoyAI-VL-Interaction-listen-gpu1',.32,{'image':128,'video':0}),
'joy2_fast':('/private/ckpts/JoyAI-VL-Interaction',2,18262,'JoyAI-VL-Interaction-listen-gpu2',.60,{'image':128,'video':0}),
'joy7_fast':('/private/ckpts/JoyAI-VL-Interaction',7,18267,'JoyAI-VL-Interaction-listen-gpu7',.65,{'image':128,'video':0}),
'omni6_fast':('/private/ckpts/Qwen3-Omni/Instruct',6,18266,'Qwen3-Omni-Instruct-listen-gpu6',.72,{'image':1,'audio':1,'video':0}),
'joy_fast':('/private/ckpts/JoyAI-VL-Interaction',4,18160,'JoyAI-VL-Interaction-listen-fast',.68,{'image':128,'video':0}),
'omni_audio_fast':('/private/ckpts/Qwen3-Omni/Instruct',5,18162,'Qwen3-Omni-Instruct-listen-audio-fast',.68,{'image':1,'audio':1,'video':0}),
'minicpm':('/private/codes/full-duplex/code/MiniCPM-o-Demo/models/MiniCPM-o-4_5',6,18063,'MiniCPM-o-4.5-listen-vllm',.35,{'image':128,'audio':1,'video':0}),
'joy':('/private/ckpts/JoyAI-VL-Interaction',3,18060,'JoyAI-VL-Interaction-listen-vllm',.35,{'image':128,'video':0}),
'audio':('/private/ckpts/Qwen2-Audio-7B-Instruct',1,18061,'Qwen2-Audio-7B-Instruct-listen-vllm',.35,{'audio':1}),
'omni':('/private/ckpts/Qwen3-Omni/Instruct',7,18062,'Qwen3-Omni-Instruct-listen-vllm',.63,{'image':128,'audio':1,'video':0})}
path,gpu,port,served,mem,limits=models[name]
free,total=map(float,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(','))
mem=min(mem,round((free-12288)/total,3))
if mem<.20:raise RuntimeError('Insufficient headroom for another model')
cmd=['/private/software/conda/envs/vllm/bin/python','-m','vllm.entrypoints.openai.api_server','--model',path,'--served-model-name',served,'--host','127.0.0.1','--port',str(port),'--gpu-memory-utilization',str(mem),'--max-model-len',('8192' if (name=='audio' or name.startswith('omni') and name.endswith('_fast')) else '32768'),'--max-num-seqs','8','--max-num-batched-tokens','8192','--limit-mm-per-prompt',json.dumps(limits),'--enforce-eager','--trust-remote-code','--skip-mm-profiling']
if name.endswith('_fast'):
 cmd.remove('--enforce-eager');cmd[cmd.index('--max-num-seqs')+1]='16'
record=out/(name+'.vllm.process.json')
if record.exists():
 old=json.loads(record.read_text());proc=Path('/proc')/str(old['pid'])/'cmdline'
 if proc.exists() and served.encode() in proc.read_bytes():raise RuntimeError('Already running')
log=open(out/(name+'.vllm.log'),'a')
p=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',LD_LIBRARY_PATH='/private/software/conda/envs/vllm/lib:'+os.environ.get('LD_LIBRARY_PATH','')))
record.write_text(json.dumps(dict(pid=p.pid,gpu=gpu,port=port,served=served,command=cmd),indent=2)+'\n');print(name,p.pid)
