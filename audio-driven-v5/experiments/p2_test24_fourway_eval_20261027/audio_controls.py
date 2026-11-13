import sys,runpy,time,subprocess,os,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
read=lambda p:json.loads(Path(p).read_text())
# GPU0 belongs to metric shard0; wait for its completion and actual GPU release.
while not (OUT/'E0/omni/shard-0.json').exists() or not read(OUT/'E0/omni/shard-0.json').get('finished'):time.sleep(5)
while int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>2048:time.sleep(5)
subprocess.run([str(PY),str(OUT/'check_audio_decode_gpu.py'),'cuda:0'],check=True)
exps=read(OUT/'experiments.json');e=exps['E0F'];runpy.run_path(str(ROOT/'scripts/p2_eval_generate.py'))['bootstrap'](['--run',e['run']])
from p2_listen.generation import generate
sid=max(read(OUT/'training_audio_supervision_audit.json')['E0']['rows'],key=lambda r:r['target_rms'])['sample_id']
obs=next(o for o in read(ROOT/'experiments/p2_train24_fourway_eval_20261027/observations.json') if o['sample_id']==sid)
for mode in ['text_only','remove_u']:
 dest=OUT/'audio_controls'/mode
 if (dest/'completed.json').exists():continue
 print('start',mode,sid,flush=True)
 generate(e['config'],obs,dest,checkpoint=e['checkpoint'] if mode=='remove_u' else None,mode=mode,seed=137,device='cuda:0')
 print('finished',mode,flush=True)
print('audio controls finished',flush=True)
