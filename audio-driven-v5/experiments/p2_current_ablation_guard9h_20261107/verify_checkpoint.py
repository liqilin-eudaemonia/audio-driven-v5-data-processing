"""Read-only complete state verification for a progressing ablation."""
from pathlib import Path
import sys,time,runpy,torch
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from p2_listen.training import rng_state,restore_rng
WORK=Path(__file__).resolve().parent
arm=sys.argv[1];step=int(sys.argv[2]);assert arm in ('feature','whisper')
out=ROOT/'experiments'/('p2_current500h_av_'+arm+'31_20261107')
if arm=='feature':
 from p2_current_av_feature.checkpoint import checked
else:
 from p2_current_av_whisper.checkpoint import checked
p=out/f'step_{step:06d}.pt';v=checked(p)
assert v['step']==step and v['identity']==read(out/'run.json')and len(v['rank_rng'])==31
original=rng_state()
for r in v['rank_rng']:restore_rng(r)
restore_rng(original)
steps=sorted({int(s['step'])for s in v['optimizer']['state'].values()if 'step'in s});assert steps==[step]
assert not torch.cuda.is_initialized()
g=runpy.run_path(str(WORK/'guardian.py'));snap=g['probe'](out);assert snap['healthy31']
until=time.time()+60
while read(out/'status.json')['step']<=step and time.time()<until:time.sleep(2)
status=read(out/'status.json');assert status['step']>step
prefix='CHECKPOINT'if arm=='feature'else'WHISPER_CHECKPOINT'
value=dict(at=time.time(),success=True,arm=arm,path=str(p),bytes=p.stat().st_size,sha256=sha(p),strict_run_identity=True,model_and_optimizer_finite=True,optimizer_steps=steps,rank_rng_cpu_restore_checked=31,cuda_initialized=False,subsequent_step=status['step'],live_cluster=snap)
atomic_json(WORK/f'{prefix}{step}_VERIFIED.json',value)
print(dict(checkpoint_valid=True,arm=arm,checkpoint=step,subsequent_step=status['step'],live31=True))
