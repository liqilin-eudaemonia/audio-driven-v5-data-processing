"""Validate completed GPU S0 runs, matching DDP updates, replay and decoded AV."""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import torch
import av
import soundfile as sf
import numpy as np
from p2_listen.contracts import read,sha,atomic_json
from p2_listen.training import checked_checkpoint,indices
from p2_listen.config import load,source_identity

torch.set_num_threads(4)
c=load(OUT/'config.json');jobs=read(OUT/'job_status.json')
parser=argparse.ArgumentParser();parser.add_argument('--training-only',action='store_true');args=parser.parse_args()
assert jobs['finished'] and all(jobs['jobs'][k]['state']=='passed' for k in ('E0','E1_first5','E1_resume','E1_ddp')),jobs
assert source_identity(c)==read(OUT/'preflight.json')['source']
for path,h in read(OUT/'protected_before.json').items():assert sha(path)==h,path
RUN=OUT/'runs/s0_20261027';DDP=OUT/'ddp/runs/s0_20261027/E1'
def logs(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
r0=logs(RUN/'E0/train_rank0.jsonl');r1=logs(RUN/'E1/train_rank0.jsonl')
d0=logs(DDP/'train_rank0.jsonl');d1=logs(DDP/'train_rank1.jsonl')
for rr in (r0,r1,d0,d1):
    assert [v['step'] for v in rr]==list(range(1,8))
    for v in rr:
        assert all(math.isfinite(v[k]) for k in ('base','event','seconds'))
        assert all(math.isfinite(z) for z in v['grad_norm'].values())
        assert all(z is None or math.isfinite(z) for z in v['event_loss_per_class'].values())
run0=read(RUN/'E0/run.json');run1=read(RUN/'E1/run.json');rund=read(DDP/'run.json')
for k in ('initialization','observations','targets','labels','statistics','K_train','source','assets'):
    assert run0[k]==run1[k]==rund[k],k
assert not any(r['debug_positive_events'] for r in (run0,run1,rund))
assert run1['statistics']['head_shake']['disabled_reason']=='no_positive_support'
assert len(run1['K_train'])==6
for rr in (r0,r1):
    assert len({idx for row in rr for idx,_ in indices(50,row['step'],8,20261027)})==50
for v in r0:assert v['paths_first_micro']['event']==[0.,0.]
for j in range(2):assert any(v['paths_first_micro']['event'][j]>0 for v in r1)
assert r0[0]['event']==r1[0]['event']
for one,a,b in zip(r1,d0,d1):
    assert sorted(a['ordinals']+b['ordinals'])==one['ordinals']
    assert one['event_known_bins']==a['event_known_bins']==b['event_known_bins']
    assert a['grad_norm']==b['grad_norm']
first_loss_error=abs(r1[0]['event']-d0[0]['event'])
assert first_loss_error<2e-6,first_loss_error
single=checked_checkpoint(RUN/'E1/step_000007.pt');distributed=checked_checkpoint(DDP/'step_000007.pt')
errors=[]
for k,t in single['model'].items():
    d=distributed['model'][k];assert torch.isfinite(t).all() and torch.isfinite(d).all()
    errors.append((k,float((t-d).abs().max())))
errors.sort(key=lambda p:p[1],reverse=True)
comparison=dict(passed=errors[0][1]<=2e-5,steps=7,world=2,global_batch=8,same_initialization=True,
                same_sample_ordinals=True,same_global_class_counts=True,rank_grad_norms_equal=True,
                first_step_event_loss_error=first_loss_error,model_max_abs_error=errors[0][1],
                model_absolute_tolerance=2e-5,largest_parameter_errors=errors[:10])
atomic_json(OUT/'ddp/comparison.json',comparison)
assert comparison['passed'],comparison
del single,distributed
if args.training_only:
    print(json.dumps(comparison,indent=2));raise SystemExit(0)
replay=read(OUT/'replay_validated.json');assert replay['passed']
source_only=read(OUT/'generated_source_only/source_only_audit.json')
assert source_only['generated'] and not source_only['gt_access'] and not source_only['blocked_attempts']
meta=read(OUT/'generated_source_only/completed.json');assert meta['status']=='success'
o=read(OUT/'generation_observation.json');duration=o['requested_duration_s']
with av.open(OUT/'generated_source_only/generated.mp4') as f:
    s=f.streams.video[0];assert (s.width,s.height)==(512,512) and float(s.average_rate)==25
    frames=list(f.decode(video=0));assert abs(len(frames)/25-duration)<=1/25+1e-6
    assert all(abs(float(v.time)-i/25)<1e-4 for i,v in enumerate(frames))
with av.open(OUT/'generated_source_only/generated.mp4') as f:
    audio_frames=list(f.decode(audio=0));assert audio_frames
    assert all(np.isfinite(v.to_ndarray()).all() for v in audio_frames)
wave,rate=sf.read(OUT/'generated_source_only/listener.wav');assert np.isfinite(wave).all()
assert abs(len(wave)/rate-duration)<=1/rate+1e-6
media=dict(passed=True,width=512,height=512,fps=25,video_frames=len(frames),wav_samples=len(wave),wav_rate=rate,
           requested_duration_s=duration,source_only_guard=True,quality_evaluated=False)
atomic_json(OUT/'generated_source_only/media_validation.json',media)
summary=dict(passed=True,host='172.29.104.220',physical_gpus=[4,5,6],gpu7_used=False,
             samples=50,policy=c['events']['admission_policy'],debug_positive_override=False,
             effective_classes=run1['K_train'],single_gpu_E0_steps=7,single_gpu_E1_steps=7,
             resume_from_step=5,resume_to_step=7,ddp_steps=7,ddp_world=2,
             event_to_u_E0_zero=True,event_to_u_E1_both_modalities=True,
             ddp_comparison=comparison,replay_passed=True,replay_state_restore_exact=replay['state_restore_exact'],
             replay_forward_loss_exact=replay['forward_loss_exact'],source_only_generation=media,
             original_artifacts_unchanged=True,model_code_changed=False,
             verification_code_changed=True,
             formal_training_started=False,quality_evaluated=False)
atomic_json(OUT/'summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
