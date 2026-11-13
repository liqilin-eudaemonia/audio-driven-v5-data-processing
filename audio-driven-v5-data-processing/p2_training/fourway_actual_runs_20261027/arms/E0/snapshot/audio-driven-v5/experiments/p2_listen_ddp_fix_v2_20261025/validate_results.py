"""Strict final acceptance: both arms, all GPU jobs, replay, decoded AV, integrity."""
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import av
import numpy as np
import soundfile as sf
from p2_listen.contracts import read,rows,sha,digest,atomic_json
from p2_listen.config import load,source_identity
from p2_listen.training import indices

cfg=load(OUT/'config.json');jobs=read(OUT/'job_status.json')
assert jobs['finished'] and jobs['passed'],jobs
assert all(v['state']=='passed' for v in jobs['jobs'].values())
assert source_identity(cfg)==read(OUT/'preflight.json')['source']
for path,h in read(OUT/'protected_before.json').items():assert sha(path)==h,path
unit=read(OUT/'unit_tests.json');assert unit['passed'] and unit['tests_run']==105
for path,h in unit['code'].items():assert sha(ROOT/path)==h,path
cpu_ddp=read(OUT/'cpu_ddp.json')
assert cpu_ddp['world_size']==2 and cpu_ddp['rank_without_labels']
assert all(cpu_ddp[k]['max_parameter_error']<=2e-5 for k in ('probe_only','joint_event'))
pair=read(OUT/'ddp_comparison.json');assert pair['passed'] and set(pair['arms'])=={'E0','E1'}
assert all(x['passed'] and x['model_absolute_tolerance']==2e-5 for x in pair['arms'].values())
observations=rows(OUT/'manifests/observations.json');assert len(observations)==50
run=OUT/'runs/s0_20261027'
traces={}
for arm in ('E0','E1'):
 ident=read(run/arm/'run.json');assert not ident['debug_positive_events']
 assert ident['numerics']['deterministic_algorithms'] and not ident['numerics']['warn_only']
 assert ident['numerics']['policy']=='p2-deterministic-training-v2'
 assert ident['numerics']['gradient_accumulation']=='fp64' and ident['numerics']['gradient_reduction']=='sum'
 assert ident['numerics']['loss_normalization']=='global_batch'
 assert len(ident['K_train'])==6
 assert [digest(read(OUT/'labels/canonical'/(o['sample_id']+'.json'))) for o in observations]==ident['labels']
 records=[json.loads(x) for x in (run/arm/'train_rank0.jsonl').read_text().splitlines()]
 assert [v['step'] for v in records]==list(range(1,8))
 assert len({i for v in records for i,_ in indices(50,v['step'],8,20261027)})==50
 for v in records:
  assert math.isfinite(v['base']) and math.isfinite(v['event'])
  assert all(math.isfinite(x) for x in v['grad_norm'].values())
 if arm=='E0':assert all(v['paths_first_micro']['event']==[0.,0.] for v in records)
 else:assert all(any(v['paths_first_micro']['event'][j]>0 for v in records) for j in (0,1))
 traces[arm]=dict(steps=7,samples_covered=50,event_U_gradients=[v['paths_first_micro']['event'] for v in records],
                  step_seconds=[v['seconds'] for v in records])
replay=read(OUT/'replay.json');assert replay['passed'] and replay['state_restore_exact'] and replay['forward_comparison']['passed']
assert replay['numerics']==ident['numerics'] and replay['deterministic_algorithms']
gen=OUT/'generated_source_only';meta=read(gen/'completed.json');assert meta['status']=='success'
guard=read(gen/'source_only_audit.json');assert guard['generated'] and guard['filesystem_guard_active'] and not guard['gt_access'] and not guard['blocked_attempts']
o=read(OUT/'generation_observation.json');duration=o['requested_duration_s']
with av.open(gen/'generated.mp4') as f:
 s=f.streams.video[0];assert (s.width,s.height)==(512,512) and float(s.average_rate)==25
 frames=list(f.decode(video=0));assert abs(len(frames)/25-duration)<=1/25+1e-6
 assert all(abs(float(v.time)-i/25)<1e-4 for i,v in enumerate(frames))
with av.open(gen/'generated.mp4') as f:
 audio=list(f.decode(audio=0));assert audio and all(np.isfinite(v.to_ndarray()).all() for v in audio)
wave,rate=sf.read(gen/'listener.wav');assert np.isfinite(wave).all() and abs(len(wave)/rate-duration)<=1/rate+1e-6
media=dict(passed=True,size=[512,512],fps=25,video_frames=len(frames),wav_samples=len(wave),wav_rate=rate,duration_s=duration,source_only=True,quality_evaluated=False)
atomic_json(gen/'media_validation.json',media)
summary=dict(passed=True,host='172.29.104.220',gpus=[4,5,6],samples=50,cpu_tests=105,numerics=ident['numerics'],
 cpu_ddp=cpu_ddp,
 single_gpu=traces,ddp=pair['arms'],ddp_parameter_tolerance=2e-5,resume_steps=[5,7],
 replay_passed=True,replay_state_restore_exact=True,replay_forward_loss_exact=replay['forward_loss_exact'],
 replay_max_abs_error=replay['max_abs_error'],generation=media,original_artifacts_unchanged=True,
 remaining_checks=[],formal_training_started=False,quality_evaluated=False)
atomic_json(OUT/'summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
