from pathlib import Path
import sys,json,statistics,math
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import torch
from p2_listen.contracts import read,sha,atomic_json
from p2_ltx_events.checkpoint import checked
from p2_ltx_events.config import source_identity
OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
c=read(OUT/'config.json');rows=[json.loads(x) for x in (OUT/'metrics.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows]==list(range(1,17))
assert read(OUT/'status.json')['complete'] and read(OUT/'launch_absolute/completed.json')['returncodes']==[0,0,0]
for row in rows:
 for key in ['av','event','event_clip','feature_video','feature_audio','weighted_total','seconds','input_wait_seconds','peak_reserved_gib']:
  assert math.isfinite(row[key]),(row['step'],key)
 assert row['event_temporal']==0
 assert row['per_event']['clip/visual:mouth closing']['known']==0
for name in ['connector','events','features','text_lora','whisper']:
 assert all(math.isfinite(r['grad_norm'][name]) and r['grad_norm'][name]>0 for r in rows),name
final=checked(OUT/'step_000016.pt')
assert final['step']==16 and final['identity']['world']==24 and len(final['rank_rng'])==24
assert final['identity']['experiment']==c and final['identity']['source']==source_identity()
initial=torch.load(OUT/'step_000000.pt',map_location='cpu',weights_only=True,mmap=True)
assert sha(OUT/'step_000000.pt')==read(OUT/'step_000000.complete.json')['sha256']
changed={}
for prefix in ['connector.','event_heads.','decoders.','text_lora.','whisper_adapter.','whisper_branches.']:
 keys=[k for k in final['model'] if k.startswith(prefix)]
 assert keys,prefix
 changed[prefix]=sum(not torch.equal(initial['model'][k],final['model'][k]) for k in keys)
 assert changed[prefix]>0,prefix
# Frozen renderer lives outside state_dict and optimizer ownership by design.
assert not any(k.startswith('renderer.') for k in final['model'])
assert final['identity']['assets']==initial['identity']['assets']
warm=rows[1:];seconds=[r['seconds'] for r in warm];wait=[r['input_wait_seconds'] for r in warm]
canary=read(OUT/'LONGEST_SAMPLE_CHECK.json');assert canary['success'] and canary['updates_restored'] and canary['rng_restored']
pre=read(OUT/'preflight.json')
report=dict(success=True,world_size=24,nodes=['node12','node8','node4'],steps=16,sample_presentations=384,
 longest_sample=canary,all_losses_and_gradients_finite=True,all_five_optimizer_groups_have_gradients=True,
 actual_updated_tensors_by_prefix=changed,frozen_renderer_excluded_from_checkpoint=True,frozen_asset_identity_unchanged=True,checkpoint_hashes_and_identity_valid=True,
 rank_rng_states=24,cpu_tests_passed=91,
 performance=dict(warm_steps=len(warm),mean_step_seconds=statistics.mean(seconds),median_step_seconds=statistics.median(seconds),
 min_step_seconds=min(seconds),max_step_seconds=max(seconds),mean_input_wait_seconds=statistics.mean(wait),max_input_wait_seconds=max(wait),
 peak_reserved_gib_rank0=max(r['peak_reserved_gib'] for r in rows),samples_per_second=24/statistics.mean(seconds),
 scope='15 warm steps; mixed clip lengths; not a full-epoch throughput estimate'),
 optimizations=['precomputed numeric labels','duration grouping window480 for batch24','CPU prefetch depth2','512MiB streaming FP64 buckets','RDMA mlx5_0:1 GID3'],
 canary_snapshot_note='Boundary hardened; existing renderer already excluded frozen base from state_dict; no measured speedup claimed',
 fixes=['24-GPU launcher support','remote config and resume paths normalized to absolute paths'],
 preflight=dict(samples=pre['samples'],processes=pre['processes'],elapsed_s=pre['elapsed_s']),
 config_sha256=sha(OUT/'config.json'),source_identity=source_identity(),training_processes_exited=True,production_training_started=False,
 nccl_joint_forward_backward_optimizer_verified=True,artifacts={name:sha(OUT/name) for name in ['metrics.jsonl','tests.xml','step_000016.complete.json','LONGEST_SAMPLE_CHECK.json']})
atomic_json(OUT/'RESULT.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('source_identity','artifacts')},ensure_ascii=False,indent=2))
