import sys,json,time,hashlib,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write,result_path
from listen_benchmark.j_omni_crop import effective_row
from listen_benchmark.j_omni import fingerprint
cfg=read(W/'config.json');selected={}
for label in ['D1_after_roi','D2_after_roi']:
 for r in read(D/label/'finished_ids.json'):selected.setdefault((r['node'],r['gpu']),r['sample_id'])
assert len(selected)==32
checks=[]
for (node,gpu),sid in sorted(selected.items()):
 r=read(result_path(W,'caption',sid));s=read(W/'workers'/f'{node}-gpu-{gpu}.json')
 assert '+roi-rgb-v1' in s['execution_revision'] and '+roi-rgb-v1' in r['execution_revision']
 assert s['completed_clips']>0 and time.time()-s['heartbeat']<60 and not s.get('last_error') and not s['failed_clips']
 assert r['generation_checks_passed'] and r['engine_max_num_seqs']==1 and r['preparation_metrics']['roi_crop']
 actual,provenance=effective_row(dict(sample_id=sid,source=r['original_source']),cfg)
 assert r['source']==actual['source'] and r['crop_provenance']==provenance
 ev=r['input_evidence'];assert ev['video_evidence']['source']==r['source']['source']
 assert ev['video_evidence']['video_transform']==r['source']['video_transform']
 assert ev['video_evidence']['requested_fps']==8
 assert ev['audio_identity']['recipe_sha256']==fingerprint(r['audio_recipe'])
 n=ev['nod_supplement'];p=Path(n['detector_file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==n['detector_sha256']
 nod=read(p);assert nod['source']['video_transform']==r['source']['video_transform']
 assert nod['source']['source_start_s']==r['source']['source_start_s'] and nod['source']['source_end_s']==r['source']['source_end_s']
 checks.append(dict(node=node,gpu=gpu,sample_id=sid,source_audio_crop_nod_passed=True,pid=s['pid']))
for rel,digest in cfg['code_hashes'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==digest,rel
before=read(D/'C_before_roi/summary.json');bwork=read(D/'C_before_roi/exact_work.json');after=[read(D/l/'summary.json') for l in ['D1_after_roi','D2_after_roi']];work=[read(D/l/'exact_work.json') for l in ['D1_after_roi','D2_after_roi']]
rate=statistics.mean(r['clips_per_second'] for r in after);video=statistics.mean(r['video_seconds_per_second'] for r in work)
assert all(r['clips_per_second']>before['clips_per_second']*1.1 and r['model_failures']==0 for r in after)
for label in ['D1_after_roi','D2_after_roi']:
 for node in cfg['nodes']:assert read(D/label/f'{node}.json')['valid_no_restarts']
for path in ['queues/caption/state.json','visual_quality/queues/measure/state.json']:assert not read(W/path)['failed']
result=dict(at=time.time(),status='deployed_and_measured',before_clips_per_second=before['clips_per_second'],after_windows=after,after_clips_per_second=rate,observed_speedup=rate/before['clips_per_second'],before_video_seconds_per_second=bwork['video_seconds_per_second'],after_video_seconds_per_second=video,before_gpu_util=before['gpu_util'],after_gpu_util=statistics.mean(r['gpu_util'] for r in after),all_32_source_checks=checks,paired_input_replay=read(D/'paired_prepare_result.json')['speedup'],limitations='Sequential production windows contain different clips; observed rate ratio is not a controlled full-corpus speedup guarantee. Six real input replays and matched input-only ABBA preserve pixels and nod evidence. GPU utilization remains below full load.',config=str(D/'accepted_config.json'),goal_resumed=False,recurring_reports_created=False)
write(D/'accepted_config.json',cfg);write(D/'acceptance.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['all_32_source_checks','after_windows']}))
