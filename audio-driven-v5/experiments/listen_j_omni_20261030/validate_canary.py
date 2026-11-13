from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_omni import read_audio,prepare_input,fingerprint,file_sha
from listen_benchmark.native_caption_nod import render
from listen_benchmark.native_caption_quality import generation_failures
from listen_benchmark.j_annotation import read,write
run=Path(__file__).resolve().parent;work=run/'canary';cfg=read(work/'config.json')
files=sorted((work/'caption/results').glob('*/*.json'));assert len(files)==16,len(files)
counts=dict(samples=16,audible=0,silent=0,with_nod=0,without_nod=0,normal_finish=0);replay=[]
for file in files:
 r=read(file);sid=r['sample_id'];ev=r['input_evidence'];s=r['source'];a=r['audio_recipe'];wave,sr,identity=read_audio(a)
 assert identity==ev['audio_identity'];assert wave.shape==(round(s['source_end_s']*sr)-round(s['source_start_s']*sr),1)
 assert a['source_time_origin_s']==round(s['source_start_s']*sr)/sr==ev['video_evidence']['source_time_origin_s']
 assert ev['video_evidence']['video_transform']==s['video_transform']
 assert ev['second_per_grid_ts']==[.25] and ev['fps']==8
 assert ev['audio_encoder_samples']==len(wave)*16000//sr+(1 if len(wave)*16000%sr else 0)
 row=dict(sample_id=sid,duration_s=s['duration_s'],nod_supplement=ev['nod_supplement'],audio_preflight=dict(full_silence=identity['full_silence']))
 assert render(row,style='first_plus_evidence')==r['prompt'] and 'PRIOR PARAGRAPH' not in r['prompt']
 assert not generation_failures(r['caption'],r['finish_reason']) and r['generation_checks_passed']
 assert r['pipeline_id']==cfg['pipeline_id'] and a['extra_mute_intervals']==[]
 counts['audible']+=not identity['full_silence'];counts['silent']+=identity['full_silence'];counts['with_nod']+=bool(ev['nod_supplement']['events']);counts['without_nod']+=not ev['nod_supplement']['events'];counts['normal_finish']+=1
 if len(replay)<2 and (not replay or identity['full_silence']!=replay[0]['silent']):
  origin=read(work/'job_origins'/f"{s['pair']}.json");p=prepare_input(dict(sample_id=sid,source=s),origin['old_work'],work,cfg['nodes']['canary']['resource_work'])
  assert p['evidence']['video_frames_sha256']==ev['video_frames_sha256'] and p['evidence']['audio_encoder_sha256']==ev['audio_encoder_sha256']
  replay.append(dict(sample_id=sid,silent=identity['full_silence'],video_audio_replay_equal=True))
assert counts['audible'] and counts['silent'] and counts['with_nod'] and counts['without_nod'],counts
workers=[read(p) for p in (work/'workers').glob('*.json')];assert len(workers)==4 and all(w['completed_clips']==4 and w['stage']=='stopped' for w in workers)
assert read(run/'validation/cross_host_locks.json')['passed']
report=dict(passed=True,counts=counts,exact_replays=replay,unit_tests=12,cross_host_lock_verified=True,raw_results_sha256={str(f.relative_to(work)):file_sha(f) for f in files},semantic_quality_verified=False,new_wav_files=False)
write(run/'validation/canary_audit.json',report);print(json.dumps(report,ensure_ascii=False))
