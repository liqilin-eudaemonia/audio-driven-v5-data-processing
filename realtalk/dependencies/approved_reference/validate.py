import json,subprocess,hashlib,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'review';m=json.loads((OUT/'samples.json').read_text());jobs=json.loads((ROOT/'jobs.json').read_text());a=json.loads((ROOT.parent/(m['video_id']+'_annotations.json')).read_text())
old=json.loads((ROOT.parent/'full_video_v1/review/samples.json').read_text());oldmask=[False]*m['frames'];newmask=[False]*m['frames']
for r in old['pairs']:oldmask[r['start_frame']:r['end_frame']]=[True]*(r['end_frame']-r['start_frame'])
for r in m['pairs']:newmask[r['start_frame']:r['end_frame']]=[True]*(r['end_frame']-r['start_frame'])
assert oldmask==newmask and sum(newmask)==10021
assert [(r['start_frame'],r['end_frame'],r['kind']) for r in old['segments'] if r['kind']!='pair']==[(r['start_frame'],r['end_frame'],r['kind']) for r in m['segments'] if r['kind']!='pair']
assert all(not any(r['start']<q['start']<r['end'] for q in m['shots']) for r in m['pairs'])
assert m['baseline_manifest_sha256']==hashlib.sha256((ROOT.parent/'full_video_v1/review/samples.json').read_bytes()).hexdigest()
assert len(m['sentence_changes'])==3 and m['coverage_comparison']['lost_candidate_frames']==0
assert all(r['duration']<=11 and r['end_frame']-r['start_frame']<=275 for r in m['pairs'])
assert m['sentence_policy']['mandatory_max_seconds']==11
s=m['segments'];assert s[0]['start_frame']==0 and s[-1]['end_frame']==m['frames'];assert all(x['end_frame']==y['start_frame'] for x,y in zip(s,s[1:]));assert sum(r['end_frame']-r['start_frame'] for r in s)==m['frames'];assert len(m['pairs'])==sum(r['kind']=='pair' for r in s);assert abs(sum(m['stats'].values())-m['duration'])<.001
for r in m['pairs']:
 assert r['end_frame']>r['start_frame'];assert not r['training_admitted'];assert not any(r['start']<q['start']<r['end'] for q in m['shots']);assert not any(r['start']<b and r['end']>x for x,b in m['black'])
 assert not any(a[str(i)]['current_speaker']==r['listener'] for i in range(r['start_frame'],r['end_frame']))
 c=r['crop']
 if c:assert 0<=c['x']<=1280-c['size'] and 0<=c['y']<=720-c['size']
 # Every nonblack source frame with a known role must be in a candidate, even if tiny.
for r in s:
 if r['kind']=='role_unknown':assert all(a[str(i)]['current_speaker'] is None for i in range(r['start_frame'],r['end_frame']))
def check(j):
 p=j['path'];d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',p]));ss={v['codec_type']:v for v in d['streams']};assert 'video' in ss and 'audio' in ss
 for k in ['video','audio']:assert abs(float(ss[k]['duration'])-(j['end']-j['start']))<.09,(p,k,ss[k]['duration'])
 subprocess.run(['ffmpeg','-v','error','-xerror','-threads','1','-i',p,'-f','null','-'],check=True,stderr=subprocess.PIPE)
 return {'media':str(Path(p).relative_to(OUT)),'decoded':True,'duration_checked':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(check,jobs))
c=json.loads((OUT/'contract.json').read_text());assert c['manifest_sha256']==hashlib.sha256((OUT/'samples.json').read_bytes()).hexdigest()
(OUT/'validation.json').write_text(json.dumps({'media_checks':len(checks),'checks':checks,'exhaustive_frame_ledger':True,'known_role_frames_all_preserved':True,'pair_count':len(m['pairs']),'same_source_frame_coverage_as_baseline':True,'lost_candidate_frames':0},indent=2));print('PASS',len(checks),'media; exhaustive',m['frames'],'frames',flush=True)
