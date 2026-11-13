"""Verify per-card outputs against committed J records and original metadata."""
from pathlib import Path
import sys,json,time,hashlib,tarfile
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,adapt_clip
from listen_benchmark.j_omni import read_audio,file_sha
from listen_benchmark.official_audio_gate import make_gate
from listen_audio_postprocess.bulk import official
import numpy as np
@lru_cache(maxsize=24)
def original(path):
 b=Path(path).read_bytes();d=json.loads(b);rec=read(Path(path).parent/'complete.json');h=hashlib.sha256(b).hexdigest()
 assert d['complete'] and h==rec['result_sha256'] and d['cluster_contract']==rec['contract']
 return d,h
cfg=read(W/'config.json');assert all(file_sha(ROOT/p)==h for p,h in cfg['code_hashes'].items())
checks=[];archive_nodes=set()
for entry in read(W/'validation/production_v2_24gpu_audit.json')['checks']:
 r=read(entry['file']);s=r['source'];d,h=original(s['source_result']);assert h==s['source_result_sha256']
 assert Path(s['source_result']).is_relative_to(ROOT/'data_corpora/seamless_listen_j_full_20261029/results')
 c=d['clips'][s['source_clip_index']];row=adapt_clip(c,{k:s.get(k) for k in ['pair','split','source_catalog_status']},s['source_result'],h,s['source_clip_index']);assert row['source']==s
 assert c['listener_id']==s['listener_id'] and c['listener_id']!=c['speaker_id']
 full_audio=c['normalized_audio']['listener'];assert full_audio['contract']['source']==c['sources']['listener']
 marker=read(Path(full_audio['path']).with_suffix('.json'));assert marker['contract']==full_audio['contract'] and marker['sha256']==s['normalized_audio']['sha256']
 recipe=r['audio_recipe'];sr=recipe['source']['sample_rate'];assert recipe['source']['first_sample']==round(c['start']*sr) and recipe['source']['last_sample']==round(c['end']*sr)
 assert recipe['provenance']['member']==str(Path(c['sources']['listener']['member']).with_suffix('.json'))
 archive_checked=False
 if r['node'] not in archive_nodes:
  with tarfile.open(recipe['provenance']['archive'],'r:') as tar:
   b=tar.extractfile(recipe['provenance']['member']).read();wav_member=str(Path(c['sources']['listener']['member']).with_suffix('.wav'));assert tar.getmember(wav_member).isfile()
  assert hashlib.sha256(b).hexdigest()==recipe['provenance']['sha256']
  off=official(json.loads(b),recipe['source_time_origin_s'],s['duration_s']);off={k:[{q:v for q,v in x.items() if q in ('id','start','end')} for x in off[k]] for k in ('words','sentences','vad')}
  assert off==recipe['official_listener'];archive_nodes.add(r['node']);archive_checked=True
 wave,rate,identity=read_audio(recipe);assert identity==r['input_evidence']['audio_identity']
 gate,_=make_gate(recipe['official_listener'],len(wave),rate);assert np.all(wave[gate==0]==0)
 assert len(wave)==round(c['end']*sr)-round(c['start']*sr)
 assert recipe['source_time_origin_s']==r['input_evidence']['video_evidence']['source_time_origin_s']
 assert r['input_evidence']['video_evidence']['video_transform']==s['video_transform'] and s['video_transform']['output_size']==[512,512]
 nod=read(r['input_evidence']['nod_supplement']['detector_file']);assert nod['source']==s
 assert recipe['extra_mute_intervals']==[] and recipe['language_filter']=='disabled_user_official_union_only'
 checks.append(dict(node=r['node'],gpu=r['gpu'],sample_id=r['sample_id'],committed_J_record_equal=True,listener_identity=True,source_wav_marker_verified=True,original_archive_metadata_reopened=archive_checked,crop_matches=True,nod_source_matches=True,audio_length_and_origin_match=True,muted_outside_gate=True,actual_waveform_hash_match=True,source_result=s['source_result'],source_audio=s['normalized_audio']['path']))
report=dict(passed=True,at=time.time(),samples=len(checks),nodes=sorted(archive_nodes),all_runtime_code_hashes_match=True,original_metadata_reopened=len(archive_nodes),checks=checks,scope='24 sampled outputs across all GPU workers; not exhaustive corpus or semantic audit')
write(W/'validation/input_provenance_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
