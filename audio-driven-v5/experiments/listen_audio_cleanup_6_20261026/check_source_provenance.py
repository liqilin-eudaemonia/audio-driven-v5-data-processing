"""Read-only verification from archived standalone WAV through normalization to clips."""
import sys,json,subprocess,io,hashlib,concurrent.futures
from pathlib import Path
import numpy as np,soundfile as sf
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.backchannel_face_stats import SourceReader
from p3_listen.media import uri,members
run=ROOT/'experiments/listen_audio_cleanup_6_20261026';reader=SourceReader(ROOT/'experiments/p2_features_full_crop384_20261025');jobs=[]
for c in json.loads((run/'cases.json').read_text()):
 label=json.loads((run/'cases'/c['id']/'original_labels.json').read_text());row=reader.get(label['source'])
 for role in ['listener','speaker']:
  info=row['normalized_audio'][role];source=info['contract']['source'];wav_name=str(Path(source['member']).with_suffix('.wav'));u=uri(source,'.wav');offset,size=members(source['archive'])[wav_name]
  jobs.append((c,row,role,info,source,wav_name,u,offset,size))
reader.close()
def check(job):
 c,row,role,info,source,name,u,offset,size=job;gain=info['gain_db'];peak=10**(-1/20)
 filters=(f'volume={gain:.6f}dB,aresample=192000,apad=pad_len=959,alimiter=limit={peak}:level=false:attack=5:release=50,atrim=start_sample=959,asetpts=PTS-STARTPTS,aresample=48000')
 p=subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','1','-i',u,'-af',filters,'-ac','1','-ar','48000','-c:a','pcm_s24le','-f','wav','pipe:1'],capture_output=True,check=True)
 expected,sr=sf.read(io.BytesIO(p.stdout),dtype='float32');actual,ar=sf.read(info['path'],dtype='float32')
 assert sr==ar==48000 and len(actual)==len(expected)
 maximum=float(np.max(np.abs(actual-expected)));clip_path=run/'cases'/c['id']/('original.wav' if role=='listener' else 'speaker.wav');clip,cr=sf.read(clip_path,dtype='float32');start=round(row['start']*sr)
 segment=actual[start:start+len(clip)];clipdiff=float(np.max(abs(segment-clip)));quantized=np.clip(np.floor(segment.astype(np.float64)*32768),-32768,32767)/32768;quantdiff=float(np.max(abs(quantized-clip)));hashmatches=hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()==info['sha256']
 result=dict(sample_id=c['id'],role=role,source_archive=source['archive'],actual_audio_member=name,actual_audio_offset=offset,actual_audio_bytes=size,stored_index_member=source['member'],reads_standalone_wav=True,normalization_gain_db=gain,normalized_path=info['path'],normalized_sha_matches=hashmatches,reproduced_from_official_wav_max_abs_error=maximum,clip_vs_normalized_max_abs_error=clipdiff,clip_subtype=sf.info(clip_path).subtype,clip_vs_pcm16_quantized_max_abs_error=quantdiff,passed=maximum<=2**-22 and quantdiff==0 and hashmatches)
 print(c['id'].split('_')[2],role,'maxdiff',maximum,'clipdiff',clipdiff,'PASS',result['passed'],flush=True)
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(check,jobs))
(run/'source_provenance_check.json').write_text(json.dumps(dict(schema='standalone-wav-provenance-v1',read_only_check=True,results=results,all_passed=all(x['passed'] for x in results)),ensure_ascii=False,indent=2))
assert all(x['passed'] for x in results)
