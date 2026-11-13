"""Build a six-case lossless, matched-gain audio comparison (read-only source data)."""
import json, hashlib, shutil
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/listen_audio_cleanup_6_20261026'
OUT=ROOT/'data_previews/listen_benchmark_review/audio_cleanup_6'
METHODS={
 'original':dict(title='原始 listener',note='未处理的 listener 音轨，作为对照。'),
 'df_gentle':dict(title='温和降噪',note='DeepFilterNet3 · 12 dB 抑制上限。使用前后文，补偿模型延迟。'),
 'df_full':dict(title='完整降噪',note='DeepFilterNet3 · 不设抑制上限。可能削弱短促、轻声或非语言发声。'),
 'av':dict(title='视听提取',note='AV-MossFormer2 · 根据 listener 人脸画面提取声音。模型原生 16 kHz，可能削弱笑声或不明显的口部发声。'),
 'reference':dict(title='参考音轨消除',note='实验性线性参考消除；可靠性检查不通过时原样保留。不是已验证的 AEC。'),
 'speaker':dict(title='speaker 参考音轨',note='仅供判断声音来源；这是另一人的音轨，不是处理结果。')}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def db(x):return float(10*np.log10(np.mean(np.asarray(x,dtype=float)**2)+1e-24))
def envelope(x,sr):
 step=round(sr*.02);return [round(float(np.max(abs(x[i:i+step]))),6) for i in range(0,len(x),step)]

def main():
 OUT.mkdir(parents=True,exist_ok=True);cases=json.loads((RUN/'cases.json').read_text());checks=[]
 for c in cases:
  src=RUN/'cases'/c['id'];dst=OUT/'media'/c['id'];dst.mkdir(parents=True,exist_ok=True)
  for fn in ['original.wav','speaker.wav','video.mp4','poster.png','original_labels.json','av_face_crop.png','av_model_crop.png']:
   if (src/fn).exists():shutil.copy2(src/fn,dst/fn)
  y,sr=sf.read(src/'original.wav',dtype='float32');c['methods']={}
  for k in METHODS:
   path=src/(k+'.wav')
   if not path.exists():continue
   z,rate=sf.read(path,dtype='float32');assert rate==sr and len(z)==len(y) and np.isfinite(z).all()
   if k not in ['original','speaker']:shutil.copy2(path,dst/path.name)
   info=json.loads((src/(k+'_info.json')).read_text()) if (src/(k+'_info.json')).exists() else {}
   m=dict(file=f'media/{c["id"]}/{k}.wav',wave=envelope(z,sr),rms_change_db=round(db(z)-db(y),2),peak=float(max(abs(z))),info=info)
   if k not in ['original','speaker']:
    residual=y-z;sf.write(dst/(k+'_removed.wav'),residual,sr,subtype='FLOAT');m['removed_file']=f'media/{c["id"]}/{k}_removed.wav';m['removed_wave']=envelope(residual,sr)
   c['methods'][k]=m
  c['common_peak']=max(max(v['wave']) for k,v in c['methods'].items() if k!='speaker')
  checks.append(dict(id=c['id'],samples=len(y),sr=sr,all_methods_exact_length=True,original_copy_sha_matches=sha(src/'original.wav')==c['source_hashes']['original.wav'],source_listener_unchanged=sha(Path(c['original_source']['listener']))==c['source_hashes']['original.wav'],source_speaker_unchanged=sha(Path(c['original_source']['speaker']))==c['source_hashes']['speaker.wav'],methods=list(c['methods'])))
 payload=dict(schema='listener-cleanup-trial-v1',training_applied=False,methods=METHODS,cases=cases)
 (OUT/'review.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2));(RUN/'verification.json').write_text(json.dumps(checks,indent=2))
 assert all(c['source_listener_unchanged'] and c['source_speaker_unchanged'] and c['original_copy_sha_matches'] for c in checks)
 template=(ROOT/'scripts/audio_cleanup_review.html').read_text();(OUT/'index.html').write_text(template)
 # Self-contained Chinese font subset for this machine's browser environment.
 from fontTools.ttLib import TTFont
 from fontTools import subset
 font=TTFont('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2)
 sub=subset.Subsetter();sub.populate(text=template+json.dumps(payload,ensure_ascii=False));sub.subset(font);font.flavor='woff';font.save(OUT/'review.woff')
 print(json.dumps(dict(cases=len(cases),methods={c['id']:list(c['methods']) for c in cases},output=str(OUT)),ensure_ascii=False))
if __name__=='__main__':main()
