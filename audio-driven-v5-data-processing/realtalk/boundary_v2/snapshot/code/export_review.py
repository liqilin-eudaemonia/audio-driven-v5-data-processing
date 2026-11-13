from pathlib import Path
import json,subprocess,concurrent.futures,hashlib
R=Path(__file__).resolve().parents[1];O=R.parent/'full_20261023/reports/boundary_v2_review';(O/'media').mkdir(parents=True,exist_ok=True);m=json.loads((R/'manifest.json').read_text())
def ff(args):subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-filter_threads','1',*map(str,args)],check=True)
source=R/'review_source.mp4';ff(['-i',R/'source_video.mp4','-i',R/'source_audio.wav','-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','160k','-movflags','+faststart',source])
p=O/'source.mp4'
if not p.exists():p.symlink_to(source.resolve())
ann=json.loads(Path(m['source']['annotation']).read_text());tracks=[]
for i in range(m['frames']):
 f=ann.get(str(i),{});tracks.append([None if (b:=f.get('people',{}).get(p,{}).get('bbox')) is None else [round(v,1) for v in b] for p in ['p0','p1']])
old=json.loads((R.parent/'full_20261023/manifests'/f"{m['video_id']}.json").read_text());m['old_pairs']=[{k:p[k] for k in ['pair_id','start_frame','end_frame','speaker','listener']} for p in old['pairs']];m['tracks']=tracks
exports=[]
def export(p):
 c=p['crop']
 if not c:return {'pair_id':p['pair_id'],'status':'crop_unresolved_use_full_frame_review'}
 dst=O/'media'/(p['pair_id']+'.mp4');base=O/'media'/(p['pair_id']+'_video.mp4');ff(['-ss',p['start'],'-i',R/'source_video.mp4','-vf',f"crop={c['width']}:{c['height']}:{c['x']}:{c['y']}:exact=1",'-frames:v',p['end_frame']-p['start_frame'],'-an','-c:v','libx264','-preset','veryfast','-crf','20',base]);ff(['-i',base,'-ss',p['start'],'-i',R/'source_audio.wav','-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','128k','-t',p['duration'],'-movflags','+faststart',dst]);base.unlink();probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',dst],text=True));v=next(s for s in probe['streams'] if s['codec_type']=='video');assert int(v['nb_frames'])==p['end_frame']-p['start_frame'];assert int(v['width'])==c['width'];aud=next(s for s in probe['streams'] if s['codec_type']=='audio');assert abs(float(aud['duration'])-p['duration'])<.08;return {'pair_id':p['pair_id'],'status':'exported','path':'media/'+dst.name,'audio':'original_mixture_not_isolated_listener','frames':int(v['nb_frames'])}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for i,x in enumerate(pool.map(export,m['pairs']),1):
  exports.append(x)
  if i%20==0:print('EXPORTED',i,flush=True)
m['exports']={x['pair_id']:x for x in exports};(O/'data.json').write_text(json.dumps(m,ensure_ascii=False,separators=(',',':')));(R/'exports.json').write_text(json.dumps(exports,ensure_ascii=False,indent=2));(O/'manifest.json').write_text((R/'manifest.json').read_text());(O/'summary.json').write_text((R/'summary.json').read_text());print('EXPORT_DONE',len(exports),flush=True)
