import pathlib,json,hashlib,tarfile,collections,statistics
from PIL import Image,ImageDraw
w=pathlib.Path('experiments/listen_j_upright_regen_20261105');parents={r['sample_id']:r for r in map(json.loads,(w/'rotation_candidates.jsonl').open())};dirs={r['source_key']:r for r in json.loads((w/'rotation_sources_confirmed.json').read_text())};counts=collections.Counter();pixel=[];roll=[];seen=set()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def canon(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
for p in (w/'restored_rows').glob('*.json'):
 d=json.loads(p.read_text());sid=d['sample_id'];assert sid in parents and sid not in seen;seen.add(sid);o=parents[sid];prov=json.loads(pathlib.Path(d['provenance_file']).read_text());src=d['request']['source'];q=d['request']['quality']
 assert d['parent_caption_sha256']==o['caption_file_sha256'] and prov['original_source']==o['source'];s=o['source']['source'];key=hashlib.sha256((s['archive']+'\0'+s['member']).encode()).hexdigest()[:24]
 assert prov['rotation_ccw_degrees']==dirs[key]['rotation_ccw_degrees'];assert prov['native_crop']==o['source']['video_transform']['crop']
 assert not prov['spatial_resize'] and prov['timestamps_preserved'] and prov['external_audio_recipe_unchanged']
 for k in ['sample_id','pair','split','group_id','listener_id','speaker_id','normalized_audio','source_start_s','source_end_s','duration_s']:assert src[k]==o['source'][k]
 assert sha(src['source']['archive'])==prov['derived_archive_sha256']
 with tarfile.open(src['source']['archive']) as t:
  m=t.getmember(src['source']['member']);assert m.offset_data==src['source']['offset'] and m.size==src['source']['byte_size']
 assert o['source_start_s']<=prov['first_frame_s']<o['source_start_s']+.1 and 0<o['source_end_s']-prov['last_frame_s']<.1
 assert sha(q['measurement_file'])==q['measurement_sha256'] and canon(src)==q['source_sha256'];m=json.loads(pathlib.Path(q['measurement_file']).read_text());assert m['source']==src and m['status']=='ok'
 pixel.append(prov['representative_pixel_check']);roll.append(d['roll_median']);counts['total']+=1;counts['usable' if d['restoration_usable'] else 'excluded_by_original_clarity_rules']+=1
assert seen==set(parents) and counts['total']==1699 and counts['usable']==1690
result=dict(status='passed',counts=dict(counts),source_time_audio_participant_split_mismatches=0,archive_hash_mismatches=0,measurement_hash_mismatches=0,representative_pixel_max_mean_error=max(x['mean_abs_error'] for x in pixel),representative_pixel_max_absolute_error=max(x['max_abs_error'] for x in pixel),frames_checked='All rendered frames decode/PTS support checked by restorer; one original-vs-restored pixel comparison per clip',source_direction_reviewed=747)
(w/'RESTORATION_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result)
reverse=[r for r in dirs.values() if r['rotation_ccw_degrees']==-90];out=Image.new('RGB',(len(reverse)*260,320),'white');draw=ImageDraw.Draw(out)
for i,r in enumerate(reverse):
 im=Image.open(r['frame_file']).transpose(Image.Transpose.ROTATE_270);im.thumbnail((250,290));out.paste(im,(i*260,25));draw.text((i*260+2,3),r['review_id']+' corrected clockwise',fill='black')
out.save(w/'clockwise_direction_checked.jpg')
