#!/usr/bin/env python3
"""Seeded group-wise random review of the completed AV500h selection.

Reuse the existing final-crop video renderer and exact float32 paired player.
No captions, statistics, selection membership or training caches are changed.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import numpy as np
import orjson
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_j_upright_av500h_v3 as task
from build_j_face224_training_review import render_video,save_audio,write
from listen_benchmark.j_omni import read_audio,check_file,official_metadata
from listen_audio_postprocess.bulk import official
RUN=task.WORK
WORK=ROOT/'experiments/listen_j_av500h_random24_review_20261105'
OUT=ROOT/'data_previews/listen_benchmark_review/j_av500h_random24_20261105'
SEED='j-av500h-selected-random24-20261105-v1'
QUOTAS={'main_face224':16,'saved_audio_supplement':8}
GROUPS={'main_face224':'主集','saved_audio_supplement':'有声补充'}

def sample():
 frozen=WORK/'sampling.json'
 if frozen.exists():
  r=task.read(frozen);assert r['selection_complete_sha256']==task.sha(RUN/'selection/COMPLETE.json');return r
 delivery=task.read(RUN/'DELIVERY_COMPLETE.json');assert delivery['status']=='complete'
 assert task.sha(RUN/'selection/index.sqlite')==delivery['selection_files']['index.sqlite']['sha256']
 c=sqlite3.connect(f'file:{RUN}/selection/index.sqlite?mode=ro',uri=True);c.row_factory=sqlite3.Row
 heaps={k:[] for k in QUOTAS};population=Counter()
 for r in c.execute('SELECT sid,split,selection_group,file,offset,caption_sha256 FROM samples ORDER BY sid'):
  assert r['split']=='train';g=r['selection_group'];population[g]+=1
  rank=int.from_bytes(hashlib.sha256((SEED+'\0'+g+'\0'+r['sid']).encode()).digest(),'big')
  entry=(-rank,r['sid'],dict(r));h=heaps[g]
  if len(h)<QUOTAS[g]:heapq.heappush(h,entry)
  elif entry>h[0]:heapq.heapreplace(h,entry)
 c.close();assert population=={'main_face224':264063,'saved_audio_supplement':20201}
 picks=[]
 for g in QUOTAS:
  for i,(neg,sid,row) in enumerate(sorted(heaps[g],reverse=True),1):
   with Path(row['file']).open('rb') as f:f.seek(row['offset']);raw=f.readline()
   label=orjson.loads(raw);assert label['sample_id']==sid and label['selection_group']==g and label['caption_file_sha256']==row['caption_sha256']
   picks.append(dict(number=len(picks)+1,review_id=('M' if g=='main_face224' else 'V')+f'{i:02d}',sample_id=sid,selection_group=g,selection_file=row['file'],selection_offset=row['offset'],selection_row_sha256=hashlib.sha256(raw).hexdigest(),rank=str(-neg),caption_sha256=row['caption_sha256'],pair=label['pair']))
 r=dict(seed=SEED,sampling='Deterministic SHA256 priority sampling without replacement within each selection group; 16 main + 8 voiced supplement. No caption/content/quality-based replacement.',quotas=QUOTAS,population_by_group=dict(population),population=284264,population_hours=delivery['hours'],representative_prevalence_estimate=False,test_included=False,selection_complete_sha256=task.sha(RUN/'selection/COMPLETE.json'),selection_train_sha256=delivery['selection_files']['train.jsonl']['sha256'],selected=picks)
 write(frozen,r);return r

def build_one(pick):
 sid=pick['sample_id'];dest=OUT/'media'/sid;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'sample.json').exists():
  r=task.read(dest/'sample.json');assert r['audit']['selection_row_sha256']==pick['selection_row_sha256'];return r
 with Path(pick['selection_file']).open('rb') as f:f.seek(pick['selection_offset']);row_raw=f.readline()
 assert hashlib.sha256(row_raw).hexdigest()==pick['selection_row_sha256'];row=orjson.loads(row_raw)
 snap=task.from_manifest(row)
 with Path(row['statistics_file']).open('rb') as f:f.seek(row['statistics_offset']);stat_raw=f.readline()
 stat=orjson.loads(stat_raw);assert stat['sample_id']==sid and stat['caption_file_sha256']==row['caption_file_sha256'] and stat['source']==row['source'] and stat['caption']==row['caption']
 cap_raw=Path(row['caption_file']).read_bytes();cap=orjson.loads(cap_raw);src=row['source']
 check_file(src['source_result'],src['source_result_sha256']);pair=task.read(src['source_result']);clip=pair['clips'][src['source_clip_index']]
 assert clip['sample_id']==sid and clip['listener_id']==src['listener_id'] and clip['speaker_id']==src['speaker_id'] and clip['split']=='train'
 assert abs(clip['start']-src['source_start_s'])<1e-6 and abs(clip['end']-src['source_end_s'])<1e-6
 original_media=cap.get('original_effective_source',cap['source'])['source']
 assert clip['sources']['listener']==original_media
 wave,sr,identity=read_audio(row['audio_recipe'])
 assert identity['processed_waveform_sha256']==cap['input_evidence']['audio_identity']['processed_waveform_sha256']
 assert (not bool(np.any(wave)))==row['listener_audio_is_silent']
 listener=save_audio(dest,'listener',wave,sr)
 a=clip['normalized_audio']['speaker'];assert a['contract']['source']==clip['sources']['speaker']
 assert Path(clip['sources']['speaker']['member']).stem==src['speaker_id'];check_file(a['path'],a['sha256'])
 with sf.SoundFile(a['path']) as f:
  assert f.channels==1 and f.samplerate==sr==a['sample_rate'];first,last=round(src['source_start_s']*sr),round(src['source_end_s']*sr);assert 0<=first<last<=len(f);f.seek(first);sw=f.read(last-first,dtype='float32',always_2d=True)
 assert abs(len(sw)-len(wave))<=1;speaker=save_audio(dest,'speaker',sw,sr)
 video=render_video(src,dest)
 subprocess.run(['ffmpeg','-y','-v','error','-threads','1','-i',str(dest/'video_only.mp4'),'-i',str(dest/'listener.m4a'),'-map','0:v:0','-map','1:a:0','-c','copy','-t',str(src['duration_s']),'-movflags','+faststart',str(dest/'listener.mp4')],check=True,capture_output=True)
 (dest/'video_only.mp4').unlink()
 # Official JSON belongs to the original listener archive, including restored rows.
 metadata=official_metadata(original_media,WORK/'review_metadata');assert metadata['provenance']['sha256']==row['audio_recipe']['provenance']['sha256']
 off=official(metadata['raw'],row['audio_recipe']['source']['first_sample']/sr,src['duration_s'])
 cleaned={k:[{q:v for q,v in x.items() if q in ('id','start','end')} for x in off[k]] for k in ('words','sentences','vad')}
 assert cleaned==row['audio_recipe']['official_listener']
 (dest/'caption_original.json').write_bytes(cap_raw);(dest/'statistics_original.json').write_bytes(stat_raw);(dest/'selection_record.json').write_bytes(row_raw)
 write(dest/'audio_recipe.json',row['audio_recipe']);write(dest/'crop_provenance.json',dict(final=row['crop_provenance'],annotation=row['crop_provenance']))
 item=dict(id=sid,number=pick['number'],review_id=pick['review_id'],selection_group=pick['selection_group'],selection_group_label=GROUPS[pick['selection_group']],split='train',pair=src['pair'],group_id=src['group_id'],duration=src['duration_s'],source_start_s=src['source_start_s'],source_end_s=src['source_end_s'],listener_id=src['listener_id'],speaker_id=src['speaker_id'],video=video,media=f'media/{sid}/',listener_audio=listener,speaker_audio=speaker,face_p10=row['face_p10'],face_p50=row['face_p50'],face_valid_fraction=stat['face_sample_detection_fraction'],source_size=src['video_transform']['source_size'],final_crop=src['video_transform']['crop'],final_crop_revision=row['crop_provenance']['revision'],annotation_crop=src['video_transform']['crop'],annotation_origin='corrected_crop_pipeline',annotation_crop_differs_from_final=False,original_crop=cap['original_source']['video_transform']['crop'],caption=row['caption'],types=stat['counted_types'],behavior_quantities=stat['behavior_quantities'],nod=stat['nod'],official_listener=off,audio_support=stat['audio'],speaker_text=clip.get('text',''),caption_review_flags=row['review_flags'],stat_status=stat['record_status'],training_admitted=False,human_ground_truth=False,orientation_restored='orientation_restoration' in row,source_paths=dict(caption=row['caption_file'],statistics=row['statistics_file'],statistics_offset=row['statistics_offset'],selection=pick['selection_file'],selection_offset=pick['selection_offset'],pair=src['source_result'],measurement=row['quality']['measurement_file'],normalized_listener=src['normalized_audio']['path'],normalized_speaker=a['path']),audit=dict(caption_sha256=row['caption_file_sha256'],statistic_line_sha256=hashlib.sha256(stat_raw).hexdigest(),selection_row_sha256=pick['selection_row_sha256'],listener_audio_identity=identity,final_crop_verified=True,official_metadata_and_recipe_equal=True,paired_audio_sample_difference=len(sw)-len(wave),reference_same_first_video_frame=True,preview_sha256=task.sha(dest/'listener.mp4')))
 write(dest/'sample.json',item);print('BUILT',pick['review_id'],sid,flush=True);return item

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--sample-only',action='store_true');args=p.parse_args();assert 1<=args.workers<=2
 WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);frozen=sample()
 if args.sample_only:print(json.dumps(frozen));return
 with ThreadPoolExecutor(args.workers) as pool:items=list(pool.map(build_one,frozen['selected']))
 data=dict(schema='j-av500h-random24-current-source-review-v1',seed=SEED,samples=items,population_count=284264,population_hours=frozen['population_hours'],sampling=frozen['sampling'],selection_train_sha256=frozen['selection_train_sha256'],representative_prevalence_estimate=False,new_training_manifest_created=True,new_training_latents_created=False)
 write(OUT/'data.json',data)
 html=(ROOT/'scripts/j_face224_training_review.html').read_text()
 html=html.replace('468h候选池：最终裁框与训练素材复核','最新500h训练选集：随机24条复核')
 html=html.replace('最终裁框＋listener音视频＋speaker音频＋原文标注','最新500h训练选集 · 随机24条')
 html=html.replace('来源：最终保留的 train＋dev、原始人脸短边 p10≥224 px，共267,070段／468.18h。固定随机抽取20条train＋4条dev，未按“效果好坏”挑选。','来源：最终500h训练选集，284,264条，全部train。组内固定随机抽主集16条（M01–M16）、有声补充8条（V01–V08），不按caption内容或观看效果挑换。补充组超额抽取，抽样比例不代表全量比例。')
 html=html.replace('<b>这是训练素材复核，不是已完成训练缓存的回放。</b>这批J数据尚未导出新的训练清单和latent缓存。此处按最终修正版裁框、原配方净化音频及现有25fps／512规格重放；原split与准入标记保持不变。部分历史标注来自旧框，会明确标出。','<b>500h清单已完成，本页重放其中的实际素材。</b>展示最新caption、最终裁框、原配方净化音频和冻结v3行为／次数统计。视频为512×512、25fps试听副本；训练张量缓存尚未重建。默认仅听listener，可切换speaker或双方同时听。')
 html=html.replace('<option value="train">train</option><option value="dev">dev</option>','<option value="main_face224">主集16条</option><option value="saved_audio_supplement">有声补充8条</option>')
 html=html.replace("const storage='j-face224-final-review-20261101'","const storage='j-av500h-random24-review-20261105'")
 html=html.replace('`样本${current.number} · ${current.split}', '`${current.review_id} · ${current.selection_group_label} · ${current.split}')
 html=html.replace("[current.split,'最终修正版裁框'","[current.split,current.selection_group_label,...(current.caption_review_flags.length?['原caption复核标记：'+current.caption_review_flags.join(', ')]:[]),'最终修正版裁框'")
 html=html.replace("f==='all'||f===s.split||","f==='all'||f===s.selection_group||f===s.split||")
 html=html.replace('${s.number} · ${s.split}', '${s.review_id} · ${s.selection_group_label} · ${s.split}')
 html=html.replace('new_training_manifest_created:false','new_training_manifest_created:true')
 html=html.replace('j-face224-review-feedback','j-av500h-random24-review-feedback')
 html=html.replace("localStorage.removeItem('j-face224-final-review-20261101')","localStorage.removeItem('j-av500h-random24-review-20261105')")
 (OUT/'index.html').write_text(html)
 font=ROOT/'data_previews/listen_benchmark_review/j_caption_quality_20261105/review.woff'
 if font.exists():shutil.copyfile(font,OUT/'review.woff')
 write(WORK/'build_audit.json',dict(samples=len(items),groups=dict(Counter(r['selection_group'] for r in items)),unique_ids=len({r['id'] for r in items}),unique_pairs=len({r['pair'] for r in items}),listener_nonzero=sum(not r['listener_audio']['full_silence'] for r in items),ordinary_caption_review=sum(bool(r['caption_review_flags']) for r in items),orientation_restored=sum(r['orientation_restored'] for r in items),all_caption_and_statistics_and_selection_shas_verified=True,all_audio_waveform_and_official_recipe_verified=True,template_source_sha256=task.sha(ROOT/'scripts/j_face224_training_review.html'),data_sha256=task.sha(OUT/'data.json'),source_labels_statistics_selection_changed=False,new_training_caches=False))
 print('COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
