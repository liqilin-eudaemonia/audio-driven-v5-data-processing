#!/usr/bin/env python3
"""Reproducible pass/review sampling and final-crop media replay for human review."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import orjson
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from build_j_face224_training_review import render_video,save_audio
from listen_benchmark.j_omni import read_audio,check_file,official_metadata
from listen_audio_postprocess.bulk import official

WORK=ROOT/'experiments/listen_j_caption_quality_review_20261105'
OUT=ROOT/'data_previews/listen_benchmark_review/j_caption_quality_20261105'
MANIFEST=ROOT/'experiments/listen_j_labels_manifest_20261105'
SEED='j-caption-quality-review-20261105-v1'
BUCKETS={
 'invented_vocal_on_silence':'静音中发声提示',
 'coverage_uncertain':'动作次数／区间覆盖待核',
 'missing_reaction_cue':'反应线索遗漏提示',
 'missing_nod_cue':'点头遗漏提示',
 'missing_vocal_cue':'发声线索遗漏提示',
 'generation_anomaly':'生成重复／未结束',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sample():
    frozen=WORK/'sampling.json'
    if frozen.exists():
        result=json.loads(frozen.read_text())
        summary=json.loads((MANIFEST/'summary.json').read_text())
        assert result['manifest_sha256']==summary['files']['labels']['sha256']
        return result
    audit=json.loads((MANIFEST/'COVERAGE_AUDIT.json').read_text());assert audit['passed']
    quotas={}
    for split,total_per_cell in [('train',5),('dev',1)]:
        for silent in [False,True]:
            for face in ['small','large']:quotas[f'pass:{split}:{int(silent)}:{face}']=total_per_cell
    for bucket in BUCKETS:
        if bucket=='generation_anomaly':quotas['review:'+bucket]=6
        else:
            quotas[f'review:{bucket}:single']=4
            quotas[f'review:{bucket}:multiple']=2
    heaps={k:[] for k in quotas};population=Counter();h=hashlib.sha256();offset=0
    with (MANIFEST/'labels.jsonl').open('rb') as f:
        for raw in f:
            h.update(raw);r=orjson.loads(raw);keys=[]
            if r['split'] in ('train','dev'):
                if r['generation_checks_passed'] and not r['review_flags']:
                    face='large' if r['face_p10'] is not None and r['face_p10']>=224 else 'small'
                    keys=[f'pass:{r["split"]}:{int(r["listener_audio_is_silent"])}:{face}']
                else:
                    flags=set(r['review_flags']);kind='single' if len(flags)==1 else 'multiple'
                    keys=[f'review:{flag}:{kind}' for flag in flags if flag in BUCKETS and flag!='generation_anomaly']
                    if flags & {'repeated_generation','generation_not_finished'}:keys.append('review:generation_anomaly')
            for key in keys:
                population[key]+=1
                rank=int.from_bytes(hashlib.sha256((SEED+'\0'+key+'\0'+r['sample_id']).encode()).digest(),'big')
                item=(-rank,offset,len(raw),r['sample_id'],r['pair'])
                heap=heaps[key]
                if len(heap)<128:heapq.heappush(heap,item)
                elif item>heap[0]:heapq.heapreplace(heap,item)
            offset+=len(raw)
    summary=json.loads((MANIFEST/'summary.json').read_text())
    assert h.hexdigest()==summary['files']['labels']['sha256']
    selected=[];used_ids=set();used_pairs=set()
    # Select small strata first, while preserving one sample per source pair.
    selected_by_key={}
    for key in sorted(quotas,key=lambda k:(population[k],k)):
        chosen=[]
        for neg_rank,pos,length,sid,pair in sorted(heaps[key],reverse=True):
            if sid in used_ids or pair in used_pairs:continue
            chosen.append(dict(sample_id=sid,pair=pair,manifest_offset=pos,manifest_bytes=length,rank=str(-neg_rank),stratum=key))
            used_ids.add(sid);used_pairs.add(pair)
            if len(chosen)==quotas[key]:break
        assert len(chosen)==quotas[key],(key,len(chosen),quotas[key])
        selected_by_key[key]=chosen
    with (MANIFEST/'labels.jsonl').open('rb') as f:
        for key in quotas:
            for item in selected_by_key[key]:
                f.seek(item['manifest_offset']);raw=f.read(item['manifest_bytes']);r=orjson.loads(raw)
                assert r['sample_id']==item['sample_id']
                group='pass' if key.startswith('pass:') else 'review'
                bucket='pass' if group=='pass' else key.split(':')[1]
                selected.append(dict(**item,group=group,bucket=bucket,bucket_label='原自动检查通过' if group=='pass' else BUCKETS[bucket],
                    split=r['split'],review_flags=r['review_flags'],listener_audio_is_silent=r['listener_audio_is_silent'],
                    face_p10=r['face_p10'],caption_file=r['caption_file'],caption_sha256=r['caption_file_sha256'],
                    manifest_row_sha256=hashlib.sha256(raw).hexdigest()))
    for n,item in enumerate(selected,1):
        item['number']=n
        item['review_id']=('P'+str(n).zfill(2)) if item['group']=='pass' else ('R'+str(n-24).zfill(2))
    assert len(selected)==60 and sum(r['group']=='pass' for r in selected)==24 and len(used_pairs)==60
    result=dict(seed=SEED,manifest=str(MANIFEST/'labels.jsonl'),manifest_sha256=h.hexdigest(),
        caption_identity_version=summary['caption_identity_version'],population=summary['exported_captions'],
        sampling='Deterministic SHA256 priority sampling within fixed strata; no replacement; one clip per source pair. Train/dev only.',
        quotas=quotas,stratum_populations=dict(population),selected=selected,
        representative_prevalence_estimate=False,test_included=False,labels_modified=False)
    write(frozen,result)
    print('SAMPLED',len(selected),Counter(r['group'] for r in selected),flush=True)
    return result


def build_one(arg):
    pick,row=arg;sid=pick['sample_id'];dest=OUT/'media'/sid;dest.mkdir(parents=True,exist_ok=True)
    done=dest/'sample.json'
    if done.exists():
        old=json.loads(done.read_text());assert old['audit']['caption_sha256']==pick['caption_sha256']
        return old
    label_raw=Path(row['caption_file']).read_bytes();cap=orjson.loads(label_raw)
    assert hashlib.sha256(label_raw).hexdigest()==row['caption_file_sha256']==pick['caption_sha256']
    assert cap['caption']==row['caption'] and cap['source']==row['source']
    src=cap['source'];original=cap.get('original_source',src)
    measurement=orjson.loads(Path(row['quality']['measurement_file']).read_bytes())
    assert measurement['source']==original and measurement['crop']['new']==src['video_transform']['crop']
    assert cap['crop_provenance']['revision']=='quality-bounded-crop-v1-20261030'
    check_file(src['source_result'],src['source_result_sha256'])
    clip=orjson.loads(Path(src['source_result']).read_bytes())['clips'][src['source_clip_index']]
    assert clip['sample_id']==sid and clip['listener_id']==src['listener_id'] and clip['speaker_id']==src['speaker_id']
    assert clip['sources']['listener']==src['source']
    wave,sr,identity=read_audio(cap['audio_recipe'])
    assert identity['processed_waveform_sha256']==cap['input_evidence']['audio_identity']['processed_waveform_sha256']
    assert bool(not np.any(wave))==row['listener_audio_is_silent']
    listener=save_audio(dest,'listener',wave,sr)
    a=clip['normalized_audio']['speaker'];assert a['contract']['source']==clip['sources']['speaker']
    assert Path(clip['sources']['speaker']['member']).stem==src['speaker_id']
    check_file(a['path'],a['sha256'])
    with sf.SoundFile(a['path']) as f:
        assert f.channels==1 and f.samplerate==sr==a['sample_rate']
        first,last=round(src['source_start_s']*sr),round(src['source_end_s']*sr)
        assert 0<=first<last<=len(f);f.seek(first);speaker_wave=f.read(last-first,dtype='float32',always_2d=True)
    assert abs(len(speaker_wave)-len(wave))<=1
    speaker=save_audio(dest,'speaker',speaker_wave,sr)
    video=render_video(src,dest)
    subprocess.run(['ffmpeg','-y','-v','error','-threads','1','-i',str(dest/'video_only.mp4'),'-i',str(dest/'listener.m4a'),
        '-map','0:v:0','-map','1:a:0','-c','copy','-t',str(src['duration_s']),'-movflags','+faststart',str(dest/'listener.mp4')],check=True,capture_output=True)
    (dest/'video_only.mp4').unlink()
    metadata=official_metadata(src['source'],WORK/'review_metadata')
    assert metadata['provenance']['sha256']==cap['audio_recipe']['provenance']['sha256']
    off=official(metadata['raw'],cap['audio_recipe']['source']['first_sample']/sr,src['duration_s'])
    core=[]
    for key in ['words','sentences','vad']:
        core.extend((float(v['start']),float(v['end'])) for v in cap['audio_recipe']['official_listener'][key] if v['end']>v['start'])
    merged=[]
    for start,end in sorted(core):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    ev=cap['input_evidence'];nod=(ev.get('nod_supplement') or {}).get('events') or []
    vocal=ev.get('vocal_supplement') or [];reaction=(ev.get('reaction_supplement') or {}).get('cues') or []
    (dest/'caption_original.json').write_bytes(label_raw)
    write(dest/'manifest_record.json',row);write(dest/'audio_recipe.json',cap['audio_recipe'])
    write(dest/'crop_provenance.json',cap['crop_provenance'])
    item=dict(id=sid,number=pick['number'],review_id=pick['review_id'],group=pick['group'],bucket=pick['bucket'],bucket_label=pick['bucket_label'],
        split=row['split'],pair=src['pair'],group_id=src['group_id'],duration=src['duration_s'],
        source_start_s=src['source_start_s'],source_end_s=src['source_end_s'],listener_id=src['listener_id'],speaker_id=src['speaker_id'],
        video=video,media=f'media/{sid}/',listener_audio=listener,speaker_audio=speaker,
        face_p10=row['face_p10'],face_p50=row['face_p50'],source_size=src['video_transform']['source_size'],
        final_crop=src['video_transform']['crop'],final_crop_revision=cap['crop_provenance']['revision'],
        annotation_crop=src['video_transform']['crop'],annotation_origin='corrected_crop_pipeline',annotation_crop_differs_from_final=False,
        caption=cap['caption'],speaker_text=clip.get('text',''),caption_review_flags=cap['review_flags'],
        remaining_gaps=cap['remaining_gaps'],finish_reason=cap['finish_reason'],
        generation_checks_passed=cap['generation_checks_passed'],nod=dict(candidate_intervals=[[r['start'],r['end']] for r in nod]),
        vocal_cues=vocal,reaction_cues=reaction,official_listener=off,
        audio_support=dict(official_core_s=sum(b-a for a,b in merged),official_core_intervals=merged),
        training_admitted=False,human_ground_truth=False,
        source_paths=dict(caption=row['caption_file'],manifest=str(MANIFEST/'labels.jsonl'),manifest_offset=pick['manifest_offset'],
            pair=src['source_result'],measurement=row['quality']['measurement_file']),
        audit=dict(caption_sha256=pick['caption_sha256'],manifest_row_sha256=pick['manifest_row_sha256'],
            listener_audio_identity=identity,final_crop_verified=True,reference_same_first_video_frame=True,
            preview_sha256=sha(dest/'listener.mp4'),paired_audio_sample_difference=len(speaker_wave)-len(wave)))
    write(done,item)
    print('BUILT',pick['review_id'],sid,flush=True)
    return item


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-only',action='store_true');p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();assert 1<=args.workers<=2
    WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    frozen=sample()
    if args.sample_only:return
    tasks=[]
    with (MANIFEST/'labels.jsonl').open('rb') as f:
        for pick in frozen['selected']:
            f.seek(pick['manifest_offset']);raw=f.read(pick['manifest_bytes'])
            assert hashlib.sha256(raw).hexdigest()==pick['manifest_row_sha256']
            tasks.append((pick,orjson.loads(raw)))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:items=list(pool.map(build_one,tasks))
    data=dict(schema='j-caption-quality-human-review-v1',seed=SEED,samples=items,
        manifest_sha256=frozen['manifest_sha256'],caption_identity_version=frozen['caption_identity_version'],
        sampling=frozen['sampling'],population=frozen['population'],representative_prevalence_estimate=False,
        group_counts=dict(Counter(r['group'] for r in items)),bucket_labels=BUCKETS)
    write(OUT/'data.json',data)
    shutil.copyfile(ROOT/'scripts/j_caption_quality_review.html',OUT/'index.html')
    # The previous page's subset font lacks new review text and user-entered
    # Chinese feedback. Serve the complete existing CJK face for this page.
    font=Path('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        from fontTools.ttLib import TTFont
        complete_font=TTFont(str(font),fontNumber=2)
        complete_font.flavor='woff';complete_font.save(OUT/'review.woff')
    write(WORK/'build_audit.json',dict(samples=len(items),groups=data['group_counts'],
        unique_pairs=len({r['pair'] for r in items}),labels_verbatim=True,audio_waveform_hashes_verified=True,
        all_final_crops_verified=True,source_manifest_sha256=frozen['manifest_sha256'],
        data_sha256=sha(OUT/'data.json'),new_labels_generated=False))
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
