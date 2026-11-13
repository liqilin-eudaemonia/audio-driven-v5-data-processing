"""Export recovery proposals and stratified identity controls for listening."""
import argparse,concurrent.futures,json,random,shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import evaluate_listen_v8_recovery as evaluation
import publish_listen_v7_sample as exporter
from p3_listen.common import read,write,sha
ROOT=evaluation.ROOT;EXP=evaluation.OUT;OUT=ROOT/'data_previews/seamless_listen_v8_recovery'
KINDS={'asr':'ASR 区间误拦修复','boundary_soft':'边界软疑点',
    'head_shoulders':'头肩裁剪新增候选',
    'rejected_content':'仍被筛掉 · 内容/角色问题',
    'rejected_audio':'仍被筛掉 · 音质问题',
    'rejected_boundary':'仍被筛掉 · 边界硬疑点',
    'identity_only_clear':'仅身份不确定 · 边界正常','identity_only_soft_review':'仅身份不确定 · 边界软疑点',
    'identity_with_reference_match':'对照 · 同时有串音线索','identity_with_other_holds':'对照 · 同时有其他疑点'}

def select(final,include_crop=False):
    path=OUT/'base_selection.json'
    if path.exists():selected=read(path)
    else:
        reservoirs=read(EXP/'review_reservoirs.json');selected=[];groups=set();rng=random.Random(2026102308)
        for kind,n in [('boundary_soft',6),('identity_only_clear',4),('identity_only_soft_review',4),('identity_with_reference_match',2),('identity_with_other_holds',2)]:
            rows=list(reservoirs[kind]);rng.shuffle(rows);added=0
            for row in rows:
                if row['group_id'] in groups:continue
                selected.append(dict(row,review_kind=kind));groups.add(row['group_id']);added+=1
                if added==n:break
            assert added==n,(kind,added)
        write(path,selected)
    if final:
        groups={r['group_id'] for r in selected};rows=read(EXP/'asr_review_candidates.json');random.Random(2026102309).shuffle(rows);added=[]
        for row in rows:
            if row['split']!='train' or row['admission_pool']!='primary' or row['group_id'] in groups:continue
            added.append(dict(row,review_kind='asr'));groups.add(row['group_id'])
            if len(added)==6:break
        selected=added+selected
    if include_crop:
        pairs={r['pair'] for r in selected};rows=read(ROOT/'experiments/p3_listen_v8_downward_crop_preview_20261023/review_candidates.json')
        random.Random(2026102310).shuffle(rows);added=[]
        for row in rows:
            if row['split']!='train' or row['admission_pool']!='primary' or row['pair'] in pairs:continue
            added.append(dict(row,review_kind='head_shoulders'));pairs.add(row['pair'])
            if len(added)==6:break
        selected=added+selected
        for row in read(ROOT/'experiments/p3_listen_v8_downward_crop_preview_20261023/rejected_controls.json'):
            if row['pair'] not in pairs:
                selected.append(row);pairs.add(row['pair'])
    for row in selected:
        row['preview']=f"media/{row['sample_id']}/";row['review_label']=KINDS[row['review_kind']]
    return selected

def export(row):
    folder=OUT/row['preview'];marker=folder/'export.json'
    if marker.exists():
        old=read(marker)
        if old.get('media_verified') and all(old[k]==row[k] for k in ('start','end','sources','listener_crop','normalized_audio')):
            assert all(sha(folder/(mode+'.mp4'))==info['sha256'] for mode,info in old['media_checks'].items())
            result=dict(old,**row)
        else:result=exporter.export(row)
    else:result=exporter.export(row)
    context_start=max(0,row['start']-8);tracks=[];context_end=row['end']+8
    for role in ('speaker','listener'):
        with sf.SoundFile(row['normalized_audio'][role]['path']) as f:
            context_end=min(context_end,len(f)/f.samplerate)
    n=round((context_end-context_start)*48000)
    for role in ('speaker','listener'):
        with sf.SoundFile(row['normalized_audio'][role]['path']) as f:
            f.seek(round(context_start*48000));tracks.append(f.read(n,dtype='float32'))
    assert len(tracks[0])==len(tracks[1])==n
    sf.write(folder/'context.wav',np.stack(tracks,axis=1),48000,subtype='PCM_24')
    meta={};evidence={}
    for role in ('speaker','listener'):
        src=row['sources'][role];raw=evaluation.baseline.full.legacy.metadata(src['archive'],src['member'])
        segs=raw.get('metadata:transcript',[])
        meta[role]=[dict(start=s['start'],end=s['end'],text=s.get('transcript','')) for s in segs
            if s.get('start') is not None and s.get('end') is not None and s['end']>context_start and s['start']<context_end]
        vad,words=evaluation.baseline.full.legacy.annotations(raw,row['start'],row['end'])
        evidence[role]=dict(aligned_words=len(words),vad_seconds=evaluation.recovery.union_seconds(vad))
    result.update(context=dict(start=context_start,end=context_end,clip_offset=row['start']-context_start,transcript=meta),
        review_evidence=dict(raw_annotation_crosscheck=evidence,automatic_reasons=row['recovery']['audio_reasons'],
            listener_reference_max_correlation=row['qc4']['reference']['max_correlation'],
            human_listening_review='pending',note='Annotation and model evidence only; no confirmed voice attribution from these checks.'))
    if row['listener_crop'].get('crop_recovery'):
        src=row['sources']['listener'];c=row['listener_crop']['old_tight_crop']
        uri=f"subfile,,start,{src['offset']},end,{src['offset']+src['byte_size']},,:{src['archive']}"
        exporter.run(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-filter_threads','1','-noautorotate','-ss',str(row['start']),'-i',uri,
            '-vf',f"crop={c['size']}:{c['size']}:{c['x']}:{c['y']},scale=512:512",'-frames:v','1',str(folder/'old_tight_reference.jpg')])
    write(marker,result);return result

def main(final,include_crop=False):
    OUT.mkdir(parents=True,exist_ok=True);exporter.OUT=OUT
    selected=select(final,include_crop);write(OUT/'selection.json',dict(seed=2026102308,samples=selected,training_admitted=False,
        note='Stratified examples from distinct pairs; some crop/exclusion examples share identity components. Not an estimate of acceptance rate.'))
    with concurrent.futures.ThreadPoolExecutor(4) as pool:samples=list(pool.map(export,selected))
    write(OUT/'samples.json',dict(samples=samples,selection_sha256=sha(OUT/'selection.json'),
        summary=read(EXP/'inventory_summary.json'),training_admitted=False))
    shutil.copyfile(ROOT/'scripts/listen_v8_recovery_review.html',OUT/'index.html')
    from fontTools import subset
    from fontTools.ttLib import TTFont
    source=Path('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if source.exists():
        font=TTFont(source,fontNumber=2);sub=subset.Subsetter();sub.populate(text=(OUT/'index.html').read_text()+(OUT/'samples.json').read_text());sub.subset(font);font.flavor='woff';font.save(OUT/'review.woff')
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_v8_recovery'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    print(json.dumps(dict(samples=len(samples),url='http://127.0.0.1:8875/listen_v8_recovery/')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--include-crop',action='store_true');a=p.parse_args();main(a.final,a.include_crop)
