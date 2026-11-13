#!/usr/bin/env python3
"""Publish frozen v4 results/proposals; never run models or modify labels."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.semantic_assignment_prompt import source_payload, digest
from listen_j_annotation_nod import render_video

WORK=ROOT/'experiments/listen_qwen30b_assignments_20261030'
BASE=ROOT/'data_previews/listen_benchmark_review'
OUT=BASE/'semantic_qwen30b_20261030'


def read(p):return json.loads(p.read_text())


def media_source(job,prior,bundles,targets):
    sid=job['record']['sample_id'];role=job['provenance']['role']
    if role=='j50_visual_nod_pilot':
        return {'kind':'j','source':bundles[sid]['source'],
            'note':'本轮 J 片段：同一 listener 预裁框；播放原始独立 listener 音轨，可能有残留串音。本轮未重新处理音频。'}
    if role=='prior15_visual_nod_pilot':
        s=prior[sid];base=BASE/'prompt_versions_20261029'
        return {'kind':'existing','video':str(base/s['video']),'audio':str(base/s['audio']['original']),
            'note':'此前15段：沿用原裁剪视频，播放原始独立 listener 音轨。'}
    if role=='current_postprocessed':
        return {'kind':'existing','video':targets[sid]['target_video'],'audio':targets[sid]['target_audio'],
            'note':'历史回归：沿用该样本已有 listener 预览音轨；本轮仅检查文本整合，不重新静音或转写。'}
    return {'kind':'existing','video':str(BASE/'media'/sid/'video.mp4'),
        'audio':str(BASE/'media'/sid/'audio.wav'),
        'note':'历史回归：沿用此前复核页面的 listener 音轨，不代表新的 J 数据音频处理结果。'}


def make_media(sample,spec):
    folder=OUT/'media'/sample['id'];folder.mkdir(parents=True,exist_ok=True)
    movie=folder/'listener.mp4';identity=digest(dict(spec=spec,duration=sample['duration']))
    if movie.exists() and (folder/'meta.json').exists():
        meta=read(folder/'meta.json')
        if meta['identity']!=identity:raise ValueError('Media identity changed')
        return sample['id'],meta
    with tempfile.TemporaryDirectory(prefix='preview-',dir=folder) as td:
        tmp=Path(td);origin=0.;detail={}
        if spec['kind']=='j':
            src=spec['source'];video,detail=render_video({'sample_id':sample['id'],'source':src},tmp)
            audio=src['normalized_audio']['path'];origin=src['source_start_s']
            assert src['listener_id'] in Path(audio).name
        else:video=Path(spec['video']);audio=spec['audio']
        assert video.is_file() and Path(audio).is_file(),(video,audio)
        temp=tmp/'mux.mp4'
        cmd=['ffmpeg','-nostdin','-v','error','-y','-threads','1','-i',str(video),
            '-ss',str(origin),'-threads','1','-i',str(audio),'-t',str(sample['duration']),
            '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','160k',
            '-threads','1','-movflags','+faststart',str(temp)]
        subprocess.run(cmd,check=True,capture_output=True,timeout=120)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(temp)],text=True))
        assert any(s['codec_type']=='audio' for s in probe['streams'])
        assert any(s['codec_type']=='video' for s in probe['streams'])
        assert abs(float(probe['format']['duration'])-sample['duration'])<.15
        temp.replace(movie)
    meta={'identity':identity,'preview_only':True,'audio_note':spec['note'],'source':spec,
        'render':detail,'duration_s':float(probe['format']['duration']),
        'sha256':hashlib.sha256(movie.read_bytes()).hexdigest()}
    (folder/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    return sample['id'],meta


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prior={s['id']:s for s in read(BASE/'prompt_versions_20261029/data.json')['samples']}
    bundles={j['sample_id']:j for j in map(json.loads,(ROOT/'experiments/listen_qwen30b_fusion4_20261030/j50_bundles.jsonl').read_text().splitlines())}
    targets={s['sample_id']:s for s in read(ROOT/'experiments/p2_train50h_postprocessed_reselected_20261027/manifests/targets.json')}
    samples=[];specs={}
    for arm in ['all65','regression8']:
        run=WORK/'runs'/arm;protocol=read(run/'protocol.json');summary=read(run/'summary.json');audit=read(run/'audit.json')
        jobs=[json.loads(l) for l in (WORK/f'{arm}.jsonl').read_text().splitlines()]
        assert digest(jobs)==protocol['input_sha256'] and audit['verified']==summary['accepted']
        rejected={x['job_id']:x for x in summary['rejected']}
        for job in jobs:
            record=job['record'];sid=record['sample_id'];key=job['job_id'];resultfile=run/'results'/f'{key}.json'
            result=read(resultfile) if resultfile.exists() else None
            if result:
                assert result['source_record']==record and result['provenance']['protocol_sha256']==digest(protocol)
                attempt=result['provenance']['attempt']
            else:
                assert key in rejected
                available=[n for n in range(3) if (run/'attempts'/f'{key}.proposal.{n}.json').exists()]
                attempt=max(available) if available else None
            raw=None;proposed=[];review=None
            if attempt is not None:
                entry=read(run/'attempts'/f'{key}.proposal.{attempt}.json')
                assert entry['protocol_sha256']==digest(protocol)
                raw=json.loads(entry['raw']);by={s['source_event_id']:s for s in source_payload(record)['visual_sources']}
                originals={s['event_id']:s for s in record['events']}
                for a in raw['assignments']:
                    if a['target_id']!=a['source_id']:continue
                    src=by[a['source_id']];i=a['quote_index'];assert 0<=i<len(src['quote_options'])
                    origin=originals[a['source_id']]
                    proposed.append({'event_id':a['source_id'],'start':origin['start'],'end':origin['end'],
                        'description':src['quote_options'][i],'modality':'visual',
                        'source_event_ids':[x['source_id'] for x in raw['assignments'] if x['target_id']==a['source_id']],
                        'timing_source_id':a['source_id'],'proposal_only':True})
                rp=run/'attempts'/f'{key}.review.{attempt}.json'
                if rp.exists():review=json.loads(read(rp)['raw'])
            original=deepcopy(record['events'])
            sample={'id':sid,'job_id':key,'cohort':arm,'role':job['provenance']['role'],
                'short':sid.split('_')[2],'duration':record['duration_s'],'status':'accepted' if result else 'rejected',
                'originals':original,'events':deepcopy(result['events']) if result else [],'proposed':proposed,
                'decisions':result.get('source_decisions',[]) if result else [],'raw_proposal':raw,
                'review':review,'error':rejected.get(key),'attempt':attempt,
                'video':f'media/{sid}/listener.mp4','source_sha256':digest(record),'protocol_sha256':digest(protocol),
                'training_admitted':False,'focus':[]}
            if sample['short']=='S1400':sample['focus']=['未通过：查看点头是否保留，以及长区间是否吞并互不重叠的动作。']
            if sample['short']=='S0220':sample['focus']=['预检查拒绝：推演后面还有可见动作，无法安全截取单一前缀；本段没有新模型提案。']
            if sample['short']=='S5193':sample['focus']=['18条视觉→11条，3条音频保留。检查重复是否减少、不同时间的动作是否仍分开。','仍有 showing continued engagement 推演残留，自动通过不是完全正确。']
            if sample['short']=='S0127':sample['focus']=['检查约5.56–6.2秒点头是否仍在；注意 as if beginning to speak 推测仍有残留。']
            specs[sid]=media_source(job,prior,bundles,targets);sample['audio_note']=specs[sid]['note'];samples.append(sample)
    assert len(samples)==73 and len({s['id'] for s in samples})==73
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(make_media,s,specs[s['id']]) for s in samples]
        for i,f in enumerate(as_completed(futures),1):
            sid,_=f.result();print(f'media {i}/73 {sid}',flush=True)
    data={'schema':'qwen30b-assignments-review-v1','training_admitted':False,'human_ground_truth':False,
        'run_id':WORK.name,'model':'Qwen3-30B-A3B-Instruct-2507','samples':samples,
        'summary':{'all65':{'total':65,'accepted':29},'regression8':{'total':8,'accepted':6}},
        'note':'本页为已完成的 v4 结果，不是后来讨论的局部整合／失败原文回退新流程。'}
    temp=OUT/'data.tmp';temp.write_text(json.dumps(data,ensure_ascii=False));temp.replace(OUT/'data.json')
    # This page has new Chinese text; the older page's subset font lacks glyphs.
    from fontTools import subset
    from fontTools.ttLib import TTFont
    fonts=[Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    font=TTFont(next(p for p in fonts if p.is_file()),fontNumber=2)
    options=subset.Options();options.flavor='woff'
    options.desubroutinize=True;options.layout_features=[]
    sub=subset.Subsetter(options=options)
    sub.populate(text=(ROOT/'scripts/qwen30b_assignment_review.html').read_text()+(OUT/'data.json').read_text())
    sub.subset(font);font.flavor='woff';font.save(OUT/'review.woff')
    shutil.copyfile(ROOT/'scripts/qwen30b_assignment_review.html',OUT/'index.html')
    (WORK/'preview.json').write_text(json.dumps({'url':'http://localhost:8876/listen_benchmark_review/semantic_qwen30b_20261030/','samples':73,'media_with_listener_audio':73,'data_sha256':hashlib.sha256((OUT/'data.json').read_bytes()).hexdigest()},indent=2)+'\n')
    print(OUT,flush=True)


if __name__=='__main__':main()
