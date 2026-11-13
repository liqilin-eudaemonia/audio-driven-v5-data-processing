"""Prepare twenty real voiced figure candidates from sealed training metadata.

Selection for illustration only; never rewrites captions, Event labels or media.
The established final-crop/PCM review exporter is reused with two CPU workers.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
WORK=ROOT/'experiments/data_pipeline_figure_cases20_20261112'
OUT=ROOT/'data_previews/listen_benchmark_review/data_pipeline_cases20_20261112'
LTX=ROOT/'experiments/listen_j_upright_av500h_train_dev_v3_20261105/selection'
QWEN=ROOT/'experiments/listen_j_upright_response_sft_20261105/final'
STATS=ROOT/'experiments/listen_j_upright_av500h_v3_20261105/tables'
EVENT=ROOT/'experiments/p2_clip_guarded_current500h_20261106'
NUMERIC=ROOT/'experiments/p2_clip_numeric_guarded_current500h_20261106'
SEED='figure-cases20-20261112-v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def db(p):return sqlite3.connect(Path(p).resolve().as_uri()+'?mode=ro',uri=True)


def select():
    WORK.mkdir(exist_ok=True)
    frozen=WORK/'selection.json'
    if frozen.exists():return read(frozen)
    pool=WORK/'metadata_pool.json'
    if not pool.exists():
        c=db(STATS/'sample_stats.sqlite');c.row_factory=sqlite3.Row
        for name,p in [('ltx',LTX/'index.sqlite'),('qwen',QWEN/'index.sqlite')]:c.execute('ATTACH DATABASE ? AS '+name,(p.resolve().as_uri()+'?mode=ro',))
        sql="""SELECT s.sid,s.pair,s.listener,s.split,s.duration,s.face_p10,s.face_p50,s.types,l.file,l.offset
        FROM samples s JOIN ltx.samples l ON l.sid=s.sid JOIN qwen.samples q ON q.sid=s.sid
        JOIN behaviors b ON b.sid=s.sid AND b.type='audio:speech' AND b.counted=1
        WHERE s.duration BETWEEN 5 AND 8 AND s.face_p10>=224 AND s.types BETWEEN 2 AND 8 AND q.split='train'
        ORDER BY s.face_p10 DESC,s.sid LIMIT 2000"""
        save(pool,dict(query=sql,rows=[dict(r)for r in c.execute(sql)]));c.close()
    import numpy as np
    nm=read(NUMERIC/'manifest.json');names=nm['contract']['classes'];ids=read(NUMERIC/'sample_ids.json')
    arrays={k:np.load(NUMERIC/(k+'.npy'),mmap_mode='r') for k in ['y','known','weight']}
    ec=db(EVENT/'labels.sqlite');reasons=Counter();eligible=[]
    with (LTX/'selected.jsonl').open('rb')as stream:
        for ix in read(pool)['rows']:
            stream.seek(ix['offset']);raw=stream.readline();row=json.loads(raw);sid=row['sample_id']
            assert sid==ix['sid'] and row['split']=='train'
            if row['listener_audio_is_silent']:reasons['silent']+=1;continue
            if row['review_flags'] or not row['generation_checks_passed']:reasons['caption_flag']+=1;continue
            if re.search(r'\b(indicating|suggesting|seems|appears to|as if|thoughtful|engaged)\b',row['caption'],re.I):reasons['interpretation']+=1;continue
            quotes=re.findall(r'"([^"\n]{1,90})"',row['caption'])
            quotes=[q for q in quotes if 0<len(q.split())<=6]
            if not quotes or len(row['caption'].split())>150:reasons['no_short_quoted_response']+=1;continue
            audio=row['audio_statistics'];core=audio['official_core_s']
            if not .10<=core<=3.0 or not 1<=audio['official_bout_count']<=3:reasons['audio_support']+=1;continue
            rec=ec.execute('select ordinal,value from labels where sid=?',(sid,)).fetchone();assert rec
            ordinal,label_raw=rec;assert ids[ordinal]==sid
            targets=[]
            for j,name in enumerate(names):
                y=float(arrays['y'][ordinal,j]);known=bool(arrays['known'][ordinal,j]);weight=float(arrays['weight'][ordinal,j])
                enabled=nm['statistics'][name]['positive']>=5000
                targets.append(dict(category=name,y=y,known=known,weight=weight,class_enabled=enabled,effective_weight=weight*known*enabled))
            positive=[t['category']for t in targets if t['class_enabled'] and t['known'] and t['y']==1]
            if 'audio:speech'not in positive or not any(x.startswith('visual:')for x in positive):reasons['event_support']+=1;continue
            term=re.sub(r'[^a-z -]','',quotes[0].lower()).strip()
            family='yeah'if re.search(r'\b(yeah|yes|yep)\b',term)else'okay'if re.search(r'\b(ok|okay)\b',term)else'right'if'right'in term else'oh'if term.startswith('oh')else'nonlexical'if re.search(r'(hmm|mhm|uh|hm)',term)else'other'
            vis=[x for x in positive if x.startswith('visual:')]
            score=min(row['face_p10'],420)/100+min(len(vis),4)*.45+(.7 if 'visual:head nodding'in vis else 0)+(.5 if len(quotes)==1 else 0)
            score-=abs(row['duration_s']-6.5)*.12+max(0,core-1.5)*.2
            eligible.append(dict(sample_id=sid,metadata=ix,row=row,quotes=quotes,vocal_family=family,positive_event_classes=positive,event_targets=targets,
                sparse_event=json.loads(label_raw),selection_line_sha256=hashlib.sha256(raw).hexdigest(),score=score,
                tie=hashlib.sha256((SEED+sid).encode()).hexdigest()))
    ec.close()
    picked=[];people=Counter();pairs=set();groups=Counter();families=Counter()
    while eligible and len(picked)<20:
        eligible.sort(key=lambda x:(-(x['score']-families[x['vocal_family']]*.85-groups[x['row']['group_id']]*.5),x['tie']))
        chosen=None
        for x in eligible:
            person=x['row']['listener_id'].split('_')[-1]
            if x['row']['pair']not in pairs and people[person]<2:
                chosen=x;break
        if chosen is None:break
        eligible.remove(chosen);r=chosen['row'];person=r['listener_id'].split('_')[-1]
        pairs.add(r['pair']);people[person]+=1;groups[r['group_id']]+=1;families[chosen['vocal_family']]+=1
        chosen['number']=len(picked)+1;chosen['review_id']=f'C{len(picked)+1:02d}';picked.append(chosen)
    assert len(picked)==20
    result=dict(schema='data-pipeline-figure-cases20-v1',seed=SEED,created_at=time.time(),human_ground_truth=False,
        selection='Purposeful large-face voiced illustration candidates, not prevalence sampling. All in final LTX500h and Qwen SFT train; one clip per pair, max two per listener participant.',
        pool_limit=2000,rejections=dict(reasons),vocal_families=dict(families),selected=picked)
    save(frozen,result)
    print('SELECTED',len(picked),families,flush=True)
    return result


def build_one(x):
    import numpy as np
    import soundfile as sf
    import build_j_caption_quality_review as old
    from listen_benchmark.official_audio_gate import make_gate
    old.OUT=OUT;old.WORK=WORK;old.MANIFEST=LTX
    row=x['row'];sid=x['sample_id'];dest=OUT/'media'/sid
    cached=dest/'sample.json'
    if cached.exists():
        item=read(cached)
        if ('original_listener_audio' in item and item.get('event_targets')==x['event_targets']
                and item['audit']['caption_sha256']==row['caption_file_sha256']
                and all((dest/name).exists() for name in ['listener.mp4','original_listener.wav','listener.wav','event_targets.json'])):
            return item
    pick=dict(sample_id=sid,number=x['number'],review_id=x['review_id'],group='pass',bucket='figure',bucket_label='／'.join(x['quotes']),
        caption_sha256=row['caption_file_sha256'],manifest_offset=x['metadata']['offset'],manifest_row_sha256=x['selection_line_sha256'])
    item=old.build_one((pick,row))
    recipe=row['audio_recipe'];a=recipe['source']
    with sf.SoundFile(a['path'])as f:
        assert f.samplerate==a['sample_rate'];f.seek(a['first_sample']);raw=f.read(a['last_sample']-a['first_sample'],dtype='float32',always_2d=True)
    gate,info=make_gate(recipe['official_listener'],len(raw),a['sample_rate'])
    processed=raw*gate[:,None];removed=raw-processed
    assert np.array_equal(np.fromfile(dest/'listener.f32',dtype='<f4').reshape(-1,1),processed)
    original=old.save_audio(dest,'original_listener',raw,a['sample_rate']);suppressed=old.save_audio(dest,'removed_listener',removed,a['sample_rate'])
    for name,wave in [('original_listener',raw),('listener',processed),('removed_listener',removed),('speaker',np.fromfile(dest/'speaker.f32',dtype='<f4').reshape(-1,1))]:
        sf.write(dest/(name+'.wav'),wave,a['sample_rate'],subtype='FLOAT')
    inside=float(np.square(raw[gate>0]).sum());outside=float(np.square(raw[gate==0]).sum())
    item.update(event_targets=x['event_targets'],positive_event_classes=x['positive_event_classes'],quoted_responses=x['quotes'],
        original_listener_audio=original,removed_listener_audio=suppressed,original_caption_verbatim=True,
        in_final_ltx500h=True,in_qwen_sft_train=True,raw_energy_outside_gate_fraction=outside/max(inside+outside,1e-30),
        source_paths=dict(item['source_paths'],manifest=str(LTX/'selected.jsonl')))
    old.write(dest/'event_targets.json',dict(sample_id=sid,targets=x['event_targets'],sparse_event=x['sparse_event'],
        source=str(NUMERIC),semantics='Stored y/known/weight plus frozen class enablement; not temporal labels.'))
    old.write(dest/'sample.json',item)
    return item


def native_keyframes(item):
    """Export four actual source frames with the same final crop, without a model."""
    import av
    from listen_visual_quality.measurement import SourceBytes
    from listen_benchmark.frame_crop import crop_original_frame_reused
    dest=OUT/item['media'];src=read(dest/'manifest_record.json')['source']
    duration=item['duration'];start=src['source_start_s']
    cues=item['vocal_cues'];nod=item['nod']['candidate_intervals']
    candidates=[.16]
    if nod:candidates.append(sum(nod[0])/2)
    if cues:candidates.append((cues[0]['start']+cues[0]['end'])/2)
    candidates += [duration*.35,duration*.65,duration-.2]
    picked=[]
    for t in candidates:
        t=max(0,min(duration-.04,float(t)))
        if all(abs(t-u)>.22 for u in picked):picked.append(t)
        if len(picked)==4:break
    picked.sort();wanted=[start+t for t in picked];result=[]
    with SourceBytes(src['source'])as member,av.open(member)as container:
        stream=container.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=1
        container.seek(int(max(0,start-1)/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
        prev=None;prev_t=None;i=0
        for frame in container.decode(video=0):
            if frame.pts is None:continue
            t=float(frame.pts*frame.time_base)
            while i<len(wanted) and t>=wanted[i]:
                chosen=prev if prev is not None and abs(prev_t-wanted[i])<=abs(t-wanted[i]) else frame
                pts=float(chosen.pts*chosen.time_base)
                filename=f'frame_{i+1:02d}.png';crop_original_frame_reused(chosen,src['video_transform']).save(dest/filename)
                result.append(dict(file=filename,requested_pair_time_s=picked[i],source_pts_s=pts,pair_time_s=pts-start))
                i+=1
            if i==len(wanted):break
            prev=frame;prev_t=t
    assert len(result)==4
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--select-only',action='store_true');p.add_argument('--viewer-only',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    assert 1<=a.workers<=2
    frozen=select()
    if a.select_only:return
    OUT.mkdir(parents=True,exist_ok=True)
    if a.viewer_only:
        items=read(OUT/'data.json')['samples'];assert len(items)==20 and all(len(x['native_keyframes'])==4 for x in items)
    else:
        with ThreadPoolExecutor(max_workers=a.workers)as pool:items=list(pool.map(build_one,frozen['selected']))
        with ThreadPoolExecutor(max_workers=a.workers)as pool:frames=list(pool.map(native_keyframes,items))
        for item,values in zip(items,frames):
            item['native_keyframes']=values
            save(OUT/item['media']/'sample.json',item)
    save(OUT/'data.json',dict(schema='data-pipeline-figure-cases20-v1',seed=SEED,samples=items,
        manifest_sha256=read(LTX/'SUMMARY.json')['manifest_sha256'],caption_identity_version='final-upright',human_ground_truth=False))
    # Reuse the existing synchronized PCM/video viewer; changes are scoped to
    # this new review artifact, never its original template or historical pages.
    template=(ROOT/'scripts/j_caption_quality_review.html').read_text()
    template=template.replace('新标签人工抽检：自动通过24条／复核36条','数据处理流程图：20个有声真实候选')
    template=template.replace('新标签人工抽检：自动通过组／复核组','数据处理流程图素材候选')
    template=template.replace('60条','20条').replace('/60','/20').replace('P01、R01','C01、C02')
    template=template.replace('j-caption-quality-review-20261105-v1','data-pipeline-figure-cases20-20261112-v1')
    template=template.replace('从本轮全量新标签中固定抽取20条，每条来自不同会话pair。自动通过组覆盖train/dev、非零音频/全静音、人脸p10≥224/较小脸；复核组覆盖5类主要提示及生成异常。',
        '20条均来自最终LTX500h与Qwen SFT train交集，按大脸、明确短发声、视觉Event正例和原文质量定向筛选。每条来自不同pair；这是画图候选，不代表总体分布。')
    template=template.replace('自动通过组','素材候选').replace('自动通过24条','素材候选20条')
    template=template.replace('<label>分组 <select','<label hidden>分组 <select').replace('<label>原因 <select','<label hidden>原因 <select')
    template=re.sub(r'<a href="\?group=pass">.*?</a> · <a href="\?group=review">.*?</a> · ','',template)
    template=template.replace('原素材候选','流程图候选')
    template=template.replace('P01–P24','C01–C20').replace('j-caption-quality-review-feedback-20261105.json','data-pipeline-cases20-feedback-20261112.json')
    template=template.replace('<option value="usable">基本可用</option>','<option value="main">选作主case</option><option value="backup">备用case</option>')
    template=template.replace('<option value="label_issue">存在实质标签问题</option>','<option value="skip">暂不选</option>')
    template=template.replace('<h3>本轮新caption（原文）</h3>', '<h3>原生画面关键帧（点击下载）</h3><div id="nativeFrames" style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px"></div><h3>本轮新caption（原文）</h3>')
    template=template.replace('<button data-mode="speaker">speaker音轨</button>',
        '<button data-mode="speaker">speaker音轨</button><button data-mode="original">净化前listener</button><button data-mode="removed">被置零部分</button>')
    template=template.replace("loadPCM(current.speaker_audio,m+'speaker.f32',token)]", "loadPCM(current.speaker_audio,m+'speaker.f32',token),loadPCM(current.original_listener_audio,m+'original_listener.f32',token),loadPCM(current.removed_listener_audio,m+'removed_listener.f32',token)]")
    template=template.replace('buffers={listener:result[0],speaker:result[1]}','buffers={listener:result[0],speaker:result[1],original:result[2],removed:result[3]}')
    template=template.replace('<h3>原自动检查提示（待人工确认）</h3>',
        '<h3>整段Event监督（真实缓存）</h3><p class="muted">1为正例；0＋有效mask为弱负例；—为屏蔽或停用。权重不是动作次数。</p><table><thead><tr><th>类别</th><th>y</th><th>mask</th><th>启用</th><th>有效样本权重</th></tr></thead><tbody id="eventTargets"></tbody></table><p><a id="eventDownload" download>下载Event标签JSON</a> · <a id="rawWav" download>净化前WAV</a> · <a id="cleanWav" download>净化后WAV</a></p><h3>原自动检查提示（待人工确认）</h3>')
    needle="$('reference').src=m+'reference.png';"
    replacement=needle+"$('eventTargets').innerHTML=current.event_targets.map(t=>'<tr><td>'+esc(t.category)+'</td><td>'+((t.known&&t.class_enabled)?t.y:'—')+'</td><td>'+Number(t.known)+'</td><td>'+Number(t.class_enabled)+'</td><td>'+t.effective_weight.toFixed(1)+'</td></tr>').join('');$('eventDownload').href=m+'event_targets.json';$('rawWav').href=m+'original_listener.wav';$('cleanWav').href=m+'listener.wav';"
    replacement+="$('nativeFrames').innerHTML=current.native_keyframes.map(f=>'<a href=\"'+m+f.file+'\" target=\"_blank\" download><img style=\"width:100%\" src=\"'+m+f.file+'\"><small>'+f.pair_time_s.toFixed(2)+'s</small></a>').join('');"
    assert needle in template;template=template.replace(needle,replacement)
    (OUT/'index.html').write_text(template)
    font=ROOT/'data_previews/listen_benchmark_review/j_caption_quality_20261105/review.woff'
    if font.exists() and not (OUT/'review.woff').exists():
        import shutil
        shutil.copyfile(font,OUT/'review.woff')
    with (WORK/'candidates.csv').open('w',encoding='utf-8-sig',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=['review_id','sample_id','duration_s','face_p10','spoken_text','positive_events','caption','outside_gate_energy_fraction','media_directory']);writer.writeheader()
        for x in items:writer.writerow(dict(review_id=x['review_id'],sample_id=x['id'],duration_s=x['duration'],face_p10=x['face_p10'],spoken_text=' | '.join(x['quoted_responses']),positive_events=' | '.join(x['positive_event_classes']),caption=x['caption'],outside_gate_energy_fraction=x['raw_energy_outside_gate_fraction'],media_directory=str(OUT/x['media'])))
    save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),samples=len(items),output=str(OUT),no_relabeling=True,only_cpu=True,
        all_final_crops_verified=all(x['audit']['final_crop_verified']for x in items),all_20_have_stored_audio_speech_positive=True))
    print('READY',str(OUT),flush=True)


if __name__=='__main__':main()
