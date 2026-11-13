"""Export every non-retained connected interval in the two audited conversations.

These are human-review excerpts, not new accepted training pairs. Context is
retained and labelled separately. No production screening decisions are edited.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent
OUT=EXP/'review/manual_review';OUT.mkdir(exist_ok=True)
PREVIEW=ROOT/'data_previews/seamless_listen_full_source_recovery'
PAIRS=['naturalistic_V02_S3579_I00000175','naturalistic_V03_S1909_I00000205']
def read(p):return json.loads(p.read_text())
def overlap(a,b,x,y):return min(b,y)>max(a,x)+1e-6
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(pair,ordinal):
    review=read(PREVIEW/pair/'review.json');replay=read(EXP/(pair+'.json'))
    prodpath=ROOT/'data_corpora/seamless_listen_local_full/results'/(pair+'.json')
    prevpath=ROOT/'data_corpora/seamless_listen_v8_audio_final_full/results'/(pair+'.json')
    prod,prev=read(prodpath),read(prevpath)
    lookup={r['sample_id']:r for r in prev['clips']+prod['clips']+prod['superseded_attempts']}
    groups=[]
    for row in review['timeline']:
        if row['status']=='keep':continue
        if groups and abs(row['start']-groups[-1]['end'])<=1e-6:
            groups[-1]['end']=row['end'];groups[-1]['timeline'].append(row)
        else:groups.append(dict(start=row['start'],end=row['end'],timeline=[row]))
    rows=[]
    for n,g in enumerate(groups,1):
        a,b=g['start'],g['end'];sid=f'S{ordinal}-{n:02d}'
        words=[dict(w,participant='AB'[si]) for si,p in enumerate(replay['people']) for w in p['words']
               if overlap(a,b,w['start'],w['end'])]
        words.sort(key=lambda w:(w['start'],w['end']))
        quality=any(t['status'] in ('withheld','blocked') for t in g['timeline'])
        category='quality' if quality else 'speech' if words else 'tiny' if b-a<.12 else 'other'
        parent_ids={sid for t in g['timeline'] for sid in t['clip_ids']}
        blocks=[x for x in review['diagnostic']['blocked'] if overlap(a,b,x['start'],x['end'])]
        parent_ids.update(x['parent_id'] for x in blocks)
        evidence=[]
        for parent in sorted(parent_ids):
            p=lookup[parent];r=p['audio_final']
            windows=[]
            for role,track in p['qc3'].items():
                for w in track['events']+track['background']:
                    x,y=p['start']+w['start'],p['start']+w['end']
                    if not overlap(a,b,x,y) or not w['decision']['reasons']:continue
                    model=w['panns']
                    windows.append(dict(start=x,end=y,role=role,participant='AB'[p[role+'_index']],
                        noise_label=model['noise_label'],noise_score=model['noise_score'],
                        human_label=model['human_label'],human_score=model['human_score'],
                        rms_dbfs=w['features']['rms_dbfs'],reasons=w['decision']['reasons']))
            reference=[dict(w,start=p['start']+w['start'],end=p['start']+w['end'])
                       for w in p['qc4']['reference']['windows'] if w.get('suspicious') and
                       overlap(a,b,p['start']+w['start'],p['start']+w['end'])]
            evidence.append(dict(parent_id=parent,parent_start=p['start'],parent_end=p['end'],
                                 pool=r['pool'],reasons=r['reasons'],quality=r['quality'],
                                 model_windows=windows,reference_windows=reference))
        lo,hi=max(0,a-3),min(replay['limit'],b+3)
        transcripts=[[dict(start=s['start'],end=s['end'],text=s['text']) for s in p['segments']
                      if overlap(lo,hi,s['start'],s['end'])] for p in replay['people']]
        roles=[x for x in replay['regions'] if overlap(lo,hi,x['start'],x['end'])]
        rows.append(dict(g,id=sid,pair=pair,category=category,context_start=lo,context_end=hi,
                         context_seconds=hi-lo,focus_seconds=b-a,words=words,evidence=evidence,blocks=blocks,
                         transcripts=transcripts,regions=roles,
                         media=f'{pair}/evidence/manual_review/{sid}/context.mp4',
                         poster=f'{pair}/evidence/manual_review/{sid}/poster.jpg',
                         audio_urls={s:f'{pair}/evidence/manual_review/{sid}/{s}.wav' for s in 'AB'},
                         evidence_url=f'{pair}/evidence/manual_review/{sid}/evidence.json',
                         production_result_sha256=sha(prodpath),human_decision='pending',training_admitted=False))
    assert abs(sum(r['focus_seconds'] for r in rows)-sum(t['end']-t['start'] for t in review['timeline'] if t['status']!='keep'))<1e-6
    return rows

def export(row):
    folder=OUT/row['id'];folder.mkdir(exist_ok=True)
    start=row['context_start'];duration=row['context_seconds']
    video=folder/'context.mp4'
    if not video.exists():
        filters=[f"drawtext=text='A - LEFT AUDIO':x=12:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7",
                 f"drawtext=text='B - RIGHT AUDIO':x=732:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7",
                 f"drawtext=text='SOURCE %{{pts\\:flt\\:{start}}} s':x=12:y=h-34:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7",
                 f"drawbox=x=2:y=2:w=iw-4:h=ih-4:color=0xffbf4a:t=5:enable='between(t,{row['start']-start},{row['end']-start})'"]
        subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-ss',str(start),'-i',str(PREVIEW/row['pair']/'source.mp4'),
            '-t',str(duration),'-vf',','.join(filters),'-filter_threads','1','-c:v','libx264','-preset','veryfast',
            '-crf','22','-threads','1','-c:a','aac','-b:a','192k','-movflags','+faststart',str(video)],check=True)
    norm=read(ROOT/'data_corpora/seamless_listen_local_full/normalized_pairs'/(row['pair']+'.json'))
    for si,track in enumerate(norm['tracks']):
        with sf.SoundFile(track['path']) as f:
            f.seek(round(start*f.samplerate));wave=f.read(round(duration*f.samplerate),dtype='float32',always_2d=True)
            assert len(wave)==round(duration*f.samplerate)
            sf.write(folder/f'{"AB"[si]}.wav',wave,f.samplerate,subtype='PCM_24')
    subprocess.run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(video),'-f','null','-'],check=True)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
    assert abs(float(info['format']['duration'])-duration)<.09
    assert next(x for x in info['streams'] if x['codec_type']=='audio')['channels']==2
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(min(duration-.05,max(.05,row['start']-start))),
                    '-i',str(video),'-frames:v','1',str(folder/'poster.jpg')],check=True)
    row['media_check']=dict(decoded=True,actual_seconds=float(info['format']['duration']),audio_channels=2,
                            audio='Existing whole-source loudness normalization; no new denoising or event suppression.')
    (folder/'evidence.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    print(row['id'],row['category'],round(row['start'],3),round(row['end'],3),flush=True)
    return row

if __name__=='__main__':
    rows=[row for i,pair in enumerate(PAIRS,1) for row in prepare(pair,i)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(export,rows))
    value=dict(revision='pair-false-rejection-human-review-20261029',rows=rows,
               count=len(rows),focus_seconds=sum(r['focus_seconds'] for r in rows),
               complete_nonretained_coverage=True,context_is_not_proposed_training_data=True,
               human_review_pending=True,production_modified=False)
    (OUT/'manifest.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE',len(rows),flush=True)
