"""Export exact, complete repaired pairs, with no added playback context."""
import collections
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent
OUT=EXP/'review/fixed_pairs'
RESULTS=EXP/'fixed_pairs'
MEDIA_DIR='fixed_pairs'
ID_PREFIX='F'
PREVIEW=ROOT/'data_previews/seamless_listen_full_source_recovery'
sys.path[:0]=[str(ROOT/'scripts'),str(EXP)]
from rerun_fixed_pairs import PAIRS,PROD,read,write
from p3_listen.common import sha
import listen_local_recovery as base

def view(pair,number):
    d=read(RESULTS/pair/'result.json');initial=d['initial_ids']
    all_rows=sorted(d['clips']+d['superseded_attempts'],key=lambda r:(r['start'],r['end']))
    roots=sorted([r for r in all_rows if r['sample_id'] in initial],key=lambda r:r['start'])
    children=[r for r in all_rows if r['sample_id'] not in initial]
    ids={r['sample_id']:f'{ID_PREFIX}{number}-{i+1:02d}' for i,r in enumerate(roots)}
    ids.update({r['sample_id']:f'{ID_PREFIX}{number}-C{i+1:02d}' for i,r in enumerate(children)})
    rows=[]
    for r in roots+children:
        rid=ids[r['sample_id']];route=r['audio_final'];q=route['quality']
        active=set(q['hard_reasons']+q['review_reasons']);windows=[]
        for w in q['evidence_windows']:
            if w['reason'] in active:
                windows.append(dict(start=r['start']+w['start'],end=r['start']+w['end'],reason=w['reason'],role=w.get('role')))
        for w in r['qc4']['reference']['windows']:
            if w.get('suspicious') and any('reference' in x or 'leakage' in x for x in active):
                windows.append(dict(start=r['start']+w['start'],end=r['start']+w['end'],reason='reference_match',role='listener'))
        models=[]
        for role,track in r['qc3'].items():
            for w in track['events']+track['background']:
                if not w['decision']['reasons']:continue
                p=w['panns'];f=w['features']
                models.append(dict(start=r['start']+w['start'],end=r['start']+w['end'],role=role,
                    participant='AB'[r[role+'_index']],noise_label=p['noise_label'],noise_score=p['noise_score'],
                    human_label=p['human_label'],human_score=p['human_score'],rms_dbfs=f['rms_dbfs'],reasons=w['decision']['reasons']))
        refs=[dict(w,start=r['start']+w['start'],end=r['start']+w['end']) for w in r['qc4']['reference']['windows'] if w.get('suspicious')]
        parent=r.get('recovery_parent_id');descendants=[]
        for child in d['clips']:
            if child['sample_id']!=r['sample_id'] and r['start']<=child['start'] and child['end']<=r['end']:
                descendants.append(dict(id=ids[child['sample_id']],start=child['start'],end=child['end'],pool=child['audio_final']['pool']))
        prefix=f'{pair}/evidence/{MEDIA_DIR}/{rid}'
        v=dict(id=rid,pair=pair,sample_id=r['sample_id'],start=r['start'],end=r['end'],duration=r['duration_s'],
            speaker_index=r['speaker_index'],text=r['text'],listener_events=r['listener_events'],
            pool=route['pool'],reasons=route['reasons'],advisory_reasons=route['advisory_reasons'],
            quality=q,model_windows=models,reference_windows=refs,evidence_windows=windows,
            boundary_flags=r['boundary_review_flags'],boundary_repair=r.get('boundary_repair'),
            boundary_evidence={k:r.get(k) for k in ('semantic_start','semantic_end','speech_start','speech_end','tail_s','start_padding_s')},
            quality_provenance=r['quality_provenance'],initial=r['sample_id'] in initial,
            parent_id=ids.get(parent),children=descendants,media=prefix+'/pair.mp4',poster=prefix+'/poster.jpg',
            audio_urls={x:prefix+f'/{x}.wav' for x in 'AB'},evidence_url=prefix+'/evidence.json',
            exact_pair_playback=True,added_context_seconds=0,production_modified=False,
            semantic_review=any(x in r['boundary_review_flags'] for x in ('before_long_gap_unconfirmed','after_long_gap_unconfirmed','inferred_clause_start','inferred_clause_end','transcript_end_unconfirmed','source_sentence_start_unconfirmed','recovered_partial_sentence')))
        folder=OUT/rid;folder.mkdir(parents=True,exist_ok=True)
        write(folder/'evidence.json',dict(view=v,full_qc=r))
        rows.append(v)
    prod=read(PROD/'results'/(pair+'.json'))
    old_kept=base.seconds([(r['start'],r['end']) for r in prod['clips'] if r['audio_final']['pool']=='keep_candidate'])
    return dict(pair=pair,rows=rows,source_seconds=d['source_seconds'],initial_count=len(roots),
        initial_pools=dict(collections.Counter(r['audio_final']['pool'] for r in roots)),
        terminal_count=len(d['clips']),terminal_pools=d['pool_counts'],old_kept_seconds=old_kept,
        kept_seconds=d['kept_seconds'],counts=d['counts'],initial_ids=[ids[r['sample_id']] for r in roots],
        terminal_ids=[ids[r['sample_id']] for r in d['clips']],revision=d['revision'],
        pause_threshold_seconds=read(RESULTS/'contract.json').get('pause_threshold_seconds',3),
        semantic_review_count=sum(r['initial'] and r['semantic_review'] for r in rows))

def export(r):
    folder=OUT/r['id'];video=folder/'pair.mp4'
    if not video.exists():
        filters=["drawtext=text='A - LEFT AUDIO':x=12:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7",
                 "drawtext=text='B - RIGHT AUDIO':x=732:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7",
                 f"drawtext=text='SOURCE %{{pts\\:flt\\:{r['start']}}} s':x=12:y=40:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7"]
        subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-ss',str(r['start']),'-i',str(PREVIEW/r['pair']/'source.mp4'),
            '-t',str(r['duration']),'-vf',','.join(filters),'-filter_threads','1','-c:v','libx264','-preset','veryfast',
            '-crf','22','-threads','1','-c:a','aac','-b:a','192k','-movflags','+faststart',str(video)],check=True)
    norm=read(PROD/'normalized_pairs'/(r['pair']+'.json'))
    for i,track in enumerate(norm['tracks']):
        path=folder/f'{"AB"[i]}.wav'
        if not path.exists():
            with sf.SoundFile(track['path']) as f:
                f.seek(round(r['start']*f.samplerate));wave=f.read(round(r['duration']*f.samplerate),dtype='float32')
                assert len(wave)==round(r['duration']*f.samplerate)
                sf.write(path,wave,f.samplerate,subtype='PCM_24')
    subprocess.run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(video),'-f','null','-'],check=True)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
    assert abs(float(info['format']['duration'])-r['duration'])<.09
    assert next(s for s in info['streams'] if s['codec_type']=='audio')['channels']==2
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(video),'-frames:v','1',str(folder/'poster.jpg')],check=True)
    r['media_check']=dict(decoded=True,actual_seconds=float(info['format']['duration']),audio_channels=2)
    print(r['id'],r['start'],r['end'],r['pool'],flush=True)

def main():
    pairs=[view(p,i+1) for i,p in enumerate(PAIRS)]
    rows=[r for d in pairs for r in d['rows']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(export,rows))
    contract=read(RESULTS/'contract.json')
    assert all(sha(PROD/'results'/(p+'.json'))==h for p,h in contract['production_results'].items())
    write(OUT/'manifest.json',dict(pairs=pairs,total_initial=sum(d['initial_count'] for d in pairs),
        total_withheld_initial=sum(r['initial'] and r['pool']!='keep_candidate' for r in rows),
        exact_pair_playback=True,production_modified=False,screening_rules_changed=False))
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
