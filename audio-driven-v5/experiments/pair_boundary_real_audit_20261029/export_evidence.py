"""Export source-backed video/audio evidence; does not change production data."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import soundfile as sf

from audit import ROOT, OUT, save, covered
sys.path.insert(0,str(ROOT/'scripts'))
from build_seamless_listening_preview import uri

REVIEW = OUT/'review'
REVIEW.mkdir(exist_ok=True)
findings = json.loads((OUT/'findings.json').read_text())
one = next(x for x in findings['findings'] if x['pair']=='naturalistic_V02_S3579_I00000175')
two = next(x for x in findings['findings'] if x['kind']=='feedback_tail' and x['domain'][0]>330)


def run(cmd):
    subprocess.run(cmd,check=True,capture_output=True)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))


def source_video(d,start,end,target):
    norm = json.loads((ROOT/'data_corpora/seamless_listen_local_full/normalized_pairs'/(d['pair']+'.json')).read_text())
    cmd = ['ffmpeg','-v','error','-nostdin','-y']
    for s in d['sources']:
        cmd += ['-threads','2','-ss',str(start),'-noautorotate','-i',uri(s['source'])]
    for t in norm['tracks']:
        cmd += ['-ss',str(start),'-i',t['path']]
    filters = [f'[{i}:v]setpts=PTS-STARTPTS,scale=360:480:force_original_aspect_ratio=decrease,'
               f'pad=360:480:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25[v{i}]' for i in range(2)]
    filters += [f"[v0][v1]hstack,drawtext=text='A (left audio)     B (right audio)':x=12:y=12:fontsize=20:"
                f"fontcolor=white:box=1:boxcolor=black@0.7,drawtext=text='source %{{pts\\:flt\\:{start}}} s':"
                f"x=12:y=h-32:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7[v]",
                '[2:a]aformat=sample_rates=48000:channel_layouts=mono,asetpts=PTS-STARTPTS[a0]',
                '[3:a]aformat=sample_rates=48000:channel_layouts=mono,asetpts=PTS-STARTPTS[a1]',
                '[a0][a1]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[a]']
    cmd += ['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[v]','-map','[a]',
            '-t',str(end-start),'-c:v','libx264','-preset','veryfast','-crf','23','-threads','2',
            '-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(target)]
    run(cmd)


def subclip(source,offset,start,end,target):
    run(['ffmpeg','-v','error','-nostdin','-y','-ss',str(start-offset),'-i',str(source),'-t',str(end-start),
         '-map','0:v','-map','0:a','-c:v','libx264','-preset','veryfast','-crf','23','-threads','2',
         '-c:a','aac','-b:a','192k','-movflags','+faststart',str(target)])


def raw_audio(d,start,end,prefix):
    for i,s in enumerate(d['sources']):
        with tarfile.open(s['source']['archive'],'r:') as tar:
            with sf.SoundFile(tar.extractfile(s['id']+'.wav')) as f:
                f.seek(round(start*f.samplerate))
                wave = f.read(round((end-start)*f.samplerate),dtype='float32',always_2d=True)
                sf.write(REVIEW/f'{prefix}_raw_{"AB"[i]}.wav',wave,f.samplerate,subtype='FLOAT')


def export_case(finding,prefix,lo,hi,full=False):
    d=json.loads((OUT/(finding['pair']+'.json')).read_text())
    source=REVIEW/f'{prefix}_full_source.mp4' if full else REVIEW/f'{prefix}_context.mp4'
    source_video(d,0 if full else lo,d['limit'] if full else hi,source)
    context=REVIEW/f'{prefix}_context.mp4'
    if full:
        subclip(source,0,lo,hi,context)
    raw_audio(d,lo,hi,prefix)
    clips=[]
    for version,rows in [('current',finding['current'][-1:]),('alternative',finding['alternative'][-2:] if prefix=='case1' else finding['alternative'])]:
        for i,c in enumerate(rows):
            path=REVIEW/f'{prefix}_{version}_{i+1}.mp4'
            subclip(source,0 if full else lo,c['start'],c['end'],path)
            clips.append(dict(version=version,file=path.name,start=c['start'],end=c['end'],text=c['text'],
                              boundary_review_flags=c['boundary_review_flags']))
    if full:
        folder=REVIEW/'case1_all_replayed_pairs';folder.mkdir(exist_ok=True)
        manifest=[]
        for i,c in enumerate(d['clips']):
            path=folder/f'{i+1:02d}_{c["start"]:.3f}_{c["end"]:.3f}.mp4'
            subclip(source,0,c['start'],c['end'],path)
            manifest.append(dict(file=str(path.relative_to(REVIEW)),speaker_index=c['speaker_index'],
                                 listener_index=c['listener_index'],start=c['start'],end=c['end'],text=c['text']))
        save(folder/'manifest.json',manifest)
    # Confirm the problematic current cut is present in final production.
    current=finding['current'][-1]
    prod=next(c for c in d['production_clips'] if c['speaker_index']==finding['speaker_index'] and
              abs(c['start']-current['start'])<1e-6 and abs(c['end']-current['end'])<1e-6)
    result=dict(pair=d['pair'],source_seconds=d['limit'],context=[lo,hi],clips=clips,
                production_sample_id=prod['sample_id'],production_audio_pool=prod['audio_final']['pool'],
                sources=d['sources'],finding=finding,
                preview_audio='Existing whole-track loudness-normalized independent WAVs; A left / B right; no new noise processing.',
                raw_audio='*_raw_A.wav and *_raw_B.wav are native original WAV slices, float PCM, no gain applied.',
                alternative_status='Segmentation diagnostic only; fresh audio QC and human review not performed.')
    save(REVIEW/f'{prefix}_evidence.json',result)
    print(prefix,'exported',flush=True)
    return result


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(export_case,one,'case1',170,190,True),
              pool.submit(export_case,two,'case2',329,340,False)]
        cases=[f.result() for f in jobs]
    checks=[]
    for path in sorted(REVIEW.rglob('*.mp4')):
        p=probe(path)
        run(['ffmpeg','-v','error','-nostdin','-threads','1','-i',str(path),'-f','null','-'])
        checks.append(dict(file=str(path.relative_to(REVIEW)),seconds=float(p['format']['duration']),
                           video_frames=int(next(s for s in p['streams'] if s['codec_type']=='video')['nb_frames']),
                           audio_channels=next(s for s in p['streams'] if s['codec_type']=='audio')['channels']))
    contract=json.loads((ROOT/'data_corpora/seamless_listen_local_full/contract.json').read_text())
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==contract['code_sha256'][name]
            for name in ['scripts/listen_local_recovery.py','scripts/listen_policy_v8.py','scripts/listen_v8_recovery.py','scripts/listen_turns_v8.py']}
    assert all(hashes.values())
    save(REVIEW/'verification.json',dict(cases=[c['pair'] for c in cases],videos=checks,
        production_hash_matches=hashes,search_pairs=len(findings['inspected']),all_videos_decode=True))
    print('verified',len(checks),'videos',flush=True)
