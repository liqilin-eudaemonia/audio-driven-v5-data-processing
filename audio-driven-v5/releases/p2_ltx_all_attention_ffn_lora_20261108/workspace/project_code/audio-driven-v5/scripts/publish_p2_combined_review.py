"""Add combined Whisper checkpoints to the existing review with exact scored PCM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,rows,save,sha
from evaluation.listener_metrics.media import read_audio
from analyze_p2_seamless200 import paired,load


def publish(work):
    work=work.resolve()
    page=ROOT/'data_previews/listen_benchmark_review/p2_seamless200_20261105'
    out=ROOT/'experiments/p2_seamless200_analysis_20261105'
    d=read(page/'data.json');p=read(work/'protocol.json');results=read(work/'RESULTS.json')
    assert (work/'COMPLETE.json').exists()
    _,_,selection,baseline,_,_=load()
    names={a['name'] for a in p['arms']}
    d['models']=[x for x in d['models'] if x['key'] not in names]
    d['summary']=[x for x in d['summary'] if x['arm'] not in names]
    for arm in p['arms']:
        name=arm['name'];label='AV＋Event＋Feature＋Whisper · '+str(arm['step'])
        d['models'].append(dict(key=name,label=label,step=arm['step'],planner=arm['planner'],checkpoint=arm['original_checkpoint']))
        folder=work/'metrics'/name/'seamless'
        video={x['id']:x for x in rows(folder/'pairs.jsonl')}
        audio={x['id']:x for x in rows(folder/'audio_pairs.jsonl')}
        assert set(video)==set(audio)==set(selection)
        summary=results[name+'/seamless'];activity=summary['audio']['activity']['acoustic']
        d['summary'].append(dict(arm=name,label=label,step=arm['step'],**summary['metrics'],
            generated_active_clips=sum(bool(x['audio']['generated']['acoustic']) for x in audio.values()),
            gt_active_clips=activity['active_reference_pairs'],gt_silent_clips=activity['silent_reference_pairs'],
            completely_silent_on_active_gt=activity['missed_entire_response'],missed_response_rate=activity['missed_response_rate'],
            false_sound_on_silent_gt=activity['false_activity_on_silent_clip'],false_sound_clip_rate=activity['false_sound_clip_rate'],
            strict_audio_f1=activity['strict']['f1'],tolerant_audio_f1=activity['tolerant']['f1'],
            gt_active_s=activity['gt_active_s'],generated_active_s=activity['generated_active_s'],
            emotion_pairs=summary['coverage']['emotion']['video_identities'],
            motion_phases=summary['coverage']['delta_resp']['valid_phases']))
        d['paired']['CombinedWhisper'+str(arm['step'])+'_vs_AV6000']=dict(left='av_only_6000',right=name,
            comparison='6000 matched steps;8000 progress reference only',
            metrics={m:paired(baseline['av_only_6000'],video,m,selection) for m in ['psnr','ssim','lpips','emo_acc','delta_resp']})
        for case in d['cases']:
            sid=case['id'];dest=page/'media'/sid;src=work/'generated'/name/'seamless'/sid
            for ext,source in [('mp4','generated.mp4'),('wav','listener.wav')]:
                target=dest/(name+'.'+ext)
                if target.is_symlink():assert target.resolve()==(src/source).resolve()
                elif target.exists():raise FileExistsError(target)
                else:target.symlink_to(src/source)
            record=audio[sid]['audio'];wave=read_audio(src/'listener.wav',case['duration'])
            wave=wave[:round(record['audit']['evaluated_duration_s']*16000)]
            b=wave.astype('<f4').tobytes();checksum=hashlib.sha256(b).hexdigest()
            assert checksum==record['audit']['generated_waveform_sha256']
            assert record['audit']['gt_waveform_sha256']==case['gt_pcm']['sha256']
            raw=dest/(name+'.f32');raw.write_bytes(b)
            pcm=dict(url=str(raw.relative_to(page)),rate=16000,samples=len(wave),channels=1,
                     sha256=checksum,exact_zero=bool((wave==0).all()),peak=float(abs(wave).max()))
            vr=video[sid]
            case['models'][name]=dict(video=str((dest/(name+'.mp4')).relative_to(page)),
                wav=str((dest/(name+'.wav')).relative_to(page)),pcm=pcm,metrics=vr['metrics'],errors=vr['errors'],
                emotion=vr.get('emotion'),response=vr.get('response'),face_detection=vr.get('face_detection'),
                audio=dict(gt=record['gt'],generated=record['generated'],comparison=record['comparison'],gt_group=record['group']))
    assert len(d['models'])==14 and len(d['summary'])==14
    for dest in [page/'data.json',out/'data.json']:save(dest,d)
    save(out/'summary_with_whisper.json',d['summary']);save(page/'summary_with_whisper.json',d['summary'])
    template=(ROOT/'scripts/p2_seamless200_review.html').read_text()
    template=template.replace('新十组对照','十二组对照').replace('十组模型实际生成了什么','十二组模型实际生成了什么').replace('（10组）','（12组）').replace('从六个消融问题看这十组结果','从六个消融问题看全部结果')
    template=template.replace('Whisper补测完成后将接入本页；','Whisper6000／8000已完成并加入下方表格和模型选择；')
    template=template.replace('这十组没有Whisper分支，当前不能判断。已安排Whisper6000／8000补测，等待现有28卡释放后自动运行；将在同一200条上对照',
        'Whisper6000／8000已补测。以同一200条对照')
    template=template.replace('查看补测状态与协议','查看Whisper补测协议').replace('各组均已含speaker音频经Qwen的间接条件，','各组均已含speaker音频经Qwen的间接条件，')
    # Results link survives browser reload, source template remains ten-arm fallback.
    template=template.replace('<details id="overview">','<p><a href="whisper_supplement/RESULTS.md">Whisper补测完整指标</a>；6000步为主要对照，8000步为进展参考。</p><details id="overview">')
    for dest in [page/'index.html',out/'index.html']:
        tmp=dest.with_suffix('.html.tmp');tmp.write_text(template);tmp.replace(dest)
    save(work/'HTML_PUBLISHED.json',dict(systems=14,cases=16,added_outputs=32,
        page='http://localhost:8876/listen_benchmark_review/p2_seamless200_20261105/',
        matched_evaluation_pcm=True,prior_tenway_results_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    publish(p.parse_args().work)
