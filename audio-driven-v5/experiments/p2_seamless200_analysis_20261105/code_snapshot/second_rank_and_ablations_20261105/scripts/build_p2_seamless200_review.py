"""Build16 disclosed examples with GT, all10 systems and exact evaluation PCM."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from analyze_p2_seamless200 import load,RUN,OUT,LABELS
from p2_testsets_eval.decoded_gt import DecodedGTCache
from evaluation.seamless import reader
from evaluation.listener_metrics.media import audio_pair,read_audio

PREVIEW=ROOT/'data_previews/listen_benchmark_review/p2_seamless200_20261105'
REVIEW_POINTS={
 1:'组合头姿较接近GT，但两组都没有完整呈现GT的口部变化。LPIPS改善是否也意味着回应动作更完整？',
 2:'组合组中段出现更明显的转头和横向位移，GT相对正面。重点看姿态是否偏离，而不只看清晰度。',
 3:'比较脸部细节和姿态稳定性；确认相似度优势是否也伴随正确的动作变化。',
 4:'ΔResp衡量运动幅度差，不验证点头方向或次数。检查幅度接近是否也更自然。',
 5:'GT保持微笑，SFT规划器组恢复了微笑，原生组多为闭口或撅嘴；也请留意SFT末尾表情是否过强。',
 6:'GT头姿相对稳定，SFT组中段出现很明显的低头。重点复核点头是否过深、过多。',
 7:'同一SFT规划器下，检查SFT H组表情一致率提高，是否也更自然，而非简单变得更爱笑。',
 8:'同一SFT规划器下，检查两种H训练条件的动作幅度差异，是否存在过度运动。',
 9:'SFT组有活动匹配，但末尾头部位移较大、面部靠近画面边缘。请同时判断声音内容、时机及视觉动作。',
 10:'GT净化波形严格全零，即使嘴部有动作，也不能据此认定应有声音。重点试听SFT额外声音是否合理。',
 11:'GT有声而十组生成活动均为空；结合speaker语境听GT是否为明显回应，确认共同漏声问题。',
 12:'GT为严格全零音轨。重点比较安静倾听时的自然微动，不把无声本身当作失败。',
 13:'同一Event模型两步数对照，检查画面差异是否也对应更合理的回应。',
 14:'幅度误差变小也可能来自少动；检查继续训练后是否还保留GT的动作。',
 15:'固定种子随机静音样本，未按任何模型指标择优；用于核对目的性选例之外的表现。',
 16:'固定种子随机非静音样本，未按任何模型指标择优；结合speaker语境检查发声与动作。'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):return str(path.relative_to(PREVIEW))


def pcm(path,wave,rate=16000):
    value=np.ascontiguousarray(wave,dtype='<f4');path.write_bytes(value.tobytes())
    return dict(url=relative(path),rate=rate,samples=len(value),channels=1,sha256=sha(path),
                exact_zero=bool(np.all(value==0)),peak=float(np.max(np.abs(value))))


def symlink(source,dest):
    source=Path(source).resolve();dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_symlink():
        if dest.resolve()!=source:raise ValueError('Media link target changed')
    elif dest.exists():raise FileExistsError(dest)
    else:dest.symlink_to(source)


def main():
    PREVIEW.mkdir(parents=True,exist_ok=True)
    p,summary,selection,video,audio,plans=load()
    cases=json.loads((OUT/'showcase_selection.json').read_text())
    refs={x['id']:x for x in map(json.loads,(Path(p['datasets']['seamless']['prepared'])/'references.jsonl').read_text().splitlines())}
    cache=DecodedGTCache(RUN/'decoded_gt_cache')
    def build(case):
        sid=case['id'];ref=refs[sid];meta=selection[sid];duration=ref['duration_s']
        dest=PREVIEW/'media'/sid;dest.mkdir(parents=True,exist_ok=True)
        frame_audit=RUN/'base_standard/conditions/seamless'/sid/'input_audit.json'
        conditioning_audit=json.loads(frame_audit.read_text())
        _,resolved=reader.resolve(ref['gt_reference'])
        assert resolved['crop_provenance']['actual_crop']==conditioning_audit['crop_provenance']['actual_crop']
        source_image=RUN/'base_standard/conditions/seamless'/sid/'first_frame.png'
        shutil.copy2(source_image,dest/'first_frame.png')
        assert sha(source_image)==sha(dest/'first_frame.png')
        reader.backend().check_file(conditioning_audit['speaker_audio_path'],conditioning_audit['speaker_audio_sha256'])
        with sf.SoundFile(conditioning_audit['speaker_audio_path']) as source:
            sr=source.samplerate;source.seek(conditioning_audit['speaker_first_sample'])
            speaker=source.read(conditioning_audit['speaker_last_sample']-conditioning_audit['speaker_first_sample'],dtype='float32')
        g=math.gcd(sr,16000)
        speaker=resample_poly(speaker,16000//g,sr//g).astype(np.float32)
        count=round(duration*16000);speaker=np.pad(speaker,(0,max(0,count-len(speaker))))[:count]
        speaker_pcm=pcm(dest/'speaker.f32',speaker);sf.write(dest/'speaker.wav',speaker,16000,subtype='FLOAT')
        first=p['arms'][0]['name'];pair=video[first][sid]['input'];gt,_,audio_audit=audio_pair(pair,duration)
        expected_gt=audio[first][sid]['audio']['audit']['gt_waveform_sha256']
        assert hashlib.sha256(gt.tobytes()).hexdigest()==expected_gt
        gt_pcm=pcm(dest/'gt.f32',gt);sf.write(dest/'gt.wav',gt,16000,subtype='FLOAT')
        frames,fps,gt_audit=cache.get(ref['gt_reference'])
        assert gt_audit['rgb_sha256']==video[first][sid]['media']['gt_sha256']
        gt_video=dest/'gt.mp4'
        if not gt_video.exists():
            command=['ffmpeg','-nostdin','-v','error','-threads','1','-f','rawvideo','-pixel_format','rgb24',
                '-video_size','512x512','-framerate',str(fps),'-i','pipe:0','-i',str(dest/'gt.wav'),
                '-map','0:v:0','-map','1:a:0','-c:v','libx264','-threads','2','-preset','fast','-crf','18',
                '-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-t',str(duration),'-movflags','+faststart',str(gt_video)]
            result=subprocess.run(command,input=frames.tobytes(),capture_output=True)
            if result.returncode:raise RuntimeError(result.stderr.decode())
        del frames
        entries={}
        for arm in p['arms']:
            key=arm['name'];folder=RUN/'generated'/key/'seamless'/sid
            symlink(folder/'generated.mp4',dest/(key+'.mp4'));symlink(folder/'listener.wav',dest/(key+'.wav'))
            record=audio[key][sid]['audio'];wave=read_audio(folder/'listener.wav',duration)
            wave=wave[:round(record['audit']['evaluated_duration_s']*16000)]
            assert hashlib.sha256(wave.tobytes()).hexdigest()==record['audit']['generated_waveform_sha256']
            assert record['audit']['gt_waveform_sha256']==expected_gt
            a=pcm(dest/(key+'.f32'),wave)
            vr=video[key][sid]
            entries[key]=dict(video=relative(dest/(key+'.mp4')),wav=relative(dest/(key+'.wav')),pcm=a,
                metrics=vr['metrics'],errors=vr['errors'],emotion=vr.get('emotion'),response=vr.get('response'),
                face_detection=vr.get('face_detection'),audio=dict(gt=record['gt'],generated=record['generated'],
                    comparison=record['comparison'],gt_group=record['group']))
        result=dict(**case,review_focus=REVIEW_POINTS[case['index']],
            frame_inspection_only=case['index'] in [1,2,5,6,9,10],
            duration=duration,face_p10=meta['face_p10_px'],source_group=meta['group_id'],
            source_start=ref['source_start'],source_end=ref['source_end'],gt_caption=meta['caption'],
            audio_selection_group=meta['audio_group'],gt_video=relative(gt_video),gt_wav=relative(dest/'gt.wav'),
            gt_pcm=gt_pcm,speaker_pcm=speaker_pcm,speaker_wav=relative(dest/'speaker.wav'),
            first_frame=relative(dest/'first_frame.png'),gt_audio_intervals=audio[first][sid]['audio']['gt'],
            base_plan=plans['base_standard'][sid],sft_plan=plans['sft_event'][sid],models=entries,
            media_audit=dict(gt_rgb_sha256=gt_audit['rgb_sha256'],gt_audio_sha256=expected_gt,
                final_crop=conditioning_audit['crop_provenance']['actual_crop'],
                input_audit=str(frame_audit),first_frame_is_original_generation_cache=True,
                first_frame_cache_sha256=sha(source_image),
                preview_gt_encoding_only=True,PCM_same_as_evaluation=True,source_only_generation=True))
        print('prepared',case['index'],sid,flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(build,cases))
    document=dict(schema='p2-seamless200-tenway-review-v1',run=str(RUN),samples=200,
        models=[dict(key=a['name'],label=LABELS[a['name']],step=a['step'],planner=a['planner'],checkpoint=a['checkpoint']) for a in p['arms']],
        summary=json.loads((OUT/'summary.json').read_text()),paired=json.loads((OUT/'paired_bootstrap.json').read_text()),
        planning_statistics=json.loads((OUT/'planning_statistics.json').read_text()),cases=records,
        demo_sampling='14 purposeful metric/audio contrast examples plus2 seeded random; not a representative performance estimate',
        audio_policy='Exact evaluation mono16k float32; selected pane only; combined mode applies fixed0.5 gain to each source')
    for path in [PREVIEW/'data.json',OUT/'data.json']:
        path.write_text(json.dumps(document,ensure_ascii=False,separators=(',',':'))+'\n')
    for name in ['ANALYSIS.md','ABLATIONS.md','summary.csv','summary.json','paired_bootstrap.json','showcase_selection.json','planning_statistics.json']:
        shutil.copy2(OUT/name,PREVIEW/name)
    template=ROOT/'scripts/p2_seamless200_review.html'
    for path in [PREVIEW/'index.html',OUT/'index.html']:path.write_bytes(template.read_bytes())
    oldnav='<nav id="seamless200-dataset-nav" style="padding:12px 28px;background:#e5eff9"><a href="../p2_seamless200_20261105/" style="color:#155e96;font-weight:700">新 Seamless200 · 10组指标与16条典型生成对照 →</a>　<span>当前：历史 ResponseNet200 精选</span></nav>'
    for path in [ROOT/'data_previews/listen_benchmark_review/p2_responsenet200_demo_20261103/index.html',ROOT/'experiments/p2_responsenet200_demo_20261103/index.html']:
        text=path.read_text()
        if 'id="seamless200-dataset-nav"' not in text:path.write_text(text.replace('<body>','<body>\n'+oldnav,1))
    print('Ready http://localhost:8876/listen_benchmark_review/p2_seamless200_20261105/',flush=True)


if __name__=='__main__':main()
