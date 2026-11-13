"""Build the ResponseNet200 Demo Showcase HTML page for av_event_6000, av_feature_6000, and sftqwen_av_event_6000."""
import concurrent.futures
import csv
import hashlib
import html
import json
import math
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path('/private/liqi_workspace/project_code/audio-driven-v5')
RUN1 = ROOT / 'experiments/p2_testsets_4ckpt_24gpu_20261102'
RUN2 = ROOT / 'experiments/p2_response200_feature_sft_event_20261103'
SEL_DIR = ROOT / 'experiments/responsenet_initial200_20261102'

EXP_OUT = ROOT / 'experiments/p2_responsenet200_demo_20261103'
PREVIEW_OUT = ROOT / 'data_previews/listen_benchmark_review/p2_responsenet200_demo_20261103'
MEDIA_DIR = PREVIEW_OUT / 'media'

SELECTED_CASES = [
    {
        'id': 'TXYN2qw_local_6840d2dfe2b6',
        'featured_arm': 'av_event_6000',
        'category': '高指标标杆',
        'title': 'av_event_6000 高指标标杆（全场视觉综合排名 #1，PSNR 26.20 dB / ΔResp 0.034）',
        'highlight': '在 9.30 秒专注倾听长片段中，av_event_6000 取得 26.20 dB PSNR、0.856 SSIM、0.105 LPIPS 与 89.7% 表情一致率，眉眼运动幅度（0.54 vs GT 0.50）自然贴合且全程零误发声。',
    },
    {
        'id': 'ceq9tRp_local_076f632f4f57',
        'featured_arm': 'av_event_6000',
        'category': '高指标标杆',
        'title': 'av_event_6000 高保真与高表情一致率代表（视觉排名 #2，Emo-Acc 97.3% / PSNR 22.64 dB）',
        'highlight': 'av_event_6000 在 8.48 秒片段上达成 97.3% 表情一致率与 22.64 dB PSNR，头部姿态微动与眼神专注度稳定优于同歩数 Feature 组。',
    },
    {
        'id': 'jD9XJrn_local_81a7e75cc3ca',
        'featured_arm': 'av_event_6000',
        'category': '丰富 Backchannel',
        'title': 'av_event_6000 丰富回应代表：原生 Qwen 主动规划轻微点头＋口头肯定 "Yeah."（命中 GT 有声片段）',
        'highlight': 'GT 为真实有声回应片段（0.61s）。原生 Qwen-3B 主动规划轻微点头与简短口头肯定 "Yeah."，av_event_6000 准确渲染出点头动作与 0.36s 语音回应，ΔResp 仅 0.084。',
    },
    {
        'id': 'NKrgNNz_local_1a6f8d457399',
        'featured_arm': 'av_event_6000',
        'category': '丰富 Backchannel',
        'title': 'av_event_6000 丰富回应代表：微笑（114帧 Happy）＋轻微点头＋语音回应 "Yeah, that makes sense."',
        'highlight': '原生 Qwen-3B 规划轻微点头与口头认同。av_event_6000 同时还原了持续微笑（114 帧 Happy，GT 为 92 帧）、点头与 1.14s 语音，PSNR 21.41 dB、Emo-Acc 78.0%（显著高于 av_feature_6000 的 35.6%）。',
    },
    {
        'id': '3xwdCKZ_local_ec82a5b557c3',
        'featured_arm': 'av_feature_6000',
        'category': '高指标标杆',
        'title': 'av_feature_6000 高指标标杆：PSNR 23.27 dB（领先 Event +3.49 dB）/ Emo-Acc 97.7%',
        'highlight': 'av_feature_6000 在该 7.34 秒片段上表现极佳，PSNR 达 23.27 dB、LPIPS 0.112、Emo-Acc 97.7%、ΔResp 0.055（运动 0.35 vs GT 0.30），展现了蒸馏特征监督对面部结构稳定性的增益。',
    },
    {
        'id': 'ueXQsPb_local_fb68a3614cf2',
        'featured_arm': 'av_feature_6000',
        'category': '高指标标杆',
        'title': 'av_feature_6000 超长片段代表：11.95 秒多类微表情还原（Emo-Acc 92.8% / Emo-Div 4/4 / ΔResp 0.024）',
        'highlight': '在 200 条子集最长的 11.95 秒片段上，av_feature_6000 完整覆盖 GT 的 4 类情绪变化，Emo-Acc 达 92.8%，运动幅度（0.40 vs GT 0.42）高度吻合。',
    },
    {
        'id': 'TXYN2qw_local_46eb2b317caf',
        'featured_arm': 'av_feature_6000',
        'category': '丰富 Backchannel',
        'title': 'av_feature_6000 丰富回应代表：原生 Qwen 规划点头认同＋极准动态匹配（ΔResp 0.006 / Emo-Acc 94.7%）',
        'highlight': '原生 Qwen-3B 规划轻微点头与非语言赞同表情。av_feature_6000 的眉眼运动幅度（0.53）与 GT（0.51）几乎零误差（ΔResp = 0.006），Emo-Acc 达 94.7%。',
    },
    {
        'id': 'iPr0oCN_local_b8e18494a028',
        'featured_arm': 'av_feature_6000',
        'category': '丰富 Backchannel',
        'title': 'av_feature_6000 丰富回应代表：高动态微笑（148帧 Happy）＋点头＋有声回应 "Yeah, I know."（ΔResp 0.099）',
        'highlight': 'GT 为高动态有声片段（GT 运动 1.37）。原生 Qwen 规划点头与 "Yeah, I know."，av_feature_6000 在生成 0.71s 语音与丰富微笑的同时，运动幅度（1.47）最贴合 GT（ΔResp 0.099，显著优于另外两组）。',
    },
    {
        'id': 'Ano4Jkc_local_edc40dffa436',
        'featured_arm': 'sftqwen_av_event_6000',
        'category': '丰富 Backchannel',
        'title': 'sftqwen_av_event_6000 丰富回应标杆：露齿笑→闭口微笑＋闭眼眨眼＋2秒点头＋结尾语音 "Yeah" 严格命中 GT 发声窗',
        'highlight': 'GT 结尾含真实有声回应（Listener ASR: "Well,"）。原生 Qwen 漏判为全静音；SFT Qwen 规划了完整多阶段非语言与语言反应链，sftqwen_av_event_6000 严格命中 GT 发声区间（Strict TP=1, tIoU=0.686），Emo-Acc 达 70.5%。',
    },
    {
        'id': 'UcTatbq_local_ef6296ce2fb4',
        'featured_arm': 'sftqwen_av_event_6000',
        'category': '丰富 Backchannel',
        'title': 'sftqwen_av_event_6000 丰富回应与高指标兼备：6秒点头＋结尾说 "Okay" 响应 Speaker "Thank you"（PSNR 21.35 / ΔResp 0.019）',
        'highlight': 'Speaker 话末说 "Thank you."（GT 有声 1.24s）。SFT Qwen 规划 6 秒处单次点头并在结尾说 "Okay"，sftqwen_av_event_6000 不仅唤醒有声回应（0.20s），且取得 21.35 dB PSNR 与 0.019 ΔResp（运动 0.39 vs GT 0.37）。',
    },
    {
        'id': 't2htkuW_local_8e1257fa6123',
        'featured_arm': 'sftqwen_av_event_6000',
        'category': '高指标标杆',
        'title': 'sftqwen_av_event_6000 微笑与点头高指标代表：Emo-Acc 100.0%（89/89帧 Happy 全中）/ PSNR 24.46 dB',
        'highlight': 'GT 全程微笑倾听（89 帧 Happy）。原生 Qwen 误写中性表情导致前两组 Emo-Acc 仅 ~32%；SFT Qwen 准确规划全程微笑与 2.5 秒处轻微点头，使 sftqwen_av_event_6000 达成 100% Emo-Acc 与 24.46 dB PSNR。',
    },
    {
        'id': 'TXYN2qw_local_4cd1473f44f2',
        'featured_arm': 'sftqwen_av_event_6000',
        'category': '高指标标杆',
        'title': 'sftqwen_av_event_6000 点头高指标代表：3秒处单次点头＋自然眨眼（PSNR 23.92 dB / Emo-Acc 98.2% / ΔResp 0.018）',
        'highlight': '面对 Speaker 称赞，SFT Qwen 规划 3 秒处单次点头与自然眨眼并保持安静倾听，sftqwen_av_event_6000 取得 23.92 dB PSNR、98.2% Emo-Acc，且生成运动（0.67）与 GT（0.67）完全一致（ΔResp 0.018）。',
    },
]


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def waveform(path: Path, duration: float, rate: int = 48000) -> np.ndarray:
    value, sr = sf.read(path, dtype='float32', always_2d=True)
    if value.shape[1] == 1:
        value = np.repeat(value, 2, axis=1)
    if value.shape[1] > 2:
        value = value[:, :2]
    if sr != rate:
        g = math.gcd(sr, rate)
        value = resample_poly(value, rate // g, sr // g, axis=0).astype(np.float32)
    count = round(duration * rate)
    return np.pad(value, ((0, max(0, count - len(value))), (0, 0)))[:count]


def mux_media(video_path: Path, listener_wav: Path, duration: float, dest_mp4: Path, speaker_wav: Path = None):
    dest_mp4.parent.mkdir(parents=True, exist_ok=True)
    if dest_mp4.exists() and dest_mp4.stat().st_size > 1000:
        return
    with tempfile.TemporaryDirectory(prefix='p2_demo_mux_') as tmp:
        tmp_dir = Path(tmp)
        if speaker_wav is not None:
            a = waveform(listener_wav, duration)
            b = waveform(speaker_wav, duration)
            mixed = a + b
            peak = float(np.max(np.abs(mixed)))
            gain = min(1.0, 0.95 / max(peak, 1e-12))
            wav_to_use = tmp_dir / 'mixed.wav'
            sf.write(wav_to_use, mixed * gain, 48000, subtype='FLOAT')
        else:
            a = waveform(listener_wav, duration)
            wav_to_use = tmp_dir / 'listener.wav'
            sf.write(wav_to_use, a, 48000, subtype='FLOAT')

        out_tmp = tmp_dir / 'out.mp4'
        cmd = [
            'ffmpeg', '-y', '-v', 'error', '-nostdin', '-threads', '2',
            '-i', str(video_path),
            '-i', str(wav_to_use),
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            '-t', f'{duration:.4f}',
            '-movflags', '+faststart',
            str(out_tmp),
        ]
        subprocess.run(cmd, check=True)
        shutil.copy2(out_tmp, dest_mp4)


def main():
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    refs = {
        json.loads(line)['id']: json.loads(line)
        for line in (SEL_DIR / 'references.jsonl').read_text().splitlines()
        if line
    }
    sel_info = {
        json.loads(line)['id']: json.loads(line)
        for line in (SEL_DIR / 'selection.jsonl').read_text().splitlines()
        if line
    }

    arms_cfg = [
        {
            'key': 'av_event_6000',
            'display': 'av_event_6000',
            'subtitle': '原生 Qwen-3B 规划 + 原始 H Event (6000步)',
            'gen_dir': RUN1 / 'generated/av_event_6000/responsenet',
            'v_metrics': {
                json.loads(l)['id']: json.loads(l)
                for l in (RUN1 / 'metrics/av_event_6000/responsenet/pairs.jsonl').read_text().splitlines()
                if l
            },
            'a_metrics': {
                json.loads(l)['id']: json.loads(l)
                for l in (RUN1 / 'metrics/av_event_6000/responsenet/audio_pairs.jsonl').read_text().splitlines()
                if l
            },
            'cond_dir': RUN1 / 'conditions/responsenet',
        },
        {
            'key': 'av_feature_6000',
            'display': 'av_feature_6000',
            'subtitle': '原生 Qwen-3B 规划 + 原始 H Feature (6000步)',
            'gen_dir': RUN2 / 'generated/base_feature_6000/responsenet',
            'v_metrics': {
                json.loads(l)['id']: json.loads(l)
                for l in (RUN2 / 'metrics/base_feature_6000/responsenet/pairs.jsonl').read_text().splitlines()
                if l
            },
            'a_metrics': {
                json.loads(l)['id']: json.loads(l)
                for l in (RUN2 / 'metrics/base_feature_6000/responsenet/audio_pairs.jsonl').read_text().splitlines()
                if l
            },
            'cond_dir': RUN1 / 'conditions/responsenet',
        },
        {
            'key': 'sftqwen_av_event_6000',
            'display': 'sftqwen_av_event_6000',
            'subtitle': 'SFT LoRA Qwen-3B 规划 + 原始 H Event (6000步)',
            'gen_dir': RUN2 / 'generated/sft_on_base_event_6000/responsenet',
            'v_metrics': {
                json.loads(l)['id']: json.loads(l)
                for l in (RUN2 / 'metrics/sft_on_base_event_6000/responsenet/pairs.jsonl').read_text().splitlines()
                if l
            },
            'a_metrics': {
                json.loads(l)['id']: json.loads(l)
                for l in (RUN2 / 'metrics/sft_on_base_event_6000/responsenet/audio_pairs.jsonl').read_text().splitlines()
                if l
            },
            'cond_dir': RUN2 / 'sft_event/conditions/responsenet',
        },
    ]

    mux_tasks = []
    records = []

    for idx, item in enumerate(SELECTED_CASES, 1):
        sid = item['id']
        ref = refs[sid]
        s_meta = sel_info[sid]
        dur = float(ref['duration_s'])

        sp_wav = Path(ref['speaker']['audio'])
        gt_mp4 = Path(ref['gt'])
        gt_wav = Path(ref['listener']['audio'])
        sp_asr_data = read_json(Path(ref['speaker']['asr']))
        li_asr_data = read_json(Path(ref['listener']['asr']))

        base_plan = read_json(RUN1 / 'conditions/responsenet' / sid / 'qwen/plan.json')
        sft_plan = read_json(RUN2 / 'sft_event/conditions/responsenet' / sid / 'qwen/plan.json')

        # Copy speaker wav
        sp_wav_rel = f'media/{sid}_speaker.wav'
        if not (PREVIEW_OUT / sp_wav_rel).exists():
            shutil.copy2(sp_wav, PREVIEW_OUT / sp_wav_rel)

        # GT mixed & listener-only
        gt_mix_rel = f'media/{sid}_gt_mixed.mp4'
        gt_only_rel = f'media/{sid}_gt_listener.mp4'
        mux_tasks.append((gt_mp4, gt_wav, dur, PREVIEW_OUT / gt_mix_rel, sp_wav))
        mux_tasks.append((gt_mp4, gt_wav, dur, PREVIEW_OUT / gt_only_rel, None))

        arm_entries = {}
        for arm in arms_cfg:
            akey = arm['key']
            gfolder = arm['gen_dir'] / sid
            gen_mp4 = gfolder / 'generated.mp4'
            gen_wav = gfolder / 'listener.wav'
            assert gen_mp4.is_file() and gen_wav.is_file(), f'Missing generated files for {akey} {sid}'

            mix_rel = f'media/{sid}_{akey}_mixed.mp4'
            only_rel = f'media/{sid}_{akey}_listener.mp4'
            mux_tasks.append((gen_mp4, gen_wav, dur, PREVIEW_OUT / mix_rel, sp_wav))
            mux_tasks.append((gen_mp4, gen_wav, dur, PREVIEW_OUT / only_rel, None))

            vm = arm['v_metrics'][sid]
            am = arm['a_metrics'][sid]
            phase = vm['response']['phases'][0]
            ac_comp = am['audio']['comparison']['acoustic']
            sp_comp = am['audio']['comparison']['speech']
            arm_entries[akey] = {
                'key': akey,
                'display': arm['display'],
                'subtitle': arm['subtitle'],
                'mixed_video': mix_rel,
                'listener_video': only_rel,
                'source_video': str(gen_mp4),
                'source_audio': str(gen_wav),
                'psnr': vm['metrics']['psnr'],
                'ssim': vm['metrics']['ssim'],
                'lpips': vm['metrics']['lpips'],
                'emo_acc': vm['metrics']['emo_acc'] * 100.0,
                'emo_div_gen': vm['metrics']['emo_div_generated'],
                'emo_div_gt': vm['metrics']['emo_div_gt'],
                'delta_resp': vm['metrics']['delta_resp'],
                'motion_gen': phase['motion_generated'],
                'motion_gt': phase['motion_gt'],
                'happy_gen': vm['emotion']['histogram_generated'][3],
                'happy_gt': vm['emotion']['histogram_gt'][3],
                'gen_acoustic_s': ac_comp['generated_active_s'],
                'gt_acoustic_s': ac_comp['gt_active_s'],
                'gen_speech_s': sp_comp['generated_active_s'],
                'gt_speech_s': sp_comp['gt_active_s'],
                'acoustic_tp': ac_comp['strict']['tp'],
            }

        records.append({
            'index': idx,
            'id': sid,
            'sample_id': ref['sample_id'],
            'pair_index': ref['pair_index'],
            'source_start': ref['source_start'],
            'source_end': ref['source_end'],
            'duration_s': dur,
            'featured_arm': item['featured_arm'],
            'category': item['category'],
            'title': item['title'],
            'highlight': item['highlight'],
            'global_visual_rank': s_meta['global_visual_rank'],
            'gt_nonzero': s_meta['gt_nonzero'],
            'gt_acoustic': s_meta['gt_acoustic'],
            'gt_speech': s_meta['gt_speech'],
            'speaker_wav': sp_wav_rel,
            'speaker_asr': sp_asr_data.get('text', ''),
            'listener_asr': li_asr_data.get('text', ''),
            'gt_mixed_video': gt_mix_rel,
            'gt_listener_video': gt_only_rel,
            'base_qwen_description': base_plan['description'],
            'base_gemma_prompt': base_plan['prompt'],
            'sft_qwen_description': sft_plan['description'],
            'sft_gemma_prompt': sft_plan['prompt'],
            'qwen_system_prompt': base_plan['system'],
            'qwen_user_prompt': base_plan['user'],
            'arms': arm_entries,
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(mux_media, *t) for t in mux_tasks]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    # Save structured manifest JSON
    manifest = {
        'schema': 'p2-responsenet200-demo-v1',
        'url': 'http://localhost:8876/listen_benchmark_review/p2_responsenet200_demo_20261103/',
        'total_cases': len(records),
        'models': ['av_event_6000', 'av_feature_6000', 'sftqwen_av_event_6000'],
        'cases': records,
    }
    (PREVIEW_OUT / 'data.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (EXP_OUT / 'data.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')

    # Build HTML page
    esc = html.escape
    cards_html = []
    for r in records:
        arm_cards = []
        for akey in ('av_event_6000', 'av_feature_6000', 'sftqwen_av_event_6000'):
            a = r['arms'][akey]
            is_feat = akey == r['featured_arm']
            badge = '<span class="featured-badge">本条主推代表</span>' if is_feat else ''
            plan_used = '原生 Qwen-3B 指导' if akey != 'sftqwen_av_event_6000' else 'SFT Qwen-3B 指导'
            arm_cards.append(f'''
            <div class="model-card {'featured' if is_feat else ''}" data-model="{esc(akey)}">
              <div class="model-header">
                <div>
                  <h4>{esc(a['display'])}</h4>
                  <span class="sub">{esc(a['subtitle'])}</span>
                </div>
                {badge}
              </div>
              <video controls playsinline preload="metadata"
                src="{esc(a['mixed_video'])}"
                data-mixed="{esc(a['mixed_video'])}"
                data-listener="{esc(a['listener_video'])}"
                data-label="{esc(a['display'])} · 样本#{r['index']:02d}"></video>
              <div class="model-metrics">
                <span><b>PSNR:</b> {a['psnr']:.2f} dB</span>
                <span><b>SSIM:</b> {a['ssim']:.3f}</span>
                <span><b>LPIPS:</b> {a['lpips']:.3f}</span>
                <span><b>Emo-Acc:</b> {a['emo_acc']:.1f}%</span>
                <span><b>ΔResp:</b> {a['delta_resp']:.3f}</span>
                <span><b>运动(生成/GT):</b> {a['motion_gen']:.2f} / {a['motion_gt']:.2f}</span>
                <span><b>Happy帧(生成/GT):</b> {a['happy_gen']} / {a['happy_gt']}</span>
                <span><b>生成声学/GT:</b> {a['gen_acoustic_s']:.2f}s / {a['gt_acoustic_s']:.2f}s</span>
              </div>
              <div class="model-links">
                <span class="mode-indicator">当前：Speaker＋Listener 叠加音轨</span>
                <a href="{esc(a['mixed_video'])}" download>下载叠加视频</a>
                <a href="{esc(a['listener_video'])}" download>下载仅Listener视频</a>
              </div>
            </div>''')

        gt_card = f'''
        <div class="model-card gt-card">
          <div class="model-header">
            <div>
              <h4>GT 真实 Listener（参考对照）</h4>
              <span class="sub">源视频 {esc(r['sample_id'])} ({r['source_start']:.2f}s – {r['source_end']:.2f}s)</span>
            </div>
            <span class="tag-gt">{'GT有声回应' if r['gt_acoustic'] else 'GT安静倾听'}</span>
          </div>
          <video controls playsinline preload="metadata"
            src="{esc(r['gt_mixed_video'])}"
            data-mixed="{esc(r['gt_mixed_video'])}"
            data-listener="{esc(r['gt_listener_video'])}"
            data-label="GT真实Listener · 样本#{r['index']:02d}"></video>
          <div class="model-metrics">
            <span><b>时长:</b> {r['duration_s']:.2f} s</span>
            <span><b>GT声学活动:</b> {r['arms']['av_event_6000']['gt_acoustic_s']:.2f} s</span>
            <span><b>GT语音时长:</b> {r['arms']['av_event_6000']['gt_speech_s']:.2f} s</span>
            <span><b>GT运动幅度:</b> {r['arms']['av_event_6000']['motion_gt']:.2f}</span>
            <span><b>GT Happy帧:</b> {r['arms']['av_event_6000']['happy_gt']} 帧</span>
            <span><b>GT Emo-Div:</b> {r['arms']['av_event_6000']['emo_div_gt']:.0f} 类</span>
          </div>
          <div class="model-links">
            <span class="mode-indicator">当前：Speaker＋Listener 叠加音轨</span>
            <a href="{esc(r['gt_mixed_video'])}" download>下载GT叠加视频</a>
            <a href="{esc(r['gt_listener_video'])}" download>下载GT仅Listener</a>
          </div>
        </div>'''

        li_asr_html = f'<p class="listener-asr"><b>GT Listener 官方/本地 ASR：</b>{esc(r["listener_asr"])}</p>' if r['listener_asr'] else ''

        cards_html.append(f'''
        <article class="case-card" data-arm="{esc(r['featured_arm'])}" data-cat="{esc(r['category'])}" data-id="{esc(r['id'])}" id="case-{r['index']:02d}">
          <div class="case-top">
            <div class="case-title-row">
              <span class="case-num">#{r['index']:02d}</span>
              <div>
                <h2>{esc(r['title'])}</h2>
                <div class="meta-pills">
                  <span class="pill pill-arm">主推模型：{esc(r['featured_arm'])}</span>
                  <span class="pill pill-cat">类型：{esc(r['category'])}</span>
                  <span class="pill">样本 ID：<code>{esc(r['id'])}</code></span>
                  <span class="pill">时长：{r['duration_s']:.2f}s</span>
                </div>
              </div>
            </div>
            <p class="highlight-box">{esc(r['highlight'])}</p>
          </div>

          <div class="info-grid">
            <div class="info-box speaker-box">
              <h3>1. Speaker 输入音频与 ASR 转写</h3>
              <audio controls preload="none" src="{esc(r['speaker_wav'])}" data-label="Speaker原始音频 · 样本#{r['index']:02d}"></audio>
              <div class="prompt-text asr-text">{esc(r['speaker_asr'])}</div>
              {li_asr_html}
            </div>
            <div class="info-box prompt-box">
              <h3>2. 原生 Qwen2.5-Omni-3B 生成指导描述（用于 av_event_6000 / av_feature_6000）</h3>
              <div class="prompt-text">{esc(r['base_qwen_description'])}</div>
              <details>
                <summary>查看完整 Gemma 文本指导 Prompt（原生 Qwen）</summary>
                <pre>{esc(r['base_gemma_prompt'])}</pre>
              </details>
            </div>
            <div class="info-box prompt-box sft-box">
              <h3>3. SFT LoRA Qwen2.5-Omni-3B 生成指导描述（用于 sftqwen_av_event_6000）</h3>
              <div class="prompt-text">{esc(r['sft_qwen_description'])}</div>
              <details>
                <summary>查看完整 Gemma 文本指导 Prompt（SFT Qwen）与 Qwen 规划指令</summary>
                <pre>【Gemma 正向指导全文】\n{esc(r['sft_gemma_prompt'])}\n\n【Qwen 规划 System Prompt】\n{esc(r['qwen_system_prompt'])}\n\n【Qwen 规划 User Prompt】\n{esc(r['qwen_user_prompt'])}</pre>
              </details>
            </div>
          </div>

          <div class="video-grid">
            {gt_card}
            {''.join(arm_cards)}
          </div>
        </article>''')

    html_doc = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ResponseNet 200条子集 · Demo汇报精选展示（av_event_6000 / av_feature_6000 / sftqwen_av_event_6000）</title>
<style>
:root {{
  --bg: #f3f6fa;
  --surface: #ffffff;
  --ink: #162436;
  --muted: #5a6d82;
  --line: #dbe3ec;
  --blue: #155e96;
  --teal: #0b7268;
  --amber: #b86b11;
  --purple: #6b46c1;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.6 system-ui, -apple-system, "Microsoft YaHei", sans-serif;
}}
header, main, footer {{
  max-width: 1720px;
  margin: auto;
  padding: 20px 28px;
}}
header {{
  padding-top: 28px;
  padding-bottom: 12px;
}}
.eyebrow {{
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--blue);
}}
h1 {{
  margin: 6px 0 8px;
  font-size: 28px;
}}
.summary-table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 10px;
  overflow: hidden;
  font-size: 13px;
  margin: 14px 0;
}}
.summary-table th, .summary-table td {{
  padding: 9px 12px;
  border-bottom: 1px solid #eef2f6;
  text-align: right;
}}
.summary-table th:first-child, .summary-table td:first-child,
.summary-table th:nth-child(2), .summary-table td:nth-child(2) {{
  text-align: left;
}}
.summary-table th {{
  background: #f0f4f8;
  color: var(--muted);
  font-weight: 600;
}}
.toolbar {{
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(255,255,255,0.95);
  backdrop-filter: blur(10px);
  border-block: 1px solid var(--line);
  padding: 12px 28px;
  display: flex;
  gap: 14px;
  align-items: center;
  flex-wrap: wrap;
}}
.btn-group {{
  display: inline-flex;
  background: #eaf0f6;
  padding: 3px;
  border-radius: 8px;
  gap: 2px;
}}
button {{
  font: inherit;
  font-size: 13px;
  border: 0;
  background: transparent;
  color: var(--muted);
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
}}
button.active {{
  background: var(--blue);
  color: #fff;
}}
#audio-toggle button.active {{
  background: var(--teal);
}}
input[type="search"] {{
  font: inherit;
  font-size: 13px;
  padding: 6px 12px;
  border: 1px solid var(--line);
  border-radius: 6px;
  width: 240px;
}}
.status-bar {{
  font-size: 12px;
  color: var(--muted);
  margin-left: auto;
}}
.download-btn {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: var(--blue);
  color: #fff;
  text-decoration: none;
  font-size: 13px;
  font-weight: 600;
  padding: 6px 13px;
  border-radius: 6px;
  white-space: nowrap;
}}
.download-btn.secondary {{
  background: #e6eff7;
  color: var(--blue);
}}
.download-btn:hover {{
  opacity: 0.92;
}}
.case-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 22px;
  margin: 22px 0;
  scroll-margin-top: 80px;
}}
.case-title-row {{
  display: flex;
  gap: 14px;
  align-items: flex-start;
}}
.case-num {{
  font-size: 26px;
  font-weight: 800;
  color: var(--blue);
  min-width: 48px;
}}
.case-card h2 {{
  margin: 0 0 6px;
  font-size: 19px;
}}
.meta-pills {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12px;
}}
.pill {{
  background: #eff4f9;
  color: var(--ink);
  padding: 2px 9px;
  border-radius: 5px;
}}
.pill-arm {{
  background: #e3f0fc;
  color: var(--blue);
  font-weight: 700;
}}
.pill-cat {{
  background: #e6f6f4;
  color: var(--teal);
  font-weight: 700;
}}
.highlight-box {{
  background: #f8fafd;
  border-left: 4px solid var(--blue);
  padding: 9px 14px;
  margin: 12px 0 16px;
  font-size: 14px;
}}
.info-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 16px;
}}
.info-box {{
  background: #f9fbfd;
  border: 1px solid #e4ecf3;
  border-radius: 8px;
  padding: 12px 14px;
}}
.info-box.sft-box {{
  background: #f4fbf8;
  border-color: #cce8df;
}}
.info-box h3 {{
  margin: 0 0 8px;
  font-size: 14px;
  color: var(--ink);
}}
.info-box audio {{
  width: 100%;
  height: 34px;
  margin-bottom: 6px;
}}
.prompt-text {{
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 13.5px;
  line-height: 1.55;
  white-space: pre-wrap;
}}
.asr-text {{
  font-weight: 500;
}}
.listener-asr {{
  font-size: 12.5px;
  color: var(--teal);
  margin: 6px 0 0;
}}
details {{
  margin-top: 8px;
  font-size: 12px;
}}
summary {{
  cursor: pointer;
  color: var(--blue);
  font-weight: 600;
}}
pre {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #eef3f8;
  padding: 10px;
  border-radius: 6px;
  font-size: 12px;
  max-height: 260px;
  overflow: auto;
}}
.video-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}}
.model-card {{
  border: 1px solid var(--line);
  border-radius: 9px;
  background: #fff;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}
.model-card.featured {{
  border: 2px solid var(--blue);
  background: #fafdff;
}}
.model-card.gt-card {{
  border-top: 3px solid #4a5568;
}}
.model-Header {{
  padding: 10px 12px;
}}
.model-header {{
  padding: 10px 12px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 8px;
  border-bottom: 1px solid #eef2f6;
}}
.model-header h4 {{
  margin: 0;
  font-size: 15px;
}}
.model-header .sub {{
  font-size: 11.5px;
  color: var(--muted);
  display: block;
}}
.featured-badge {{
  background: var(--blue);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 4px;
  white-space: nowrap;
}}
.tag-gt {{
  background: #edf2f7;
  color: #2d3748;
  font-size: 11px;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 4px;
  white-space: nowrap;
}}
.model-card video {{
  width: 100%;
  aspect-ratio: 1;
  background: #000;
  display: block;
}}
.model-metrics {{
  padding: 10px 12px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4px 10px;
  font-size: 12px;
  border-bottom: 1px solid #eef2f6;
}}
.model-links {{
  padding: 8px 12px;
  font-size: 11.5px;
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  color: var(--muted);
}}
.model-links a {{
  color: var(--blue);
  text-decoration: none;
  margin-left: auto;
}}
.model-links a + a {{
  margin-left: 0;
}}
@media (max-width: 1280px) {{
  .video-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .info-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <div class="eyebrow">AUDIO-DRIVEN V5 · DEMO 汇报精选展示页（2026-11-03）</div>
  <h1>ResponseNet 200条子集：三组核心模型（av_event_6000 / av_feature_6000 / sftqwen_av_event_6000）精选案例</h1>
  <p>本页针对 <b>av_event_6000</b>、<b>av_feature_6000</b>、<b>sftqwen_av_event_6000</b> 各精选 4 条代表样本（2 条【高指标标杆】＋ 2 条【丰富 Backchannel 回应代表】，共 12 条）。默认播放 <b>Speaker Audio + Listener 推理音视频等增益对齐叠加版</b>，支持一键切换“仅 Listener 音视频”，并完整展示 Speaker Audio ASR 与 Qwen 推理生成指导 Prompt。</p>
  <table class="summary-table">
    <thead>
      <tr>
        <th>模型 / 权重 (200条子集同口径)</th>
        <th>前端规划器与训练 H</th>
        <th>PSNR ↑</th>
        <th>SSIM ↑</th>
        <th>LPIPS ↓</th>
        <th>FID ↓</th>
        <th>FVD ↓</th>
        <th>Emo-Acc ↑</th>
        <th>Emo-Div (GT 3.14)</th>
        <th>ΔResp ↓</th>
        <th>有声漏检↓ (34条)</th>
        <th>静音误发声↓ (166条)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>av_event_6000</b></td>
        <td>原生 Qwen-3B + 原始 H Event</td>
        <td>20.038</td>
        <td><b>0.7223</b></td>
        <td><b>0.2197</b></td>
        <td><b>10.28</b></td>
        <td>144.09</td>
        <td><b>68.59%</b></td>
        <td>2.83</td>
        <td><b>0.155</b></td>
        <td>32 / 34 (94.1%)</td>
        <td><b>2 / 166 (1.2%)</b></td>
      </tr>
      <tr>
        <td><b>av_feature_6000</b></td>
        <td>原生 Qwen-3B + 原始 H Feature</td>
        <td><b>20.041</b></td>
        <td>0.7213</td>
        <td>0.2199</td>
        <td>10.64</td>
        <td>155.26</td>
        <td>67.74%</td>
        <td>2.82</td>
        <td>0.171</td>
        <td>32 / 34 (94.1%)</td>
        <td><b>2 / 166 (1.2%)</b></td>
      </tr>
      <tr>
        <td><b>sftqwen_av_event_6000</b></td>
        <td>SFT LoRA Qwen-3B + 原始 H Event</td>
        <td>19.296</td>
        <td>0.7091</td>
        <td>0.2490</td>
        <td>10.72</td>
        <td><b>143.66</b></td>
        <td>64.22%</td>
        <td><b>3.39</b></td>
        <td>0.208</td>
        <td><b>25 / 34 (73.5%)</b></td>
        <td>31 / 166 (18.7%)</td>
      </tr>
    </tbody>
  </table>
</header>

<nav class="toolbar" aria-label="筛选与音轨切换">
  <div class="btn-group" id="arm-filter">
    <button class="active" data-arm="all">全部 12 条精选</button>
    <button data-arm="av_event_6000">av_event_6000 主推 (4条)</button>
    <button data-arm="av_feature_6000">av_feature_6000 主推 (4条)</button>
    <button data-arm="sftqwen_av_event_6000">sftqwen_av_event_6000 主推 (4条)</button>
  </div>
  <div class="btn-group" id="cat-filter">
    <button class="active" data-cat="all">全部类型</button>
    <button data-cat="高指标标杆">高指标标杆 (6条)</button>
    <button data-cat="丰富 Backchannel">丰富 Backchannel (6条)</button>
  </div>
  <div class="btn-group" id="audio-toggle">
    <button class="active" data-mode="mixed">播放：Speaker＋Listener 叠加音视频</button>
    <button data-mode="listener">播放：仅 Listener 推理音视频</button>
  </div>
  <input type="search" id="search-box" placeholder="搜索样本 ID 或关键词...">
  <a class="download-btn" href="p2_responsenet200_demo_20261103.zip" download>下载完整离线包 (.zip)</a>
  <a class="download-btn secondary" href="cases_summary.csv" download>下载案例与Prompt表 (.csv)</a>
  <span class="status-bar" id="play-status">默认播放叠加音轨（0秒对齐、等增益相加防削波）；点击任一视频自动暂停其余播放器</span>
</nav>

<main id="cases-container">
  {''.join(cards_html)}
</main>

<footer>
  ResponseNet 200条固定子集 Demo 汇报页 · 生成配置：seed 137，30步 Euler，video CFG 3 / audio CFG 7，512×512 · 原始视频与音频路径见 data.json。
</footer>

<script>
'use strict';
let currentArm = 'all';
let currentCat = 'all';
let currentMode = 'mixed';

function pauseOthers(activeMedia) {{
  document.querySelectorAll('video, audio').forEach(m => {{
    if (m !== activeMedia) m.pause();
  }});
}}

document.querySelectorAll('video, audio').forEach(m => {{
  m.addEventListener('play', () => {{
    pauseOthers(m);
    const label = m.dataset.label || '正在播放媒体';
    document.getElementById('play-status').textContent = '正在有声播放：' + label;
  }});
}});

function applyFilter() {{
  const q = document.getElementById('search-box').value.trim().toLowerCase();
  document.querySelectorAll('.case-card').forEach(card => {{
    const matchArm = currentArm === 'all' || card.dataset.arm === currentArm;
    const matchCat = currentCat === 'all' || card.dataset.cat === currentCat;
    const matchQ = !q || card.textContent.toLowerCase().includes(q);
    card.hidden = !(matchArm && matchCat && matchQ);
  }});
}}

document.querySelectorAll('#arm-filter button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('#arm-filter button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentArm = btn.dataset.arm;
    applyFilter();
  }});
}});

document.querySelectorAll('#cat-filter button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('#cat-filter button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentCat = btn.dataset.cat;
    applyFilter();
  }});
}});

document.querySelectorAll('#audio-toggle button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('#audio-toggle button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
    document.querySelectorAll('video[data-mixed]').forEach(v => {{
      const t = v.currentTime || 0;
      const wasPlaying = !v.paused;
      v.pause();
      v.src = currentMode === 'mixed' ? v.dataset.mixed : v.dataset.listener;
      if (t > 0 || wasPlaying) {{
        v.addEventListener('loadedmetadata', function once() {{
          v.removeEventListener('loadedmetadata', once);
          v.currentTime = Math.min(t, v.duration || t);
          if (wasPlaying) v.play();
        }});
        v.load();
      }}
    }});
    document.querySelectorAll('.mode-indicator').forEach(el => {{
      el.textContent = currentMode === 'mixed' ? '当前：Speaker＋Listener 叠加音轨' : '当前：仅 Listener 独立音轨';
    }});
  }});
}});

document.getElementById('search-box').addEventListener('input', applyFilter);
</script>
</body>
</html>'''

    (PREVIEW_OUT / 'index.html').write_text(html_doc)
    (EXP_OUT / 'index.html').write_text(html_doc)

    # Export CSV summary for convenient PPT copy-pasting
    csv_rows = []
    for r in records:
        ev = r['arms']['av_event_6000']
        ft = r['arms']['av_feature_6000']
        sft = r['arms']['sftqwen_av_event_6000']
        csv_rows.append({
            '序号': r['index'],
            '样本ID': r['id'],
            '主推模型': r['featured_arm'],
            '类型': r['category'],
            '标题': r['title'],
            '亮点总结': r['highlight'],
            '时长(s)': r['duration_s'],
            'Speaker_ASR': r['speaker_asr'],
            'GT_Listener_ASR': r['listener_asr'],
            '原生Qwen生成指导描述': r['base_qwen_description'],
            '原生Qwen完整Gemma_Prompt': r['base_gemma_prompt'],
            'SFT_Qwen生成指导描述': r['sft_qwen_description'],
            'SFT_Qwen完整Gemma_Prompt': r['sft_gemma_prompt'],
            'GT叠加视频': r['gt_mixed_video'],
            'av_event_6000叠加视频': ev['mixed_video'],
            'av_event_6000_PSNR': round(ev['psnr'], 3),
            'av_event_6000_EmoAcc(%)': round(ev['emo_acc'], 2),
            'av_event_6000_DeltaResp': round(ev['delta_resp'], 4),
            'av_feature_6000叠加视频': ft['mixed_video'],
            'av_feature_6000_PSNR': round(ft['psnr'], 3),
            'av_feature_6000_EmoAcc(%)': round(ft['emo_acc'], 2),
            'av_feature_6000_DeltaResp': round(ft['delta_resp'], 4),
            'sftqwen_av_event_6000叠加视频': sft['mixed_video'],
            'sftqwen_av_event_6000_PSNR': round(sft['psnr'], 3),
            'sftqwen_av_event_6000_EmoAcc(%)': round(sft['emo_acc'], 2),
            'sftqwen_av_event_6000_DeltaResp': round(sft['delta_resp'], 4),
        })
    for folder in (PREVIEW_OUT, EXP_OUT):
        with (folder / 'cases_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Package portable offline zip bundle
    zip_name = 'p2_responsenet200_demo_20261103.zip'
    exp_zip = EXP_OUT / zip_name
    prev_zip = PREVIEW_OUT / zip_name
    if exp_zip.exists():
        exp_zip.unlink()
    if prev_zip.exists():
        prev_zip.unlink()

    prefix = 'p2_responsenet200_demo_20261103'
    with zipfile.ZipFile(exp_zip, 'w', compression=zipfile.ZIP_STORED) as zf:
        for fname in ('index.html', 'data.json', 'cases_summary.csv'):
            zf.write(PREVIEW_OUT / fname, arcname=f'{prefix}/{fname}')
        for mfile in sorted(MEDIA_DIR.glob('*')):
            if mfile.is_file():
                zf.write(mfile, arcname=f'{prefix}/media/{mfile.name}')
    shutil.copy2(exp_zip, prev_zip)

    print(json.dumps({
        'status': 'ok',
        'cases': len(records),
        'preview_index': str(PREVIEW_OUT / 'index.html'),
        'zip_path': str(exp_zip),
        'zip_size_mb': round(exp_zip.stat().st_size / (1024 * 1024), 2),
        'media_files': len(list(MEDIA_DIR.glob('*'))),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
