#!/usr/bin/env python3
"""Review page for reaction-cue captions, in the same layout as the vocal pilot."""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from build_listen_visual_prompt_versions_review import compact_mask
from listen_benchmark.j_omni import read_audio
from listen_benchmark.official_audio_gate import make_gate

BASE = ROOT / 'experiments/listen_glare_gate30_20261103'
RUN = BASE / 'omni_reaction_vocal_v3'
PAGE = ROOT / 'data_previews/listen_benchmark_review/native_simple/reaction_gate30_omni'
VOCAL_PREVIEW = ROOT / 'data_previews/listen_benchmark_review/native_simple/natural_caption_nod_vocal24_20261103'
LABEL = {'head shaking': '摇头', 'brow lowering': '皱眉', 'brows raised': '抬眉'}
CHANNELS = (
    ('nodding', '点头'), ('head_shaking', '摇头'), ('vocal', '发声'), ('smiling', '微笑'),
    ('laughing', '视觉大笑'), ('frowning', '皱眉'), ('surprised', '惊讶'),
)
SENT = {'head shaking': 'head_shaking', 'brow lowering': 'frowning', 'brows raised': 'surprised'}
SPOTLIGHTS = [
    {
        'category': '发声',
        'short': '3497',
        'sample_id': 'naturalistic_V02_S3497_I00000239_local_f8ab7d3d750a',
        'cue': '0.04–0.40 秒 Yeah，7.00–7.67 秒 Right, right, right.',
        'before': '只写 mouth opening and closing，没有写出听者说了什么词。',
        'after': '写出开头的 "Yeah" 和接近结尾的 "Right, right, right"，两段点头仍保留。',
    },
    {
        'category': '发声＋皱眉＋点头',
        'short': '1081',
        'sample_id': 'naturalistic_V03_S1081_I00000116_local_fc4ff19c69d6',
        'cue': '皱眉 1.36–1.92 秒，点头 0.24–0.80 / 2.40–2.88 秒，发声 2.92–3.40 / 8.14–8.68 秒 Yeah',
        'before': '没有写皱眉，也没有写点头。',
        'after': '写出约两秒处压眉、约三秒处说 "Yeah" 并点头、接近结尾再说一次 "Yeah"。',
    },
    {
        'category': '皱眉',
        'short': '0255',
        'sample_id': 'naturalistic_V00_S0255_I00000499_local_5213f12bc115',
        'cue': '皱眉 0.00–4.80 秒，三次 Yeah，6.77–7.15 秒 That\'s cool，点头 7.24–7.64 秒',
        'before': '写了微笑、三次 Yeah、That\'s cool 和点头，但没有写前半段的皱眉。',
        'after': '开头补上 brow lowers slightly，三次 "Yeah"、"That\'s cool" 和结尾点头都保留。',
    },
    {
        'category': '摇头（与点头重叠）',
        'short': '1038',
        'sample_id': 'improvised_V00_S1038_I00001135_local_0def3163ab36',
        'cue': '摇头 0.96–2.64 秒，点头 1.44–2.04 / 2.56–3.68 / 5.48–6.04 秒',
        'before': '只写三次点头，没有摇头。',
        'after': '先写 head shakes side to side，再写后面的点头，摇头没有再被点头吃掉。',
    },
    {
        'category': '摇头＋惊讶＋点头',
        'short': '5327',
        'sample_id': 'naturalistic_V02_S5327_I00000056_local_08dcf9890c09',
        'cue': '摇头 0.72–1.76 秒，抬眉 4.28–6.80 / 9.56–10.71 秒，点头 4.24–4.68 / 9.44–9.92 秒',
        'before': '只写微笑、摸脖子和结尾点头，没有写开头的摇头和两段抬眉。',
        'after': '开头写出 side-to-side head shake，四秒附近写出 raised brows 与点头，结尾再写第二次 brows raise。',
    },
    {
        'category': '摇头＋惊讶（保住原有点头）',
        'short': '3185',
        'sample_id': 'naturalistic_V02_S3185_I00000131_local_6ac61ca57701',
        'cue': '摇头 1.04–2.08 秒，点头 2.48–3.20 秒，抬眉 2.64–4.40 秒',
        'before': '只写了一次点头，没有写摇头和抬眉。',
        'after': '按顺序写出一次左右摇头、随后的抬眉，以及约 2.5 秒处的点头。',
    },
    {
        'category': '点头＋摇头',
        'short': '3070',
        'sample_id': 'naturalistic_V02_S3070_I00000495_local_9c949d34d868',
        'cue': '摇头 0.04–3.68 秒，点头 5.80–6.72 / 7.80–8.56 秒',
        'before': '只写视线变化和摸太阳穴，摇头和点头都没有写。',
        'after': '开头补上 side-to-side head shake，中后段补上 nodding his head down and up once。',
    },
]


def mentioned(text, observation):
    if observation == 'head shaking':
        if re.search(r'\bno (?:distinct |visible )?(?:[\w, ]{0,40})?shakes\b', text or '', re.I):
            return False
        return bool(re.search(r'\bshak\w*\b', text or '', re.I))
    if observation == 'brow lowering':
        return bool(re.search(
            r'\b(?:frown\w*|furrow\w*|lower\w* (?:her |his |their )?(?:eye)?brows?|(?:eye)?brows? (?:slightly |are |is )?(?:lower\w*|down|drawn|knit\w*))\b',
            text or '', re.I))
    return bool(re.search(
        r'\b(?:rais\w* (?:her |his |their )?(?:eye)?brows?|(?:eye)?brows? (?:are |is |remain |stay |stays |raise\w*)|raised brows)\b',
        text or '', re.I))


def intervals(sample_id, nod_events, vocal_cues, reaction_cues):
    detected = json.loads((BASE / 'reactions/results' / f'{sample_id}.json').read_text())
    sent = {(SENT[cue['observation']], round(cue['start'], 3), round(cue['end'], 3)) for cue in reaction_cues}
    sent.update(('nodding', round(event['start'], 3), round(event['end'], 3)) for event in nod_events)
    channels = {}
    for name, label in CHANNELS:
        if name == 'vocal':
            channels[name] = [
                {
                    'start': cue['start'], 'end': cue['end'], 'sent': True,
                    'label': f"发声 {cue.get('official_text') or cue.get('asr_text') or cue.get('observation')}",
                    'channel': name,
                }
                for cue in vocal_cues
            ]
            continue
        events = []
        for event in (detected.get('channels') or {}).get(name, {}).get('events') or []:
            key = (name, round(event['start'], 3), round(event['end'], 3))
            events.append({'start': event['start'], 'end': event['end'], 'sent': key in sent, 'label': label, 'channel': name})
        channels[name] = events
    return channels


def write_audio(media, caption):
    import numpy as np
    import soundfile as sf
    recipe = caption['audio_recipe']
    source = recipe['source']
    wave, rate, identity = read_audio(recipe)
    gate, _ = make_gate(recipe['official_listener'], len(wave), rate)
    extra = np.ones_like(gate)
    media.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(source['path']) as handle:
        handle.seek(source['first_sample'])
        raw = handle.read(source['last_sample'] - source['first_sample'], dtype='float32', always_2d=True)
    sf.write(media / 'listener.wav', raw[:, 0], rate, subtype='PCM_16')
    src = caption['source']
    prefix = src['pair'].split('_', 1)[0]
    speaker_source = Path(source['path']).parent / f"{prefix}_{src['speaker_id']}.wav"
    with sf.SoundFile(speaker_source) as handle:
        handle.seek(source['first_sample'])
        speaker = handle.read(source['last_sample'] - source['first_sample'], dtype='float32', always_2d=True)
    sf.write(media / 'speaker.wav', speaker[:, 0], rate, subtype='PCM_16')
    return {
        'original': f'media/{caption["sample_id"]}/listener.wav',
        'speaker': f'media/{caption["sample_id"]}/speaker.wav',
        'sample_rate': rate,
        'source_samples': len(wave),
        'gate': compact_mask(gate),
        'extra': compact_mask(extra),
        'processed_full_silence': bool(identity['full_silence']),
        'processed_pcm_sha256': identity['processed_waveform_sha256'],
    }


def link(dest, source):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(source.resolve())


def main():
    PAGE.mkdir(parents=True, exist_ok=True)
    for asset in ('audio.js', 'review.woff'):
        if (VOCAL_PREVIEW / asset).exists():
            shutil.copy2(VOCAL_PREVIEW / asset, PAGE / asset)
    manifest = {row['sample_id']: row for row in json.loads((RUN / 'manifest.json').read_text())}
    samples = []
    for path in sorted((RUN / 'results').glob('*.json')):
        row = json.loads(path.read_text())
        item = manifest[row['sample_id']]
        caption = json.loads(Path(item['caption_file']).read_text())
        media = PAGE / 'media' / row['sample_id']
        link(media / 'video.mp4', BASE / 'media' / row['sample_id'] / 'video.mp4')
        audio_info = write_audio(media, caption)
        silent = audio_info['processed_full_silence']
        vocal_cues = row.get('vocal_cues') if 'vocal_cues' in row else item.get('vocal_cues') or []
        cues = []
        for cue in row['reaction_cues']:
            cues.append({
                'start': cue['start'], 'end': cue['end'], 'observation': cue['observation'],
                'label': LABEL[cue['observation']], 'written': mentioned(row['caption'], cue['observation']),
            })
        for cue in vocal_cues:
            word = cue.get('official_text') or cue.get('asr_text') or cue.get('observation')
            token = (word or '').strip('.,').lower()
            cues.append({
                'start': cue['start'], 'end': cue['end'], 'observation': 'vocal',
                'label': f'发声({word})', 'written': bool(token and token.split()[0] in row['caption'].lower()),
            })
        samples.append({
            'index': row['index'], 'id': row['sample_id'], 'short': row['short'],
            'duration': item['duration_s'],
            'video': f'media/{row["sample_id"]}/video.mp4',
            'audio': audio_info, 'silent': silent,
            'channels': intervals(row['sample_id'], row['nod_events'], vocal_cues, row['reaction_cues']),
            'cues': cues, 'prior': row['prior_caption'], 'caption': row['caption'],
            'prompt': row['prompt'], 'finish_reason': row['finish_reason'],
            'completion_tokens': row['completion_tokens'], 'seconds': row['seconds'],
        })
    samples.sort(key=lambda row: row['index'])
    cues = [cue for sample in samples for cue in sample['cues'] if cue['observation'] != 'vocal']
    vocal_items = [cue for sample in samples for cue in sample['cues'] if cue['observation'] == 'vocal']
    by_label = {}
    for cue in cues:
        by_label.setdefault(cue['label'], []).append(cue)
    missed = [f"第{sample['index']}条 {sample['short']} {cue['label']} {cue['start']:.2f}–{cue['end']:.2f} 秒"
              for sample in samples for cue in sample['cues'] if not cue['written']]
    analysis = [
        f"送进新描述的反应区间共 {len(cues)} 个（摇头 7、皱眉 5、抬眉 4），发声区间 {len(vocal_items)} 个。",
        '摇头 {0}/{1}、抬眉 {2}/{3}、皱眉 {4}/{5}、发声 {6}/{7} 已写进新描述。未写出的是：{8}。'.format(
            sum(c['written'] for c in by_label.get('摇头', [])), len(by_label.get('摇头', [])),
            sum(c['written'] for c in by_label.get('抬眉', [])), len(by_label.get('抬眉', [])),
            sum(c['written'] for c in by_label.get('皱眉', [])), len(by_label.get('皱眉', [])),
            sum(c['written'] for c in vocal_items), len(vocal_items),
            '；'.join(missed) or '无'),
        '同条同时有摇头和点头时（1038、3185、5327、1361、1771、3070），摇头写成左右摇，点头也保留下来。',
        '这条没有的反应类别不再写进指令，0509、0801 不再出现“没有摇头”的否定句，1081 也不再凭空加抬眉。',
        '发声把重叠同词区间合并并去掉无词纯 VAD 后，3497 补出 Yeah 与 Right, right, right，0377 改成 uh-huh，0044 补齐两次 Yeah 与两次 Right，1081 保住前后两次 Yeah。',
    ]
    PAGE.mkdir(parents=True, exist_ok=True)
    payload = {
        'summary': {
            'samples': len(samples), 'cues': len(cues), 'written': sum(c['written'] for c in cues),
            'vocal_cues': len(vocal_items), 'vocal_written': sum(c['written'] for c in vocal_items),
        },
        'spotlights': SPOTLIGHTS,
        'analysis': analysis, 'channels': dict(CHANNELS), 'samples': samples,
    }
    (PAGE / 'review.json').write_text(json.dumps(payload, ensure_ascii=False))
    (PAGE / 'index.html').write_text(HTML)
    print(PAGE)
    print(json.dumps(payload['summary'], ensure_ascii=False))


HTML = r'''<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>30 条反应线索打标对照</title>
<style>
@font-face{font-family:Review;src:url(review.woff)}
*{box-sizing:border-box}
body{margin:0;background:#f3f5f8;color:#17243b;font:15px/1.7 system-ui,Review,sans-serif}
main{max-width:1620px;margin:auto;padding:22px}
h1{font-size:23px;margin:4px 0}h2{font-size:16px;margin:0 0 10px}p{margin:8px 0}
.muted{color:#617087;font-size:13px}
.panel{background:#fff;border:1px solid #dde3eb;border-radius:12px;padding:18px;margin-top:14px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;border:1px solid #cad3e0;background:white;border-radius:7px;padding:6px 11px;color:inherit;cursor:pointer}
button:hover{background:#f0f4fa}
.layout{display:grid;grid-template-columns:minmax(330px,35%) minmax(0,1fr);gap:18px;align-items:start}
.left{position:sticky;top:14px}
video{width:100%;max-height:460px;background:#111;border-radius:9px}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.old{border:1px solid #cbd5e1;background:#f8fafc;border-radius:10px;padding:14px}
.new{border:2px solid #1b5e20;background:#f6fcf7;border-radius:10px;padding:14px}
.caption-text{font-size:15px;line-height:1.78;white-space:pre-wrap}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace;background:#f7f9fc;padding:12px;border-radius:8px;max-height:520px;overflow:auto}
.lane{display:grid;grid-template-columns:72px 1fr;gap:8px;align-items:center;margin:6px 0}
.track{position:relative;height:22px;background:#eef2f6;border-radius:4px;cursor:pointer}
.seg{position:absolute;top:2px;bottom:2px;border-radius:3px}
.drop{background:transparent!important;border:1.5px solid #8b97a6}
.nodding{background:#2f7d4a}.head_shaking{background:#1d6fbf}.vocal{background:#0284c7}.smiling{background:#c9841a}
.laughing{background:#b42318}.frowning{background:#6b4c9a}.surprised{background:#0f766e}
.hit{outline:2px solid #111}
.analysis li{margin:6px 0}
.spots{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:10px;margin-top:8px}
.spot{border:1px solid #d5dee9;background:#f8fafc;border-radius:9px;padding:10px 12px;cursor:pointer}
.spot:hover{background:#eef4fb;border-color:#93c5fd}
.spot b{color:#0f172a}
@media(max-width:1100px){.layout,.compare{grid-template-columns:1fr}.left{position:static}}
</style>
<main>
  <h1>Qwen3-Omni 发声＋反应＋点头打标对照</h1>
  <p class="muted">合并了定版 v4 发声指令、原有点头区间，以及筛过的摇头、皱眉、抬眉。只有当条真有该类反应时才写进指令，不输入旧描述。</p>
  <div class="panel">
    <h2>五类原先没写、加入检测后写进描述的样本（点击直接跳到该条）</h2>
    <div id="spotlights" class="spots"></div>
  </div>
  <div class="panel">
    <h2>分析结果</h2>
    <ul id="analysis" class="analysis"></ul>
  </div>
  <div class="panel">
    <div id="summary" style="font-weight:600;margin-bottom:8px">加载中…</div>
    <div class="row">
      <button id="prev">← 上一条</button>
      <select id="sample"></select>
      <button id="next">下一条 →</button>
      <select id="filter">
        <option value="all">全部 30 条</option>
        <option value="cued">送进了摇头、皱眉或抬眉</option>
        <option value="miss">送了但新描述没写</option>
      </select>
      <a href="review.json" download>下载完整对照 JSON</a>
    </div>
  </div>
  <div class="layout">
    <section class="panel left">
      <h2 id="sampleId"></h2>
      <video id="video" controls muted playsinline></video>
      <audio id="listenerAudio" preload="auto"></audio>
      <audio id="speakerAudio" preload="auto"></audio>
      <div class="row" style="margin-top:10px">
        <button id="soundPlay">有声播放／暂停</button>
        <select id="audioMode">
          <option value="processed">净化 listener（模型实际输入）</option>
          <option value="original">原始 listener（未静音参考）</option>
          <option value="dialogue">双人对话混音（带 speaker 参考）</option>
          <option value="silent">静音仅看画面</option>
        </select>
      </div>
      <div class="row" style="margin-top:8px">
        <label>音量 <input id="audioVolume" type="range" min="0" max="1" step=".05" value="1"></label>
      </div>
      <p id="audioStatus" class="muted"></p>
    </section>
    <section>
      <div class="panel">
        <h2>原始检测区间</h2>
        <p class="muted">实心是送进描述的区间，空心是检测到但没有送进去。点击从该时间播放。</p>
        <div id="lanes"></div>
        <div id="events" class="row"></div>
      </div>
      <div class="panel">
        <div class="compare">
          <div class="old"><h2 style="color:#475569">打标前</h2><div id="prior" class="caption-text"></div></div>
          <div class="new"><h2 style="color:#1b5e20">打标后</h2><div id="caption" class="caption-text"></div><p id="meta" class="muted"></p></div>
        </div>
      </div>
      <details class="panel" open>
        <summary>本条完整输入 prompt</summary>
        <pre id="prompt"></pre>
      </details>
    </section>
  </div>
</main>
<script src="audio.js"></script>
<script>
let data,visible=[],current,stopAt=null;
const $=id=>document.getElementById(id);
const video=$('video');
function jump(start,end){
  stopAt=end!==undefined&&end!==null?Math.min(current.duration,end+.25):null;
  video.pause();
  video.currentTime=Math.max(0,start-(end!==undefined?.15:0));
  video.play();
}
video.addEventListener('timeupdate',()=>{
  if(stopAt!==null&&video.currentTime>=stopAt){video.pause();stopAt=null;}
  document.querySelectorAll('.seg').forEach(seg=>{
    const [a,b]=seg.title.match(/[\d.]+/g).map(Number);
    seg.classList.toggle('hit',video.currentTime>=a&&video.currentTime<b);
  });
});
async function show(id){
  current=data.samples.find(s=>s.id===id);
  if(!current)return;
  const s=current;
  stopAt=null;video.pause();video.muted=true;
  video.src=s.video;
  $('sample').value=id;
  $('sampleId').textContent=`#${s.index} · ${s.short} · ${s.duration.toFixed(2)} 秒${s.silent?' · 净化音频为静音':''}`;
  const lanes=$('lanes');lanes.replaceChildren();
  const events=$('events');events.replaceChildren();
  Object.entries(data.channels).forEach(([k,name])=>{
    const lane=document.createElement('div');lane.className='lane';
    lane.innerHTML=`<span>${name}</span><div class="track"></div>`;
    const track=lane.querySelector('.track');
    track.onclick=e=>{const r=track.getBoundingClientRect();jump(s.duration*(e.clientX-r.left)/r.width);};
    s.channels[k].forEach(ev=>{
      const seg=document.createElement('div');
      seg.className='seg '+k+(ev.sent?'':' drop');
      seg.style.left=(100*ev.start/s.duration)+'%';
      seg.style.width=(100*Math.max(ev.end-ev.start,.05)/s.duration)+'%';
      seg.title=`${ev.label||name} ${ev.start.toFixed(2)}–${ev.end.toFixed(2)}`;
      seg.onclick=e=>{e.stopPropagation();jump(ev.start,ev.end);};
      track.append(seg);
      const b=document.createElement('button');
      b.textContent=`${ev.label||name} ${ev.start.toFixed(2)}–${ev.end.toFixed(2)}${ev.sent?' · 已送入':' · 未送入'}`;
      b.onclick=()=>jump(ev.start,ev.end);
      events.append(b);
    });
    lanes.append(lane);
  });
  $('prior').textContent=s.prior;
  $('caption').textContent=s.caption;
  $('meta').textContent=`${s.completion_tokens} tokens · ${s.seconds.toFixed(1)} 秒 · finish=${s.finish_reason}`;
  $('prompt').textContent=s.prompt;
  history.replaceState(null,'','?sample='+encodeURIComponent(id));
  await window.reviewAudioLoad(s);
}
function rebuild(){
  const f=$('filter').value;
  visible=data.samples.filter(s=>f==='all'||(f==='cued'?s.cues.length:s.cues.some(c=>!c.written)));
  $('sample').replaceChildren(...visible.map(s=>{
    const o=document.createElement('option');o.value=s.id;
    const tags=s.cues.map(c=>`${c.label}${c.written?'':'未写'}`).join('、')||'无反应线索';
    o.textContent=`#${s.index} · ${s.short} · ${tags}`;
    return o;
  }));
  if(visible.length)show(visible.some(s=>s.id===current?.id)?current.id:visible[0].id);
}
$('sample').onchange=()=>show($('sample').value);
$('filter').onchange=rebuild;
$('prev').onclick=()=>{const i=visible.findIndex(s=>s.id===current?.id);if(visible.length)show(visible[(i-1+visible.length)%visible.length].id);};
$('next').onclick=()=>{const i=visible.findIndex(s=>s.id===current?.id);if(visible.length)show(visible[(i+1)%visible.length].id);};
fetch('review.json').then(r=>r.json()).then(d=>{
  data=d;
  $('spotlights').replaceChildren(...(d.spotlights||[]).map(item=>{
    const box=document.createElement('div');box.className='spot';
    box.innerHTML=`<div><b>【${item.category}】${item.short}</b> · <span class="muted">${item.cue}</span></div>`+
      `<div class="muted" style="margin-top:4px">旧：${item.before}</div>`+
      `<div style="color:#1b5e20;margin-top:2px">新：${item.after}</div>`;
    box.onclick=()=>{$('filter').value='all';rebuild();show(item.sample_id);};
    return box;
  }));
  $('analysis').replaceChildren(...d.analysis.map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
  $('summary').textContent=`${d.summary.samples} 条 · 反应区间 ${d.summary.written}/${d.summary.cues} 写进 · 发声区间 ${d.summary.vocal_written}/${d.summary.vocal_cues} 写进`;
  const id=new URLSearchParams(location.search).get('sample');
  current=data.samples.find(s=>s.id===id);
  rebuild();
});
</script>
</html>
'''

if __name__ == '__main__':
    main()
