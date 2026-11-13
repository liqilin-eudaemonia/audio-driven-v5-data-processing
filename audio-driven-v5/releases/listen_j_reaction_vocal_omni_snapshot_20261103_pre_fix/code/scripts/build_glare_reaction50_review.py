#!/usr/bin/env python3
"""Publish the 50-clip six-reaction candidates next to the caption review media."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/listen_glare_reactions50_20261103'
PAGE = ROOT / 'data_previews/listen_benchmark_review/native_simple/reaction_six50'
NAMES = {
    'nodding': '点头', 'head_shaking': '摇头', 'smiling': '微笑',
    'laughing': '视觉大笑', 'frowning': '皱眉', 'surprised': '惊讶样表情',
}


def main():
    summary = json.loads((RUN / 'summary.json').read_text())
    samples = []
    for sid in json.loads((RUN / 'samples.json').read_text())['ids']:
        row = json.loads((RUN / 'results' / f'{sid}.json').read_text())
        short = sid.split('_S', 1)[1].split('_', 1)[0]
        channels = {}
        for name in NAMES:
            events = []
            for event in row.get('channels', {}).get(name, {}).get('events', []):
                item = {'start': event['start'], 'end': event['end'], 'peak': round(event['peak_score'], 3)}
                if 'fer_detection_fraction' in event:
                    item['fer'] = round(event['fer_detection_fraction'], 3)
                events.append(item)
            channels[name] = events
        samples.append({
            'id': sid, 'short': short, 'duration': row['duration'],
            'video': f'../natural_caption_first_evidence/media/{sid}/video.mp4',
            'fer': None if row.get('fer_detection_fraction') is None else round(row['fer_detection_fraction'], 3),
            'face': None if row.get('face_detection_fraction') is None else round(row['face_detection_fraction'], 3),
            'note': row.get('note', ''),
            'channels': channels,
        })
    PAGE.mkdir(parents=True, exist_ok=True)
    (PAGE / 'data.json').write_text(json.dumps({
        'summary': summary, 'names': NAMES, 'samples': samples,
        'note': '候选区间，不是人工真值。空结果不能当成确认没有该动作。视觉大笑不看音频。',
    }, ensure_ascii=False))
    (PAGE / 'index.html').write_text(HTML)
    print(PAGE)


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>50 段六类反应候选</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#182538;font:15px/1.55 system-ui,sans-serif}
header{padding:16px 22px;background:#fff;border-bottom:1px solid #ccd5df}h1{font-size:22px;margin:0}
main{display:grid;grid-template-columns:minmax(300px,42%) 1fr;gap:16px;padding:16px 22px}
.left{position:sticky;top:10px;align-self:start}video{width:100%;max-height:52vh;background:#000;border-radius:8px}
.card{background:#fff;border:1px solid #d6dee8;border-radius:8px;padding:12px;margin-bottom:12px}
button,select{font:inherit;padding:6px 8px;border:1px solid #aab9c9;border-radius:5px;background:#fff}
nav{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.muted{color:#53657b;font-size:13px}
.lane{display:grid;grid-template-columns:72px 1fr;gap:8px;align-items:center;margin:6px 0}
.track{position:relative;height:22px;background:#eef2f6;border-radius:4px;cursor:pointer}
.seg{position:absolute;top:2px;bottom:2px;border-radius:3px;opacity:.9}
.nodding{background:#2f7d4a}.head_shaking{background:#1d6fbf}.smiling{background:#c9841a}
.laughing{background:#b42318}.frowning{background:#6b4c9a}.surprised{background:#0f766e}
.hit{outline:2px solid #111}.list button{margin:0 6px 6px 0}
@media(max-width:800px){main{display:block}.left{position:static}}
</style>
<header><h1>50 段六类反应候选</h1>
<p>同一批打标抽样。点头沿用高召回；其余五类用参考实现的默认参数。区间可重叠。分数不是概率，空结果也不等于确认没有。</p>
<p class="muted" id="summary"></p></header>
<main><section class="left"><div class="card"><video id="video" controls playsinline></video>
<p id="meta"></p><nav><button id="prev">上一段</button><select id="pick"></select><button id="next">下一段</button>
<select id="filter"><option value="all">全部</option></select></nav></div>
<div class="card" id="lanes"></div></section>
<section><div class="card list" id="events"></div></section></main>
<script>
const names={};let data,rows,i=0,current;
const video=document.getElementById('video');
fetch('data.json').then(r=>r.json()).then(d=>{
  data=d;Object.assign(names,d.names);
  const s=d.summary.channels;
  document.getElementById('summary').textContent=Object.entries(names).map(([k,n])=>`${n} ${s[k].clips}/50 段、${s[k].intervals} 段区间、${s[k].seconds.toFixed(1)} 秒`).join(' · ');
  const filter=document.getElementById('filter');
  Object.entries(names).forEach(([k,n])=>{const o=document.createElement('option');o.value=k;o.textContent='只看有'+n;filter.append(o);});
  filter.onchange=()=>{i=0;apply();};
  document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);show();};
  document.getElementById('next').onclick=()=>{i=Math.min(rows.length-1,i+1);show();};
  document.getElementById('pick').onchange=e=>{i=+e.target.value;show();};
  apply();
});
function apply(){const f=document.getElementById('filter').value;rows=data.samples.filter(s=>f==='all'||s.channels[f].length);show();}
function show(){
  const pick=document.getElementById('pick');pick.replaceChildren();
  rows.forEach((s,n)=>{const o=document.createElement('option');o.value=n;o.textContent=`${s.short} · ${Object.values(s.channels).reduce((a,e)=>a+e.length,0)} 段`;pick.append(o);});
  if(!rows.length){document.getElementById('meta').textContent='没有符合的片段';return;}
  i=Math.min(i,rows.length-1);pick.value=i;current=rows[i];
  video.src=current.video;video.load();
  const fer=current.fer==null?'表情未检测':`FER ${Math.round(current.fer*100)}%`;
  document.getElementById('meta').textContent=`${current.short} · ${current.duration.toFixed(2)} 秒 · 人脸 ${current.face==null?'—':Math.round(current.face*100)+'%'} · ${fer}${current.note?' · '+current.note:''}`;
  const lanes=document.getElementById('lanes');lanes.replaceChildren();
  const events=document.getElementById('events');events.replaceChildren();
  Object.entries(names).forEach(([k,n])=>{
    const lane=document.createElement('div');lane.className='lane';
    lane.innerHTML=`<span>${n}</span><div class="track"></div>`;
    const track=lane.querySelector('.track');
    track.onclick=e=>{const r=track.getBoundingClientRect();video.currentTime=current.duration*(e.clientX-r.left)/r.width;video.play();};
    current.channels[k].forEach(ev=>{
      const seg=document.createElement('div');seg.className='seg '+k;
      seg.style.left=(100*ev.start/current.duration)+'%';seg.style.width=(100*(ev.end-ev.start)/current.duration)+'%';
      seg.title=`${n} ${ev.start.toFixed(2)}–${ev.end.toFixed(2)}`;
      seg.onclick=e=>{e.stopPropagation();video.currentTime=ev.start;video.play();};
      track.append(seg);
      const b=document.createElement('button');b.className=k;b.textContent=`${n} ${ev.start.toFixed(2)}–${ev.end.toFixed(2)}  峰值 ${ev.peak}`;
      b.onclick=()=>{video.currentTime=ev.start;video.play();};
      events.append(b);
    });
    lanes.append(lane);
  });
}
video.ontimeupdate=()=>{
  document.querySelectorAll('.seg').forEach(seg=>{
    const [a,b]=seg.title.match(/[\d.]+/g).map(Number);
    seg.classList.toggle('hit',video.currentTime>=a&&video.currentTime<b);
  });
};
</script>
'''

if __name__ == '__main__':
    main()
