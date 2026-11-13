#!/usr/bin/env python3
"""Review page for the 30-clip holdout after the current reaction gates."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/listen_glare_gate30_20261103'
PAGE = ROOT / 'data_previews/listen_benchmark_review/native_simple/reaction_gate30'
NAMES = {'head_shaking': '摇头', 'frowning': '皱眉', 'surprised': '惊讶'}


def short(sample_id):
    return sample_id.split('_S', 1)[1].split('_', 1)[0]


def main():
    ids = json.loads((RUN / 'selection.json').read_text())['ids']
    shake = json.loads((RUN / 'reactions/headshake_direction_v1/diagnostics.json').read_text())
    expression = json.loads((RUN / 'reactions/expression_gate_v1/diagnostics.json').read_text())
    by_id = {sample_id: {'head_shaking': [], 'frowning': [], 'surprised': []} for sample_id in ids}
    for item in shake:
        event = item['event']
        by_id[item['sample_id']]['head_shaking'].append({
            'start': event['start'], 'end': event['end'], 'keep': bool(item['retain']),
        })
    for item in expression:
        by_id[item['sample_id']][item['type']].append({
            'start': item['start'], 'end': item['end'], 'keep': bool(item['retain']),
        })
    media = PAGE / 'media'
    media.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, sample_id in enumerate(ids, start=1):
        result = json.loads((RUN / 'reactions/results' / f'{sample_id}.json').read_text())
        destination = media / f'{sample_id}.mp4'
        source = RUN / 'media' / sample_id / 'video.mp4'
        if not destination.exists():
            os.link(source, destination)
        samples.append({
            'index': index, 'id': sample_id, 'short': short(sample_id),
            'duration': result['duration'], 'video': f'media/{sample_id}.mp4',
            'face': round(result['face_detection_fraction'], 3),
            'fer': round(result['fer_detection_fraction'], 3),
            'channels': by_id[sample_id],
        })
    counts = {}
    for name in NAMES:
        original = [event for sample in samples for event in sample['channels'][name]]
        kept = [event for event in original if event['keep']]
        counts[name] = {'original': len(original), 'kept': len(kept),
                        'seconds': round(sum(event['end'] - event['start'] for event in kept), 2)}
    (PAGE / 'data.json').write_text(json.dumps({
        'samples': samples, 'names': NAMES, 'counts': counts,
        'note': '排除已经看过的 50 条后随机抽取。只筛已有区间，不改起止时间。灰色是被丢掉的候选。',
    }, ensure_ascii=False))
    (PAGE / 'index.html').write_text(HTML)
    print(PAGE)
    print(json.dumps(counts, ensure_ascii=False))


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>30 条筛选结果</title>
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
.seg{position:absolute;top:2px;bottom:2px;border-radius:3px}
.keep{opacity:.9}.drop{background:transparent;border:1.5px solid #8b97a6}
.head_shaking.keep{background:#1d6fbf}.frowning.keep{background:#6b4c9a}.surprised.keep{background:#0f766e}
.hit{outline:2px solid #111}.list button{margin:0 6px 6px 0}.dropbtn{color:#53657b}
@media(max-width:800px){main{display:block}.left{position:static}}
</style>
<header><h1>30 条未看过片段的筛选结果</h1>
<p>检测和上次相同。摇头按左右往复留下，皱眉和惊讶按这段里眉毛相对平时的变化留下。实心是留下的，空心是被丢掉的。时间段没有改。</p>
<p class="muted" id="summary"></p></header>
<main><section class="left"><div class="card"><video id="video" controls playsinline></video>
<p id="meta"></p><nav><button id="prev">上一段</button><select id="pick"></select><button id="next">下一段</button>
<select id="filter"><option value="all">全部</option><option value="kept">只看有留下的</option></select></nav></div>
<div class="card" id="lanes"></div></section>
<section><div class="card list" id="events"></div></section></main>
<script>
let data,rows,i=0,current;
const video=document.getElementById('video');
fetch('data.json').then(r=>r.json()).then(d=>{
  data=d;
  const names=d.names;
  document.getElementById('summary').textContent=Object.entries(names).map(([k,n])=>`${n} ${d.counts[k].original} 个候选留下 ${d.counts[k].kept} 个、${d.counts[k].seconds.toFixed(1)} 秒`).join(' · ');
  const filter=document.getElementById('filter');
  Object.entries(names).forEach(([k,n])=>{const o=document.createElement('option');o.value=k;o.textContent='只看留下'+n;filter.append(o);});
  filter.onchange=()=>{i=0;apply();};
  document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);show();};
  document.getElementById('next').onclick=()=>{i=Math.min(rows.length-1,i+1);show();};
  document.getElementById('pick').onchange=e=>{i=+e.target.value;show();};
  apply();
});
function keptCount(sample,name){return sample.channels[name].filter(e=>e.keep).length;}
function apply(){
  const f=document.getElementById('filter').value;
  rows=data.samples.filter(s=>f==='all'||(f==='kept'?Object.keys(data.names).some(k=>keptCount(s,k)):keptCount(s,f)));
  show();
}
function show(){
  const pick=document.getElementById('pick');pick.replaceChildren();
  rows.forEach((s,n)=>{const o=document.createElement('option');o.value=n;o.textContent=`第${s.index}条 ${s.short}`;pick.append(o);});
  if(!rows.length){document.getElementById('meta').textContent='没有符合的片段';return;}
  i=Math.min(i,rows.length-1);pick.value=i;current=rows[i];
  video.src=current.video;video.load();
  document.getElementById('meta').textContent=`第${current.index}条 ${current.short} · ${current.duration.toFixed(2)} 秒 · 人脸 ${Math.round(current.face*100)}% · FER ${Math.round(current.fer*100)}%`;
  const lanes=document.getElementById('lanes');lanes.replaceChildren();
  const events=document.getElementById('events');events.replaceChildren();
  Object.entries(data.names).forEach(([k,n])=>{
    const lane=document.createElement('div');lane.className='lane';
    lane.innerHTML=`<span>${n}</span><div class="track"></div>`;
    const track=lane.querySelector('.track');
    track.onclick=e=>{const r=track.getBoundingClientRect();video.currentTime=current.duration*(e.clientX-r.left)/r.width;video.play();};
    current.channels[k].forEach(ev=>{
      const seg=document.createElement('div');seg.className='seg '+k+' '+(ev.keep?'keep':'drop');
      seg.style.left=(100*ev.start/current.duration)+'%';seg.style.width=(100*Math.max(ev.end-ev.start,0.05)/current.duration)+'%';
      seg.title=`${n} ${ev.start.toFixed(2)}–${ev.end.toFixed(2)}`;
      seg.onclick=e=>{e.stopPropagation();video.currentTime=ev.start;video.play();};
      track.append(seg);
      const b=document.createElement('button');b.className=k+(ev.keep?'':' dropbtn');
      b.textContent=`${ev.keep?'留下':'丢掉'} ${n} ${ev.start.toFixed(2)}–${ev.end.toFixed(2)}`;
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
