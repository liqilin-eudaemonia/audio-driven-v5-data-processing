"""Prepare (but do not fabricate) visual appropriateness judgments on all43."""
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_avatarforcing import WORK,read,save


def prepare():
    page=ROOT/'data_previews/listen_benchmark_review/vico43_unified_20261107'
    page.mkdir(parents=True,exist_ok=True)
    native=page/'native';target=WORK/'avatarforcing/generated'
    if not native.exists():native.symlink_to(target,target_is_directory=True)
    assert native.resolve()==target
    if (page/'data.json').exists():data=read(page/'data.json')
    else:
        # Human assessment needs complete frozen videos, not completed metrics.
        assert read(WORK/'avatarforcing/GENERATION_COMPLETE.json')['clips']==43
        old=read(ROOT/'data_previews/listen_benchmark_review/p2_vico43_compare_20261106/data.json')
        prefix='/listen_benchmark_review/p2_vico43_compare_20261106/'
        url=lambda x:x if x.startswith('/') else prefix+x
        data=dict(cases=[dict(id=c['id'],duration=c['duration'],speaker_audio=c['speaker_wav'],
            native=dict(joint=url(c['joint']['native_video']),dystream=url(c['dystream']['native_video']),
                        avatarforcing=f"native/{c['id']}/generated.mp4")) for c in old['cases']])
    tasks=[];key=[];rng=random.Random(20261107)
    items=[(c,b) for c in data['cases'] for b in ['dystream','avatarforcing']]
    rng.shuffle(items)
    sides={'dystream':[True]*22+[False]*21,'avatarforcing':[True]*21+[False]*22}
    for s in sides.values():rng.shuffle(s)
    dest=WORK/'human_review';dest.mkdir(exist_ok=True)
    media=page/'blind_media';media.mkdir(exist_ok=True)
    for i,(c,b) in enumerate(items):
        our_left=sides[b].pop()
        uid=c['id'];tid=hashlib.sha256(f'vico43-pair-20261107-{i}-{uid}-{b}'.encode()).hexdigest()[:12]
        left,right=('joint',b) if our_left else (b,'joint')
        spec=dict(task_id=tid,duration=c['duration'])
        for label,url in [('context',c['speaker_audio']),('A',c['native'][left]),('B',c['native'][right])]:
            source=(ROOT/'data_previews'/url.lstrip('/')) if url.startswith('/') else page/url
            source=source.resolve();assert source.is_file(),source
            name=f'{tid}-{label}'+('.wav' if label=='context' else '.mp4')
            link=media/name
            if not link.exists():link.symlink_to(source)
            assert link.resolve()==source
            spec[label]='blind_media/'+name
        tasks.append(spec);key.append(dict(task_id=tid,uuid=uid,baseline=b,A=left,B=right,our_side='A' if our_left else 'B'))
    save(page/'blind_tasks.json',dict(scope='visual only, both generated soundtracks muted, context=speaker audio',seed=20261107,tasks=tasks))
    save(dest/'private_key.json',key)
    with (dest/'ratings_blank.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['rater_id','task_id','appropriateness_A_B_tie_invalid','naturalness_A_B_tie_invalid','engagement_A_B_tie_invalid','comment'])
        for t in tasks:w.writerow(['',t['task_id'],'','','',''])
    (page/'blind.html').write_text(HTML)
    (dest/'PROTOCOL.md').write_text('''# ViCo43 视觉适当性盲评准备包

当前只是准备包，**没有人工评分结果**，不得把空表当成已完成人评。

共86项：每个pair比较我们与DyStream、我们与Avatar Forcing各一次；43条全部保留，不挑例子。展示顺序、A/B位置固定seed20261107随机，左右整体各43次。映射保存在本目录private_key.json，不进入公开评审页面。

评审试听共同的speaker上下文，两个候选视频均静音，不展示GT、模型名、自动分数。只评价视觉反馈，不用于声学生成或AV协调性结论。页面底部导出CSV。建议至少3名独立评审，不讨论答案；同一来源内片段高度相关，汇总时按源video_id聚类，并同时报告评审间一致性、tie/invalid数。

- 适当性：反馈是否与当前话语相容，是否有明显过度、冲突或不合时宜的动作。
- 自然度：动态是否连贯，是否有抖动、塌脸或明显失真的动作。
- 参与感：是否表现出适当的倾听参与；安静倾听可以合理，不能把动作数量多当作更好。
- 每项可选A更好、B更好、相当、无法评价；不凭PSNR或与唯一GT动作的一致性代替判断。

不预填评分、不用自动模型冒充人类。没有盲评前，自动分数只能支持画质/身份/参考动作距离，不能升级为语境适当性结论。
''')
    save(dest/'PREPARED.json',dict(tasks=86,unique_pairs=43,ratings_collected=0,labels_hidden=True,
        url='http://localhost:8876/listen_benchmark_review/vico43_unified_20261107/blind.html'))


HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>倾听反馈 · A/B盲评</title>
<style>body{font:16px/1.7 system-ui;background:#f3f4f6;padding:20px;max-width:1100px;margin:auto}video{width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}button,select,input{font:inherit;padding:6px;margin:5px}label{display:block}#id{overflow-wrap:anywhere}</style>
<h1>倾听反馈 · A/B盲评</h1><p>先听speaker上下文，再比较A/B的视觉反馈。两路生成音轨均静音。安静倾听可以合理，动作更多不代表更好。</p><label>评审代号 <input id="rater"></label><p id="id"></p><audio id="context" controls></audio>
<div class="grid"><section><h2>A</h2><video id="A" muted playsinline preload="metadata"></video></section><section><h2>B</h2><video id="B" muted playsinline preload="metadata"></video></section></div>
<button id="play">从头同步播放</button><button id="stop">暂停</button><div id="ratings"></div><label>备注 <input id="comment"></label><button id="prev">保存并上一项</button><button id="next">保存并下一项</button><button id="export">导出CSV</button><p id="status"></p>
<script>let tasks=[],i=0,playing=false;const $=x=>document.getElementById(x),names=['appropriateness','naturalness','engagement'],labels=['语境适当性','动作自然度','适当的倾听参与感'];let rows=JSON.parse(localStorage.getItem('vico43blind20261107')||'{}');
$('ratings').innerHTML=names.map((n,j)=>`<label>${labels[j]} <select id="${n}"><option value="">请选择</option><option>A</option><option>B</option><option value="tie">相当</option><option value="invalid">无法评价</option></select></label>`).join('');
function stop(){playing=false;for(const n of ['context','A','B'])$(n).pause()}function store(){if(!tasks.length)return;const t=tasks[i];rows[t.task_id]={rater_id:$('rater').value,task_id:t.task_id,...Object.fromEntries(names.map(n=>[n,$(n).value])),comment:$('comment').value};localStorage.setItem('vico43blind20261107',JSON.stringify(rows))}
function show(n){stop();i=(n+tasks.length)%tasks.length;const t=tasks[i];for(const k of ['context','A','B']){$(k).src=t[k];$(k).load()}$('id').textContent=(i+1)+'/'+tasks.length+' · '+t.task_id;const r=rows[t.task_id]||{};for(const k of names)$(k).value=r[k]||'';$('comment').value=r.comment||'';if(r.rater_id)$('rater').value=r.rater_id}
$('play').onclick=async()=>{for(const k of ['context','A','B'])$(k).currentTime=0;await Promise.all(['context','A','B'].map(k=>$(k).play()))};$('stop').onclick=stop;$('prev').onclick=()=>{store();show(i-1)};$('next').onclick=()=>{store();show(i+1)};
$('context').onplay=()=>{playing=true;for(const k of ['A','B'])$(k).play().catch(e=>$('status').textContent=e.message)};$('context').onpause=()=>{playing=false;for(const k of ['A','B'])$(k).pause()};
setInterval(()=>{if(!playing)return;const t=$('context').currentTime;for(const k of ['A','B']){const v=$(k);if(Number.isFinite(v.duration)&&Math.abs(v.currentTime-t)>.12)v.currentTime=Math.min(t,v.duration-.001)}if($('context').ended)stop()},100);
$('export').onclick=()=>{store();const cols=['rater_id','task_id',...names,'comment'];const quote=x=>'"'+String(x||'').replaceAll('"','""')+'"';let csv=cols.map(quote).join(',')+'\\n'+Object.values(rows).map(r=>cols.map(k=>quote(r[k])).join(',')).join('\\n');let a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+csv],{type:'text/csv;charset=utf-8'}));a.download='vico43_visual_ratings.csv';a.click()};
fetch('blind_tasks.json').then(r=>r.json()).then(d=>{tasks=d.tasks;show(0)}).catch(e=>$('status').textContent=e.message);window.blindDebug=()=>({loaded:tasks.length,index:i,task:tasks[i]?.task_id});</script></html>'''


if __name__=='__main__':prepare()
