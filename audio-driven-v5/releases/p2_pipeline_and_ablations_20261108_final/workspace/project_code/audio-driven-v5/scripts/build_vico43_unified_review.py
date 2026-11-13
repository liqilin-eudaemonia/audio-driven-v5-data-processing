"""Publish all 43 frozen cases, new AF baseline, scores and paired uncertainty."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_avatarforcing import WORK,read,save,sha

AF=WORK/'avatarforcing'
PAGE=ROOT/'data_previews/listen_benchmark_review/vico43_unified_20261107'
URL='http://localhost:8876/listen_benchmark_review/vico43_unified_20261107/'
LABELS={'joint':'我们 · 联合8000','dystream':'DyStream · listener适配','avatarforcing':'Avatar Forcing · source-only适配'}


def build():
    result=read(WORK/'RESULTS.json')
    old=read(ROOT/'data_previews/listen_benchmark_review/p2_vico43_compare_20261106/data.json')
    p=read(AF/'protocol.json');cases=[]
    PAGE.mkdir(parents=True,exist_ok=True)
    links={'native':AF/'generated','score':AF/'scoring/avatarforcing'}
    for name,target in links.items():
        link=PAGE/name
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        assert link.resolve()==target
    prefix='/listen_benchmark_review/p2_vico43_compare_20261106/'
    def url(x):return x if x.startswith('/') else prefix+x
    models=result['deca']['models']
    for c in old['cases']:
        uid=c['id'];r=read(AF/'generated'/uid/'complete.json')
        assert sha(AF/'generated'/uid/'generated.mp4')==r['native_sha256']
        assert sha(r['score_path'])==r['score_sha256']
        metric={name:next(x['metrics'] for x in m['per_clip'] if x['id']==uid) for name,m in models.items()}
        cases.append(dict(id=uid,duration=c['duration'],frames=c['frames'],speaker=c['speaker_video'],gt=c['gt_video'],
            speaker_audio=c['speaker_wav'],our_audio=c['joint']['generated_wav'],
            native=dict(joint=url(c['joint']['native_video']),dystream=url(c['dystream']['native_video']),
                avatarforcing=f'native/{uid}/generated.mp4'),
            score=dict(joint=url(c['joint']['scoring_video']),dystream=url(c['dystream']['scoring_video']),
                avatarforcing=f'score/{uid}.listener.mp4'),deca=metric,
            detection=read(AF/'features'/(uid+'.json')),
            max_timestamp_error_s=r['max_timestamp_error_s']))
    bfm={n:dict(metrics_raw=m['metrics_raw'],coverage=m['coverage_clips']) for n,m in result['bfm'].items()}
    with (WORK/'bfm_metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['method','metric','group','raw_value','coverage_clips'])
        for n,m in bfm.items():
            for metric,groups in m['metrics_raw'].items():
                for group,value in groups.items():writer.writerow([n,metric,group,value,m['coverage'][metric][group]])
    data=dict(count=43,cases=cases,image=result['images'],deca=models,bfm=bfm,paired=result['paired_deca_difference_joint_minus_baseline'],
        labels=LABELS,source_dialogues=result['source_dialogues'],limitations=result['limitations'])
    save(PAGE/'data.json',data)
    (PAGE/'index.html').write_text(HTML)
    lines=['# ViCo test43：统一本地实测','',f'[完整43条同步视频与评分]({URL}) · [完整结果](RESULTS.json) · [画质CSV](image_metrics.csv) · [逐条动作CSV](deca_per_clip.csv) · [设计与边界](DESIGN.md)','',
        f"三种视频方法使用全部43条正式test、{result['source_dialogues']}个来源对话、40个不同speaker时段；保留不同listener UUID。我们模型固定为先前指定的联合8000，两个baseline使用官方核心模型的source-only适配。每条seed137、一次采样，不按GT择优。原生512/25，评分256/30。",'',
        '| 方法 | PSNR ↑ | SSIM ↑ | FID ↓ | CSIM ↑ | CPBD ↑ |','|---|---:|---:|---:|---:|---:|']
    for name,lab in LABELS.items():
        lines.append('| '+lab+' | '+' | '.join(f"{result['images'][name][k]:.6f}" for k in ['psnr','ssim','fid','csim','cpbd'])+' |')
    lines+=['','以下为同一DECA表情50、头旋转3、下颌3的独立动作比较；不是DIM的EMOCA原表。DECA原始弧度，无平移；不混入BFM数值。','',
        '| 方法 | 表情 L1 ↓ | 头旋转 L1 ↓ | 下颌 L1 ↓ | 表情 Fréchet ↓ | 头旋转 Fréchet ↓ |','|---|---:|---:|---:|---:|---:|']
    for name,lab in LABELS.items():
        m=models[name]['metrics'];vals=[m['L1']['exp'],m['L1']['head_rotation'],m['L1']['jaw_rotation'],m['Frechet']['exp'],m['Frechet']['head_rotation']]
        lines.append('| '+lab+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','## 动作幅度参考','',
        '方差只反映运动幅度，不按高低直接排名；应同时检查不足与过度。','',
        '| 来源 | 表情方差 | 头旋转方差 | 下颌方差 |','|---|---:|---:|---:|']
    for name,lab in [('gt','Listener GT'),*LABELS.items()]:
        m=models['joint']['metrics']['V_D_GT'] if name=='gt' else models[name]['metrics']['V_D']
        lines.append('| '+lab+' | '+' | '.join(f'{m[g]:.8f}' for g in ['exp','head_rotation','jaw_rotation'])+' |')
    lines+=['','## BFM70独立补充','',
        '[三模型全部BFM指标CSV](bfm_metrics.csv)。以下FD是原始L1，未乘100；exp64、angle3、translation3，与DECA分开。','',
        '| 方法 | FD/L1 表情 ↓ | FD/L1 角度 ↓ | FD/L1 平移 ↓ | RTLCC差异/exp ↓ | RWTLCC差异/exp ↓ |','|---|---:|---:|---:|---:|---:|']
    for n,lab in LABELS.items():
        m=bfm[n]['metrics_raw'];vals=[m['FD']['exp'],m['FD']['angle'],m['FD']['trans'],m['RTLCC']['exp'],m['RWTLCC']['exp']]
        lines.append('| '+lab+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','## 配对差值与不确定性','',
        '差值为“我们−baseline”；下面距离指标为负表示我们较低。95%区间使用10000次、seed137，按原始video_id整组重采样，保留同来源片段的相关性。区间仅反映固定模型/固定seed下测试来源的不确定性，不包含重新训练或多seed采样差异；多项指标属于探索性报告。','',
        '| baseline | 指标 | 配对平均差 | 95%区间 | 有效pair/来源 |','|---|---|---:|---|---|']
    for name,r in result['paired_deca_difference_joint_minus_baseline'].items():
        for metric,group in [('L1','exp'),('L1','head_rotation'),('Frechet','exp'),('RTLCC','exp'),('RWTLCC','exp')]:
            v=r[metric][group];ci=v['ci95']
            lines.append(f"| {LABELS[name]} | {metric}/{group} | {v['mean']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {v['clips']}/{v['source_groups']} |")
    lines+=['','## 适配、覆盖与解释','',
        'Avatar Forcing：真实speaker进入user音频，自身音轨全零；user视觉为条件空间零向量，无speaker视频输入。模型保留官方CFG2/1、nfe10和KV cache。已裁好的共同首帧不二次裁脸；完整波形直接调用模型，避免官方CLI30秒上限截掉最长约35秒片段。旧wav2vec weight_norm参数名已显式映射，加载后逐张量相等；无监督、无训练更新。',
        'DyStream：真实speaker进入other，自身音轨全零，同一首帧、crop=False、官方5步。两者均只生成视频，音频指标N/A。网页试听speaker仅为上下文；只有我们模型有生成listener音轨。',
        '图像指标沿用同一官方ViCo实现和聚合。动作逐pair等权，RTLCC42条、RWTLCC41条，其余43条；短片段不补帧。25→30最近时间戳映射重复帧会影响动态指标，逐条映射和尾端误差已保留。',
        '这里RTLCC/RWTLCC是speaker–GT与speaker–prediction相关曲线的绝对差异，越低越好；不是原始相关系数，也不是DIM的rPCC。图像GPU批次为256，旧结果为512；同一eval-mode网络与权重，未声称不同batch的浮点结果逐字节一致。[执行环境与批次](IMAGE_BACKEND_AUDIT.json)。',
        'L2L四套官方人物迁移动作仍全部保留在RESULTS.json和逐条CSV，额外使用speaker motion，未在ViCo重训，也没有写实renderer。暂停的27749/3000/569步诊断均未进入比较。文献表保留在原交接入口，未拼入这张本地表。',
        '训练数据、预算和renderer不同，因此本实验是已有模型在共同输入上的迁移比较，不能声称受控重训或严格复现原论文表。ViCo只有单一GT，图像/动作距离不能独立证明回应是否适当，更不能证明自主声音生成优于不生成声音的baseline。','',
        '## 本轮能支持什么','',
        '在这43条固定输入上，我们的画质/身份匹配和表情参考距离较好，表情L1配对差值区间对两个baseline均为负。头旋转的L1和Fréchet差值区间都跨0，尚无稳定的头姿优势；两种baseline的CPBD更高。',
        f"我们的表情方差为GT的{models['joint']['metrics']['V_D']['exp']/models['joint']['metrics']['V_D_GT']['exp']:.1%}，显示运动幅度偏小。较低的参考距离需要与动作幅度及人工适当性共同解释，不据此宣称回应更充分。",'',
        f'[86项A/B视觉适当性盲评]({URL}blind.html) · [人工协议与空评分表](human_review/PROTOCOL.md)。已准备全部43条×2组比较，模型名隐藏、位置随机且均衡；目前人工评分0份。正式评审仅使用盲评入口，不提前查看带模型名的对照页。','',
        '[Avatar Forcing官方仓库](https://github.com/TaekyungKi/AvatarForcing) · [作者关于视觉置零的说明](https://github.com/TaekyungKi/AvatarForcing/issues/5)','']
    (WORK/'RESULTS.md').write_text('\n'.join(lines))
    save(WORK/'REVIEW.json',dict(url=URL,cases=43,html_sha256=sha(PAGE/'index.html'),data_sha256=sha(PAGE/'data.json')))
    print(URL)


HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ViCo43 · 统一本地对比</title><style>
body{font:15px/1.65 system-ui,sans-serif;background:#111827;color:#e5e7eb;margin:0;padding:24px}main{max-width:1500px;margin:auto}h1{font-size:24px}a{color:#93c5fd}.note{background:#1f2937;padding:16px;border-radius:8px;margin:16px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:8px;border-bottom:1px solid #374151;text-align:right}th:first-child,td:first-child{text-align:left}.overflow{overflow:auto}select,button{padding:8px;background:#374151;color:white;border:1px solid #64748b;border-radius:5px;margin:4px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}video{width:100%;background:black;aspect-ratio:1}.card{background:#1f2937;padding:10px;border-radius:8px}.card span{display:block}input[type=range]{width:90%}pre{white-space:pre-wrap;overflow-wrap:anywhere}#status{color:#fde68a}#clip{max-width:100%}</style>
<main><h1>ViCo test43：共同输入下的本地实测</h1>
<p>我们联合8000 · DyStream listener适配 · Avatar Forcing source-only适配</p>
<div class="note">全部43条、固定seed137、每条一次采样。两个baseline只生成视频；不使用listener真值音轨或后续画面。L2L人物迁移模型见补充表。此处不是DIM原论文表复现，DECA不与EMOCA/BFM混排。</div>
<h2>画质与身份 · 同一官方评分</h2><div id="image" class="overflow"></div>
<h2>共同DECA动作 · 视频反提取</h2><p>本地RTLCC/RWTLCC是相关曲线差异，越低越好；不是原始相关系数或DIM的rPCC。</p><div id="deca" class="overflow"></div>
<h3>动作幅度参考 · 不按高低直接排名</h3><div id="variance" class="overflow"></div><p>参考距离、动作幅度和语境适当性需共同解释；低距离不保证反馈充分。<a href="blind.html">A/B视觉适当性盲评入口（尚无人工结果）</a></p>
<details><summary>BFM70三模型独立补充（FD为原始L1，未乘100）</summary><p>表情64＋角度3＋平移3；与DECA分开。FID_fm和FID_delta_fm为片段内高斯Fréchet。</p><div id="bfm" class="overflow"></div></details>
<details><summary>L2L四个人物模型迁移动作（额外speaker motion，无写实视频）</summary><div id="l2l" class="overflow"></div></details>
<h2>全部片段</h2><select id="clip"></select><select id="geometry"><option value="native">原生生成512/25</option><option value="score">实际评分256/30</option></select>
<select id="audio"><option value="speaker">speaker上下文</option><option value="ours">我们生成的listener音轨</option><option value="mute">静音</option></select>
<div><button id="prev">上一条</button><button id="play">同步播放</button><button id="pause">暂停</button><button id="next">下一条</button></div>
<input id="seek" type="range" min="0" max="1" step=".01" value="0"><span id="time">0.00s</span><p id="status"></p>
<div class="grid"><div class="card"><span>Speaker</span><video id="speaker" muted playsinline preload="metadata"></video></div>
<div class="card"><span>Listener GT（浏览器预览）</span><video id="gt" muted playsinline preload="metadata"></video></div>
<div class="card"><span>我们 · 联合8000</span><video id="joint" muted playsinline preload="metadata"></video></div>
<div class="card"><span>DyStream · listener适配</span><video id="dystream" muted playsinline preload="metadata"></video></div>
<div class="card"><span>Avatar Forcing · source-only适配</span><video id="avatarforcing" muted playsinline preload="metadata"></video></div></div>
<audio id="sound" preload="metadata"></audio><h3>本片段DECA动作</h3><div id="perclip" class="overflow"></div>
<details><summary>边界、检测及覆盖</summary><pre id="receipt"></pre></details>
<h2>配对差值及来源聚类95%区间</h2><p>我们−baseline；距离指标为负表示我们较低。10000次整来源重采样；不包含训练/生成seed变化。集合FID不作逐片段bootstrap。</p><div id="paired" class="overflow"></div>
<p id="limits"></p><a href="../p2_vico43_compare_20261106/">旧版完整视频与BFM/L2L补充</a></main>
<script>
let D,index=0,ready=false,playing=false;const vids=['speaker','gt','joint','dystream','avatarforcing'];const $=x=>document.getElementById(x);const fmt=x=>x==null?'N/A':Number(x).toFixed(5);const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function table(head,rows){return '<table><thead><tr>'+head.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
const cols=[['L1','exp'],['L1','head_rotation'],['L1','jaw_rotation'],['Frechet','exp'],['Frechet','head_rotation'],['RTLCC','exp'],['RWTLCC','exp']];
function motionTable(models,per=false){return table(['方法',...cols.map(x=>x.join('/')+' ↓')],Object.entries(models).map(([n,r])=>[D.labels[n]||n,...cols.map(([m,g])=>fmt((per?r:r.metrics)[m][g]))]));}
function pause(){playing=false;for(const n of vids)$(n).pause();$('sound').pause()}
function sound(){const c=D.cases[index],mode=$('audio').value;$('sound').pause();if(mode!=='mute'){$('sound').src=mode==='speaker'?c.speaker_audio:c.our_audio;$('sound').currentTime=$('speaker').currentTime;if(playing)$('sound').play().catch(e=>$('status').textContent=e.message)}}
async function show(i){pause();ready=false;index=(i+D.cases.length)%D.cases.length;$('clip').value=index;const c=D.cases[index],mode=$('geometry').value;
for(const n of vids){$(n).src=n==='speaker'?c.speaker:n==='gt'?c.gt:c[mode][n];$(n).load()}
$('seek').max=c.duration;$('seek').value=0;$('perclip').innerHTML=motionTable(Object.fromEntries(Object.entries(c.deca).filter(([n])=>D.labels[n])),true);
$('receipt').textContent=JSON.stringify({id:c.id,frames:c.frames,duration:c.duration,max_timestamp_error_s:c.max_timestamp_error_s,AF_missing_face_frames:c.detection.missing_face_frames,AF_multi_face_frames:c.detection.multi_face_frames,coverage:'RTLCC短于完整lag范围N/A；RWTLCC不足4秒N/A；不填充'},null,2);
$('status').textContent='加载 '+c.id;sound();
await Promise.all(vids.map(n=>new Promise((resolve,reject)=>{const v=$(n);if(v.readyState>=1)return resolve();v.addEventListener('loadedmetadata',resolve,{once:true});v.addEventListener('error',()=>reject(Error(n+' media load failed')),{once:true})})));
ready=true;$('status').textContent=`${index+1}/43 · ${c.id} · ${c.duration.toFixed(3)}s`;}
$('play').onclick=async()=>{if(!ready)return;playing=true;await Promise.all(vids.map(n=>$(n).play()));if($('audio').value!=='mute')await $('sound').play()};$('pause').onclick=pause;
$('seek').oninput=()=>{for(const n of vids)$(n).currentTime=Math.min(+$('seek').value,$(n).duration-.001);$('sound').currentTime=+$('seek').value};
$('clip').onchange=()=>show(+$('clip').value).catch(fail);$('geometry').onchange=()=>show(index).catch(fail);$('audio').onchange=sound;$('prev').onclick=()=>show(index-1).catch(fail);$('next').onclick=()=>show(index+1).catch(fail);
function fail(e){$('status').textContent=e.stack||e.message;console.error(e)}
setInterval(()=>{if(!ready)return;const t=$('speaker').currentTime;$('seek').value=t;$('time').textContent=t.toFixed(2)+'s';if(playing){for(const n of vids.slice(1)){const v=$(n);if(Math.abs(v.currentTime-t)>.15&&!v.ended)v.currentTime=Math.min(t,v.duration-.001)}if($('audio').value!=='mute'&&Math.abs($('sound').currentTime-t)>.15)$('sound').currentTime=t;if($('speaker').ended)pause()}},100);
window.reviewDebug=()=>({ready,index,playing,count:D?.count,times:vids.map(n=>$(n).currentTime),errors:vids.filter(n=>$(n).error).map(n=>$(n).error.message)});
fetch('data.json').then(r=>r.json()).then(async d=>{D=d;$('clip').innerHTML=d.cases.map((c,i)=>`<option value="${i}">${i+1} · ${esc(c.id)} · ${c.duration.toFixed(2)}s</option>`).join('');
$('image').innerHTML=table(['方法','PSNR ↑','SSIM ↑','FID ↓','CSIM ↑','CPBD ↑'],Object.entries(d.image).map(([n,r])=>[d.labels[n],...['psnr','ssim','fid','csim','cpbd'].map(k=>fmt(r[k]))]));
$('deca').innerHTML=motionTable(Object.fromEntries(Object.entries(d.deca).filter(([n])=>d.labels[n])));$('l2l').innerHTML=motionTable(Object.fromEntries(Object.entries(d.deca).filter(([n])=>n.startsWith('learning2listen/'))));
$('variance').innerHTML=table(['来源','表情方差','头旋转方差','下颌方差'],[['Listener GT',...['exp','head_rotation','jaw_rotation'].map(g=>d.deca.joint.metrics.V_D_GT[g].toFixed(8))],...Object.keys(d.labels).map(n=>[d.labels[n],...['exp','head_rotation','jaw_rotation'].map(g=>d.deca[n].metrics.V_D[g].toFixed(8))])]);
let br=[];for(const [m,groups] of Object.entries(d.bfm.joint.metrics_raw))for(const g of Object.keys(groups))br.push([(m==='FD'?'FD/L1':m)+'/'+g,...Object.keys(d.labels).map(n=>fmt(d.bfm[n].metrics_raw[m][g])),d.bfm.joint.coverage[m][g]]);$('bfm').innerHTML=table(['指标/组',...Object.values(d.labels),'覆盖pair'],br);
let rows=[];for(const [n,r] of Object.entries(d.paired))for(const [m,g] of cols){const v=r[m][g];rows.push([d.labels[n],m+'/'+g,fmt(v.mean),'['+v.ci95.map(fmt).join(', ')+']',v.clips+'/'+v.source_groups])}$('paired').innerHTML=table(['baseline','指标','我们−baseline','95%区间','pair/来源'],rows);$('limits').textContent=d.limitations.join(' ');await show(0)}).catch(fail);
</script></html>'''


if __name__=='__main__':build()
