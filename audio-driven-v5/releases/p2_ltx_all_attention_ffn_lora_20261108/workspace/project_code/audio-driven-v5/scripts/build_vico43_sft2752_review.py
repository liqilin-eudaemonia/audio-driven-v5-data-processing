"""Append the SFT2752 arm to the frozen ViCo43 comparison without replacing it."""
import copy
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_vico43_sft2752_compare import WORK,BASE,PREVIOUS,ARM,KEY,read,save,sha
from build_vico43_unified_review import HTML

PAGE=ROOT/'data_previews/listen_benchmark_review/vico43_sft2752_20261107'
URL='http://localhost:8876/listen_benchmark_review/vico43_sft2752_20261107/'
LABELS={'joint':'联合8000＋原始Qwen',KEY:'联合8000＋SFT2752','dystream':'DyStream适配','avatarforcing':'Avatar Forcing适配'}


def build():
    previous=read(ROOT/'data_previews/listen_benchmark_review/vico43_unified_20261107/data.json')
    result=read(WORK/'COMPARISON.json');data=copy.deepcopy(previous)
    PAGE.mkdir(parents=True,exist_ok=True)
    for name,source in [('sft_native',WORK/'generated'/ARM/'vico'),('sft_score',WORK/'scoring'/ARM)]:
        dest=PAGE/name
        if not dest.exists():dest.symlink_to(source,target_is_directory=True)
        assert dest.resolve()==source
    source_prefix='/listen_benchmark_review/vico43_unified_20261107/'
    new_metrics={r['id']:r['metrics'] for r in result['deca']['models'][KEY]['per_clip']}
    for case in data['cases']:
        uid=case['id']
        for kind in ['native','score']:
            for name,url in case[kind].items():
                if not url.startswith('/'):case[kind][name]=source_prefix+url
        case['native'][KEY]=f'sft_native/{uid}/generated.mp4'
        case['score'][KEY]=f'sft_score/{uid}.listener.mp4'
        case['sft_audio']=f'sft_native/{uid}/listener.wav'
        case['planning']={
            '原始Qwen':read(BASE/'conditions/vico'/uid/'qwen/plan.json')['description'],
            'SFT2752':read(WORK/'conditions/vico'/uid/'qwen/plan.json')['description'],
        }
        case['deca'][KEY]=new_metrics[uid]
        case['detection']=read(WORK/'features'/(uid+'.json'))
        case['max_timestamp_error_s']=read(WORK/'conversion'/(uid+'.json'))['max_timestamp_error_s']
    data.update(labels=LABELS,image={n:result['images'][n] for n in LABELS},deca=result['deca']['models'],
        bfm={n:dict(metrics_raw=result['bfm'][n]['metrics_raw'],coverage=result['bfm'][n]['coverage_clips']) for n in LABELS},
        paired=result['paired_sft_minus_reference'],
        limitations=result['limitations']+['Both our arms use identical combined8000; SFT2752 changes the planner and its predicted H/Gemma only.'])
    save(PAGE/'data.json',data)
    html=HTML
    changes=[
        ('ViCo43 · 统一本地对比','ViCo43 · 原始Qwen vs SFT2752'),
        ('ViCo test43：共同输入下的本地实测','ViCo43：固定联合8000，原始Qwen vs SFT2752'),
        ('我们联合8000 · DyStream listener适配 · Avatar Forcing source-only适配','联合8000固定 · 原始Qwen／SFT2752 · DyStream／Avatar Forcing参考'),
        ("vids=['speaker','gt','joint','dystream','avatarforcing']",f"vids=['speaker','gt','joint','{KEY}','dystream','avatarforcing']"),
        ('<span>我们 · 联合8000</span>','<span>联合8000＋原始Qwen</span>'),
        ('<div class="card"><span>DyStream',f'<div class="card"><span>联合8000＋SFT2752</span><video id="{KEY}" muted playsinline preload="metadata"></video></div>\n<div class="card"><span>DyStream'),
        ('<option value="ours">我们生成的listener音轨</option>','<option value="ours">原始Qwen版listener音轨</option><option value="latest">SFT2752版listener音轨</option>'),
        ("mode==='speaker'?c.speaker_audio:c.our_audio","mode==='speaker'?c.speaker_audio:mode==='latest'?c.sft_audio:c.our_audio"),
        ('我们−baseline','SFT2752版−对照'),
        ('距离指标为负表示我们较低','距离指标为负表示SFT2752版较低'),
        ('BFM70三模型独立补充','BFM70四模型独立补充'),
        ('<details><summary>边界、检测及覆盖</summary>',
            '<details><summary>本片段：两版Qwen规划描述</summary><p>这是模型预测的反应计划，不是真值描述。</p><pre id="planning"></pre></details><details><summary>边界、检测及覆盖</summary>'),
        ("$('seek').max=c.duration;", "$('planning').textContent=JSON.stringify(c.planning,null,2);$('seek').max=c.duration;"),
        ('AF_missing_face_frames','SFT_missing_face_frames'),('AF_multi_face_frames','SFT_multi_face_frames'),
        ('<a href="blind.html">A/B视觉适当性盲评入口（尚无人工结果）</a>','本轮尚未进行人工语境适当性评价。'),
    ]
    for old,new in changes:
        assert old in html,old
        html=html.replace(old,new)
    (PAGE/'index.html').write_text(html)
    pilot=PAGE/'pilot.html'
    if pilot.exists():
        content=pilot.read_text()
        banner='<p><a href="./">查看完整43条最新结果</a>；下方保留首批9条历史预览。</p>'
        if banner not in content:pilot.write_text(content.replace('<main>','<main>'+banner,1))
    lines=['# ViCo43：固定联合8000，原始Qwen vs SFT2752','',
        f'[全部43条同步视频和评分]({URL}) · [四模型结果JSON](COMPARISON.json) · [规划描述差异](planning_comparison.csv) · [图像CSV](image_comparison.csv) · [论文LaTeX表及来源](PAPER_TABLES.json)','',
        '同一联合8000权重、同一43条speaker波形／listener首帧、同一Whisper、seed137／Euler30／CFG3与7。只替换Qwen为SFT2752，重新生成描述、提取自身H并编码Gemma；提示词模板相同，未改生成模型或训练。训练使用base H，本次推理使用SFT H。','',
        '| 模型 | PSNR ↑ | SSIM ↑ | FID ↓ | CSIM ↑ | CPBD ↑ |','|---|---:|---:|---:|---:|---:|']
    for name,label in LABELS.items():
        lines.append('| '+label+' | '+' | '.join(f"{result['images'][name][k]:.6f}" for k in ['psnr','ssim','fid','csim','cpbd'])+' |')
    lines+=['','| 模型 | 表情L1 ↓ | 头旋转L1 ↓ | 下颌L1 ↓ | 表情Fréchet ↓ | 表情方差 |','|---|---:|---:|---:|---:|---:|']
    for name,label in LABELS.items():
        m=result['deca']['models'][name]['metrics'];v=[m['L1']['exp'],m['L1']['head_rotation'],m['L1']['jaw_rotation'],m['Frechet']['exp'],m['V_D']['exp']]
        lines.append('| '+label+' | '+' | '.join(f'{x:.6f}' for x in v)+' |')
    lines+=['','## SFT2752−原始Qwen：来源聚类配对区间','',
        '10000次整video_id聚类bootstrap，seed137；10个来源，固定单次生成。距离指标为负表示SFT版较低，方差不直接按高低排名。','',
        '| 指标 | 平均差 | 95%区间 | pair/来源 |','|---|---:|---|---|']
    for metric,groups in result['paired_sft_minus_reference']['joint'].items():
        for group,v in groups.items():
            lines.append(f"| {metric}/{group} | {v['mean']:.8f} | [{v['ci95'][0]:.8f}, {v['ci95'][1]:.8f}] | {v['clips']}/{v['source_groups']} |")
    changed=read(WORK/'CONDITION_DIFFERENCE.json')
    lines+=['',f"规划描述改变{changed['description_changed']}/43条，H改变{changed['hidden_changed']}/43条；源媒体相同。",'',
        'BFM70和DECA56分表，不与DIM EMOCA文献表混排。DyStream／Avatar Forcing和L2L四套迁移结果复用封存数据。两种视频baseline不生成listener音频；此处只比较共同视觉／动作指标，不用ViCo证明自主声音生成优越性。',
        '图像／动作参考距离不等于语境适当性；同时保留动作方差与GT，人工评价尚未开展。原始Qwen版、输入缓存及旧页面保持不变。','']
    (WORK/'RESULTS.md').write_text('\n'.join(lines))
    from export_vico43_sft2752_paper_tables import export
    export()
    save(WORK/'REVIEW.json',dict(url=URL,cases=43,video_methods=4,html_sha256=sha(PAGE/'index.html'),data_sha256=sha(PAGE/'data.json'),
        paper_tables_manifest_sha256=sha(WORK/'PAPER_TABLES.json')))
    print(URL)


if __name__=='__main__':build()
