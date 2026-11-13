"""Reuse the existing43-case HTML for DyStream videos and L2L motion curves."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/p2_compare_vico43_20261106'
DY=WORK/'dystream';L2L=WORK/'learning2listen'
SOURCE=ROOT/'data_previews/listen_benchmark_review/p2_vico43_combined8000_20261106'
OUT=ROOT/'data_previews/listen_benchmark_review/p2_vico43_compare_20261106'
URL='http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink():assert dst.resolve()==src.resolve()
    else:dst.symlink_to(src.resolve())
    return str(dst.relative_to(OUT))


def curves(x):
    # Changes relative to each sequence's own first frame; no amplitude
    # normalization. These are DECA parameter traces, not rendered videos.
    x=x.astype(np.float64);diff=x-x[:1]
    return {name:np.sqrt(np.square(diff[:,sl]).mean(1)).round(7).tolist()
            for name,sl in [('expression',slice(0,50)),('head',slice(50,53)),('jaw',slice(53,56))]}


def main():
    assert read(WORK/'FINAL_AUDIT.json')['passed']
    OUT.mkdir(parents=True,exist_ok=True)
    source=read(SOURCE/'data.json');data=copy.deepcopy(source)
    motion=read(DY/'MOTION_RESULTS.json');image=read(DY/'IMAGE_RESULTS.json');l2l=read(L2L/'RESULTS.json')
    mr={r['id']:r for r in motion['per_clip']}
    lr={name:{r['id']:r for r in result['per_clip']} for name,result in l2l['models'].items()}
    prefix='/listen_benchmark_review/'+SOURCE.name+'/'
    for case in data['cases']:
        uid=case['id']
        for key in ['speaker_video','gt_video','speaker_wav','first_frame','raw_gt']:
            case[key]=prefix+case[key]
        case['native_video']=link(DY/'generated'/uid/'generated.mp4',OUT/'media'/uid/'dystream_512.mp4')
        case['scoring_video']=link(DY/'scoring/dystream'/(uid+'.listener.mp4'),OUT/'media'/uid/'dystream_256.mp4')
        case['generated_wav']=None;case['audio']['generated_wav_sha256']=None
        case['audio'].pop('generated_audio_info',None)
        case['source_sha256']['prediction']=read(DY/'generated'/uid/'complete.json')['score_sha256']
        case['audio']['policy']='DyStream generates no listener audio; native video soundtrack is speaker context only; browser uses exact speaker input WAV'
        case['plan']=dict(description='DyStream listener适配：目标自身音频为全零，真实speaker音频进入other分支；同一listener首帧，关闭额外二次裁脸。',prompt='无文本prompt；无Qwen或其他文本规划器。',system='官方checkpoint＋EMA，5步，CFG audio/other=0.5，anchor=0，all=1。',user='seed137；原生512/25，原始RGB抗混叠缩放后256/30评分。')
        case['metrics']=mr[uid]['metrics_raw'];case['timing']=mr[uid]['timing_audit']
        quality=read(DY/'motion/features/prediction'/(uid+'.json'))
        case['quality']['prediction']=dict(missing_faces=quality['missing_face_frames'],multi_face_frames=quality['multi_face_frames'])
        gt=np.load(L2L/'features/gt'/(uid+'.npz'))['motion']
        case['l2l_gt']=curves(gt);case['l2l']={}
        for name in l2l['models']:
            path=L2L/'generated'/name/(uid+'.npz');values=np.load(path)['motion']
            case['l2l'][name]=dict(curves=curves(values),metrics=lr[name][uid]['metrics'],download=link(path,OUT/'motions'/name/(uid+'.npz')))
    data.update(schema='vico43-compare-v1',model='DyStream listener adaptation',summary=motion['metrics_table_scale'],
        coverage=motion['coverage_clips'],image_summary=image['metrics'],l2l_summary={n:r['metrics'] for n,r in l2l['models'].items()},
        source_results_sha256=sha(DY/'MOTION_RESULTS.json'),source_image_sha256=sha(DY/'IMAGE_RESULTS.json'),source_generation_sha256=sha(DY/'GENERATION_COMPLETE.json'),
        l2l_results_sha256=sha(L2L/'RESULTS.json'),source_joint_page=prefix)
    save(OUT/'data.json',data)
    for name,path in [('RESULTS.md',WORK/'RESULTS.md'),('summary.csv',WORK/'image_summary.csv'),('FINAL_AUDIT.json',WORK/'FINAL_AUDIT.json'),
                      ('metrics.json',DY/'MOTION_RESULTS.json'),('protocol.json',DY/'protocol.json'),('SCORING_PROTOCOL.json',DY/'motion/protocol.json'),
                      ('l2l_results.json',L2L/'RESULTS.json')]:shutil.copyfile(path,OUT/name)
    template=(SOURCE/'index.html').read_text()
    template=re.sub(r'<p id="compare-vico-nav">.*?</p>\n?','',template)
    replacements={
        'ViCo test43 · 联合8000步生成对照':'ViCo test43 · DyStream与Learning2Listen',
        'ViCo test43 · AV＋Event＋Feature＋Whisper · 8000步':'ViCo test43 · DyStream视频／Learning2Listen动作',
        '联合模型 · 8000步':'DyStream listener适配',
        '<a href="/vico_paper_test/">ViCo 原画幅审核页</a>':'<a id="jointLink" href="'+prefix+'">联合模型8000 · 同一片段对照</a><a href="/vico_paper_test/">ViCo 原画幅审核页</a>',
        '<option value="both">speaker＋生成（各50%）</option><option value="speaker">仅speaker</option><option value="generated">仅生成声音</option>':'<option value="speaker">speaker上下文音频</option>',
        '<a id="generatedWav" download>原始生成WAV</a>':'<a id="generatedWav" hidden>无listener生成音轨</a>',
        'wav(C.generated_wav,C.audio.generated_wav_sha256)':'Promise.resolve(null)',
        'GT未提供独立listener音轨；试听为实际speaker输入音频与原始生成声音。三路视频自身静音，避免重复播放音轨。':'DyStream不生成listener声音；这里只播放真实speaker上下文音频，不把MP4里的speaker声音冒充listener生成。',
        '本条推理规划与条件':'DyStream本条适配条件',
        '完整规划prompt':'运行参数与条件',
        '规划器：原始Qwen2.5-Omni-3B，无规划LoRA；以下为生成规划，不是真值标注。':'无LLM规划器；以下是用户确认的listener适配条件，不是真值标注。',
        '· 未微调Qwen规划器':'· 无文本规划器',
        '音频来源SHA校验通过。':'speaker音频来源SHA校验通过。',
        'speaker与GT仅做浏览器H264预览转码，不参与评分。':'speaker与GT复用既有H264预览，不参与评分。生成视频音轨仅为speaker上下文。',
    }
    for old,new in replacements.items():
        assert old in template,old;template=template.replace(old,new)
    panel='''<details open id="l2lPanel"><summary>Learning2Listen · 四套人物模型的独立DECA动作结果</summary>
<p class="hint">这些是动作参数曲线，不是写实视频。输入额外包含speaker视频动作；DECA50维表情＋头部旋转3＋下颌旋转3，与上方BFM指标空间不同。</p>
<div class="bar"><label>人物模型 <select id="l2lModel"><option>conan</option><option>fallon</option><option>stephen</option><option>trevor</option></select></label><a id="motionDownload" download>下载本条T×56动作NPZ</a><a href="l2l_results.json" download>完整四模型动作分数</a></div>
<p class="hint">灰色为GT，蓝色为所选模型。显示各组系数相对本片段首帧变化的RMS，保留真实幅度；点击曲线可同步定位视频。竖线表示当前视频时间。</p>
<div class="motionCharts"><figure><figcaption>表情变化 · 50维</figcaption><canvas data-motion="expression" width="600" height="220"></canvas></figure><figure><figcaption>头部旋转向量变化 · 3维</figcaption><canvas data-motion="head" width="600" height="220"></canvas></figure><figure><figcaption>下颌旋转向量变化 · 3维</figcaption><canvas data-motion="jaw" width="600" height="220"></canvas></figure></div>
<div id="l2lMetrics" class="tablewrap"></div></details>'''
    template=template.replace('</section></main>',panel+'\n</section></main>')
    template=template.replace('</style></head>',' .motionCharts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.motionCharts canvas{display:block;width:100%;height:180px;cursor:crosshair}@media(max-width:800px){.motionCharts{grid-template-columns:1fr}}\n</style></head>')
    script='''
function drawL2L(){
 if(!C||!C.l2l)return;const name=$('l2lModel').value,r=C.l2l[name];
 $('motionDownload').href=r.download;
 for(const canvas of document.querySelectorAll('[data-motion]')){
  const k=canvas.dataset.motion,g=C.l2l_gt[k],p=r.curves[k],x=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  const max=Math.max(1e-5,...g,...p)*1.08,l=58,t=12,rgt=w-12,b=h-32;
  x.clearRect(0,0,w,h);x.fillStyle='#fff';x.fillRect(0,0,w,h);x.font='16px sans-serif';x.strokeStyle='#ddd';x.lineWidth=1;
  for(let i=0;i<=4;i++){const yy=t+(b-t)*i/4;x.beginPath();x.moveTo(l,yy);x.lineTo(rgt,yy);x.stroke();x.fillStyle='#666';x.fillText((max*(1-i/4)).toFixed(3),3,yy+5)}
  const line=(a,color)=>{x.strokeStyle=color;x.lineWidth=2;x.beginPath();a.forEach((v,i)=>{const xx=l+(rgt-l)*i/Math.max(1,a.length),yy=b-v/max*(b-t);i?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke()};
  line(g,'#858585');line(p,'#235f8c');const cursor=l+(rgt-l)*videos.gt.currentTime/C.duration;x.strokeStyle='#bb6638';x.beginPath();x.moveTo(cursor,t);x.lineTo(cursor,b);x.stroke();x.fillStyle='#666';x.fillText('0s',l,h-7);x.fillText(C.duration.toFixed(2)+'s',rgt-72,h-7);
 }
 const keys=['L1','L2','Frechet','Delta_Frechet','SND','V_D','RTLCC','RWTLCC'];
 $('l2lMetrics').innerHTML='<table><tr><th>'+name+' · 本条原始尺度</th><th>表情50</th><th>头旋转3</th><th>下颌3</th></tr>'+keys.map(k=>'<tr><td>'+k+'</td>'+['exp','head_rotation','jaw_rotation'].map(g=>'<td>'+fmt(r.metrics[k][g])+'</td>').join('')+'</tr>').join('')+'</table>';
}
$('l2lModel').onchange=drawL2L;
for(const canvas of document.querySelectorAll('[data-motion]'))canvas.onclick=e=>{if(!C)return;const f=(e.offsetX/canvas.clientWidth*600-58)/(600-12-58);seek(Math.max(0,Math.min(1,f))*C.duration);drawL2L()};
setInterval(()=>{if(C&&!changing)drawL2L()},300);
'''
    template=template.replace('async function init(){',script+'\nasync function init(){')
    template=template.replace('await loadMedia();','drawL2L();$(\'jointLink\').href=\''+prefix+'?sample=\'+encodeURIComponent(C.id);await loadMedia();')
    (OUT/'index.html').write_text(template)
    for page in [SOURCE/'index.html',ROOT/'data_previews/vico_paper_test/index.html']:
        text=page.read_text();nav='<p id="compare-vico-nav"><a href="'+URL.replace('http://localhost:8876','')+'">新增：DyStream全部43条视频＋Learning2Listen四套动作曲线 →</a></p>'
        if 'id="compare-vico-nav"' not in text:page.write_text(text.replace('</header>',nav+'\n</header>',1))
    save(WORK/'VISUALIZATION_PUBLISHED.json',dict(url=URL,cases=43,video_methods=['DyStream'],motion_methods=list(l2l['models']),
        reused_from=str(SOURCE/'index.html'),data_sha256=sha(OUT/'data.json'),html_sha256=sha(OUT/'index.html')))
    print(URL)


if __name__=='__main__':
    if (WORK/'three_method_review/PUBLISHED.json').exists():
        from build_vico_three_method_review import main as publish_unified
        publish_unified()
    else:main()
