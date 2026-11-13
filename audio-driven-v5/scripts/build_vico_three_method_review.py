"""Unify existing three-method results; preserve all original evaluation files."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/p2_compare_vico43_20261106'
OUT=ROOT/'data_previews/listen_benchmark_review/p2_vico43_compare_20261106'
JOINT=ROOT/'data_previews/listen_benchmark_review/p2_vico43_combined8000_20261106'
ART=WORK/'three_method_review'
URL='http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');temp.replace(p)


def main():
    ART.mkdir(parents=True,exist_ok=True);previous=ART/'previous';previous.mkdir(exist_ok=True)
    for name in ['index.html','data.json']:
        if not (previous/name).exists():shutil.copyfile(OUT/name,previous/name)
    old=read(previous/'data.json');joint=read(JOINT/'data.json')
    assert len(old['cases'])==len(joint['cases'])==43
    jcases={c['id']:c for c in joint['cases']};cases=[]
    prefix='/listen_benchmark_review/'+JOINT.name+'/'
    for c in old['cases']:
        c=copy.deepcopy(c);j=jcases[c['id']]
        assert c['frames']==j['frames'] and c['duration']==j['duration']
        c['joint']=dict(native_video=prefix+j['native_video'],scoring_video=prefix+j['scoring_video'],
            generated_wav=prefix+j['generated_wav'],audio_sha256=j['audio']['generated_wav_sha256'],
            source_prediction_sha256=j['source_sha256']['prediction'],metrics=j['metrics'],quality=j['quality']['prediction'],plan=j['plan'])
        c['dystream']=dict(native_video=c['native_video'],scoring_video=c['scoring_video'],generated_wav=None,
            source_prediction_sha256=c['source_sha256']['prediction'],metrics=c['metrics'],quality=c['quality']['prediction'],plan=c['plan'])
        cases.append(c)
    l2l=read(WORK/'learning2listen/RESULTS.json')
    common_path=ROOT/'experiments/common_method_comparison_20261106/vico_deca/RESULTS.json'
    common=None
    if (common_path.parent/'COMPLETE.json').exists():
        assert read(common_path.parent/'FINAL_AUDIT.json')['passed']
        shared=read(common_path)
        retrain_root=ROOT/'experiments/common_method_comparison_20261106'
        registry=retrain_root/'ACTIVE_L2L_RUN.json'
        retrained_path=(Path(read(registry)['path']) if registry.exists() else retrain_root/'l2l_retrained')/'RESULTS.json'
        gate=read(retrained_path.parent/'QUALITY_GATE.json') if (retrained_path.parent/'QUALITY_GATE.json').exists() else {}
        if (retrained_path.parent/'COMPLETE.json').exists() and gate.get('passed'):
            assert read(retrained_path.parent/'FINAL_AUDIT.json')['passed']
            shared['models']['learning2listen/retrained']=read(retrained_path)
        common=dict(space=shared['space'],limitations=shared['limitations'],models={
            name:dict(metrics=r['metrics'],coverage=r['coverage'],readout=r['readout']) for name,r in shared['models'].items()})
        per_case={name:{r['id']:r for r in model['per_clip']} for name,model in shared['models'].items()}
        for case in cases:
            case['common_deca']={name:rows[case['id']]['metrics'] for name,rows in per_case.items()}
    summaries=dict(joint=dict(label='联合模型 · 8000步',image=joint['image_summary'],motion=joint['summary'],coverage=joint['coverage'],space='BFM64+3+3',video=True),
        dystream=dict(label='DyStream · listener适配',image=old['image_summary'],motion=old['summary'],coverage=old['coverage'],space='BFM64+3+3',video=True),
        learning2listen=dict(label='Learning2Listen',variants={n:dict(metrics=r['metrics'],coverage=r['coverage']) for n,r in l2l['models'].items()},
            space='DECA50+head_rotation3+jaw_rotation3',image=None,video=False,reason='未公开写实renderer；现有产物为动作参数，暂无写实推理视频'))
    sources=[JOINT/'data.json',WORK/'dystream/IMAGE_RESULTS.json',WORK/'dystream/MOTION_RESULTS.json',WORK/'learning2listen/RESULTS.json']
    if common:sources.append(common_path)
    if common and 'learning2listen/retrained' in common['models']:sources.append(retrained_path)
    data=dict(schema='vico43-three-method-review-v1',count=43,method_families=3,video_methods=2,
        cases=cases,summaries=summaries,common_deca=common,source_sha256={str(p):sha(p) for p in sources},
        table_scale=dict(FD=100,FID_delta_fm=100,others=1),
        limits='BFM and DECA reported in distinct tables; L2L has four person-specific variants and no photorealistic video; DyStream has no generated listener audio')
    save(ART/'data.json',data);save(OUT/'data.json',data)
    with (ART/'all_methods_metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['method','variant','space','metric','group','raw_value','display_multiplier','coverage_clips'])
        for name in ['joint','dystream']:
            for k,v in summaries[name]['image'].items():writer.writerow([name,'','image',k,'',v,1,43])
            for k,groups in summaries[name]['motion'].items():
                scale=100 if k in ['FD','FID_delta_fm'] else 1
                for g,v in groups.items():writer.writerow([name,'','BFM64+3+3',k,g,v/scale,scale,summaries[name]['coverage'][k][g]])
        for name,r in l2l['models'].items():
            for k,groups in r['metrics'].items():
                for g,v in groups.items():writer.writerow(['learning2listen',name,'DECA50+3+3',k,g,v,1,r['coverage'][k][g]])
    shutil.copyfile(ART/'all_methods_metrics.csv',OUT/'all_methods_metrics.csv')
    lines=['# ViCo test43：三种方法统一对照','',f'[打开统一页面]({URL}) · [完整指标CSV](all_methods_metrics.csv) · [页面数据JSON](data.json)','',
        '三种方法为联合模型8000步、DyStream、Learning2Listen。前两种提供真实推理视频，可与speaker及listener GT同步播放；Learning2Listen仅提供四套人物模型的动作参数，没有公开写实renderer，图像指标标N/A。','',
        '| 方法 | PSNR ↑ | SSIM ↑ | 图像FID ↓ | CSIM ↑ | CPBD ↑ |','|---|---:|---:|---:|---:|---:|']
    for name in ['joint','dystream']:
        r=summaries[name];lines.append('| '+r['label']+' | '+' | '.join(f'{r["image"][k]:.6f}' for k in ['psnr','ssim','fid','csim','cpbd'])+' |')
    lines+=['| Learning2Listen（四套） | N/A | N/A | N/A | N/A | N/A |','',
        '新增共同DECA动作评分：联合模型、DyStream的视频统一反提取DECA，与四套L2L原生动作在同一50＋3＋3空间评分。总体及逐片段结果均已接入页面，完整原始数值见 `experiments/common_method_comparison_20261106/vico_deca/RESULTS.json`。原生动作与视频反提取的输出路径、训练条件仍不同，不能宣称是DIM原表复现。','',
        'BFM运动结果在页面并列表格中逐项对照；FD与FID_delta_fm展示乘100。Learning2Listen的DECA表情50维、头部旋转3维及下颌旋转3维单独列出，未与BFM混算或跨空间排名。RTLCC覆盖42条，RWTLCC覆盖41条，其余指标43条；不适用的短片段标N/A。','',
        '同步播放器默认原生512×512／25FPS，可切换实际256×256／30FPS评分副本。试听可以选择speaker、联合模型生成声音、二者各50%混合或静音。DyStream不生成listener声音，视频自带speaker上下文音轨始终静音。','',
        'Learning2Listen四套人物权重均保留，无按GT择优选择；可在同一片段切换参数曲线与逐条指标。它额外使用speaker视频动作，相关输入与4t音频对齐修正沿用已确认协议。原模型、推理产物和评测JSON不变，本次只重新组织展示。','',
        '完整原始评分与资源限制见原评测目录 `experiments/p2_compare_vico43_20261106/RESULTS.md`。']
    (ART/'RESULTS.md').write_text('\n'.join(lines)+'\n');shutil.copyfile(ART/'RESULTS.md',OUT/'THREE_METHOD_RESULTS.md')
    html=(ROOT/'scripts/vico_three_method_review.html').read_text()
    base=re.search(r'<style>(.*?)</style>',(ROOT/'data_previews/vico_paper_test/index.html').read_text(),re.S).group(1)
    html=html.replace('{{BASE_STYLES}}',base)
    temp=OUT/'index.html.tmp';temp.write_text(html);temp.replace(OUT/'index.html')
    save(ART/'PUBLISHED.json',dict(url=URL,method_families=3,video_methods=['joint','dystream'],motion_only_method='learning2listen',
        variants=['conan','fallon','stephen','trevor'],samples=43,data_sha256=sha(OUT/'data.json'),html_sha256=sha(OUT/'index.html'),previous_snapshots=str(previous)))
    print(URL)


if __name__=='__main__':main()
