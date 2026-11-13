"""Independent audit and final report for runnable compare models on ViCo43."""
import concurrent.futures
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import sqrtm

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'experiments/p2_compare_vico43_20261106'
DY=WORK/'dystream';L2L=WORK/'learning2listen'
REF=ROOT/'experiments/p2_vico_motion8000_20261106'
PARENT=ROOT/'experiments/p2_vico256_combined8000_20261106'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def covariance_distance(x,y):
    a,b=np.cov(x.T),np.cov(y.T);root=sqrtm(a@b)
    if np.iscomplexobj(root):assert np.abs(root.imag).max()<1e-5;root=root.real
    return max(0,float(np.square(x.mean(0)-y.mean(0)).sum()+np.trace(a+b-2*root)))


def run():
    for f in [DY/'IMAGE_RESULTS.json',DY/'MOTION_RESULTS.json',L2L/'RESULTS.json']:
        if not f.exists():raise RuntimeError(f'Not complete: {f}')
    p=read(DY/'protocol.json');im=read(DY/'IMAGE_RESULTS.json');mo=read(DY/'MOTION_RESULTS.json');l=read(L2L/'RESULTS.json');lp=read(L2L/'protocol.json')
    expected={s['id'] for s in p['samples']};assert len(expected)==43
    assert sha(REF/'RESULTS.json')==read(REF/'FINAL_AUDIT.json')['results_sha256']
    assert sha(PARENT/'RESULTS.json')==read(REF/'protocol.json')['parent_files']['RESULTS.json']
    for path,h in {**p['weights'],**p['code_sha256']}.items():assert sha(path)==h,path
    for q in lp['code_sha256']:assert sha(q)==lp['code_sha256'][q]
    for model,info in lp['models'].items():
        for path,h in info['files'].items():assert sha(path)==h,path
    assert len(lp['models'])==4
    model_loads=list((DY/'model_load').glob('*.json'));assert len(model_loads)>=4
    for f in model_loads:
        r=read(f);assert r['strict_equivalent_load']==[{'missing':[],'unexpected':[]}]
    def inspect(s):
        uid=s['id'];receipt=read(DY/'generated'/uid/'complete.json')
        video=DY/'scoring/dystream'/(uid+'.listener.mp4')
        assert sha(video)==receipt['score_sha256']
        assert sha(DY/'generated'/uid/'generated.mp4')==receipt['native_sha256']
        assert receipt['source_only'] and not receipt['listener_audio_generated']
        probe=read_probe(video)
        assert (probe['width'],probe['height'],probe['r_frame_rate'],int(probe['nb_read_frames']))==(256,256,'30/1',s['frames'])
        mapping=np.array(receipt['mapping']);expected_map=np.minimum(np.floor(np.arange(s['frames'])*25/30+.5).astype(int),receipt['native_frames']-1)
        assert np.array_equal(mapping,expected_map)
        assert receipt['native_frames']==int(round(s['duration']*16000))//640-1
        assert receipt['protocol_sha256']==sha(DY/'protocol.json')
        feature=DY/'motion/features/prediction'/(uid+'.npz');fr=read(feature.with_suffix('.json'))
        assert sha(feature)==fr['feature_sha256'] and fr['input_sha256']==receipt['score_sha256']
        z=np.load(feature);motion=z['motion'].astype(float);coeff=z['coeff']
        assert motion.shape==(s['frames'],70) and np.isfinite(motion).all()
        assert np.array_equal(z['motion'],np.concatenate([coeff[:,80:144],coeff[:,224:227],coeff[:,254:257]],1))
        return dict(id=uid,frames=s['frames'],missing_faces=fr['missing_face_frames'],multi_faces=fr['multi_face_frames'],max_timestamp_error_s=receipt['max_timestamp_error_s'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:checked=list(pool.map(inspect,p['samples']))
    assert sum(r['frames'] for r in checked)==16288
    # Independent scalar FD and SciPy covariance-product distance on real data.
    errors=[];mrows={r['id']:r for r in mo['per_clip']}
    for s in p['samples']:
        uid=s['id'];g=np.load(REF/'features/gt'/(uid+'.npz'))['motion'].astype(float)
        pred=np.load(DY/'motion/features/prediction'/(uid+'.npz'))['motion'].astype(float)
        metrics=mrows[uid]['metrics_raw']
        for group,sl in [('exp',slice(0,64)),('angle',slice(64,67)),('trans',slice(67,70))]:
            assert abs(float(np.abs(g[:,sl]-pred[:,sl]).sum()/g[:,sl].size)-metrics['FD'][group])<1e-12
        for group,sl in [('exp',slice(0,64)),('pose',slice(64,70))]:
            for key,a,b in [('FID_fm',g[:,sl],pred[:,sl]),('FID_delta_fm',np.diff(g[:,sl],axis=0),np.diff(pred[:,sl],axis=0))]:
                errors.append(abs(covariance_distance(a,b)-metrics[key][group]))
            assert abs(metrics['SND'][group]-metrics['FID_fm'][group]-metrics['FID_delta_fm'][group])<1e-10
    assert max(errors)<1e-5
    for key,groups in mo['coverage_clips'].items():assert set(groups.values())=={42 if key=='RTLCC' else 41 if key=='RWTLCC' else 43}
    # Check all original DECA coefficient groups, generated motion and source timings.
    deca=read(L2L/'DECA_PROTOCOL.json')
    for path,h in deca['source'].items():assert sha(path)==h,path
    for s in p['samples']:
        for role in ['speaker','gt']:
            f=L2L/'features'/role/(s['id']+'.npz');z=np.load(f);r=read(f.with_suffix('.json'))
            assert sha(f)==r['feature_sha256'];assert z['coeff'].shape==(s['frames'],236)
            assert np.array_equal(z['motion'],np.concatenate([z['coeff'][:,150:200],z['coeff'][:,200:206]],1))
    for name,result in l['models'].items():
        assert result['clips']==43 and {r['id'] for r in result['per_clip']}==expected
        for s in p['samples']:
            path=L2L/'generated'/name/(s['id']+'.npz');r=read(path.with_suffix('.json'));z=np.load(path)
            assert sha(path)==r['feature_sha256'];assert z['motion'].shape==(s['frames'],56) and np.isfinite(z['motion']).all()
            assert r['source_only'] and not r['listener_future_input']
        for window in read(L2L/'generated'/name/'alignment.json'):
            assert window['audio_mel_start']==window['speaker_start']*4
            assert window['audio_mel_frames']==window['speaker_frames']*4
        for key,groups in result['coverage'].items():assert set(groups.values())=={42 if key=='RTLCC' else 41 if key=='RWTLCC' else 43}
    # Recompute official CPBD frame weighting from individual receipts.
    import pandas as pd
    anno=pd.read_csv(p['datasets']['vico']['annotation']);counts={s['id']:s['frames'] for s in p['samples']}
    assert abs(sum(x['values']['cpbd']*counts[anno.iloc[x['index']].uuid] for x in im['cpbd'])/16288-im['metrics']['cpbd'])<1e-12
    report=dict(passed=True,dy_videos=43,dy_scored_frames=16288,dy_missing_face_frames=sum(r['missing_faces'] for r in checked),
        dy_multi_face_frames=sum(r['multi_faces'] for r in checked),max_timestamp_error_s=max(r['max_timestamp_error_s'] for r in checked),
        dy_independent_scipy_frechet_max_abs_error=max(errors),l2l_models=4,l2l_outputs=172,deca_streams=86,
        source_weights_code_and_parameter_spaces_verified=True,l2l_audio_motion_four_to_one_alignment=True,
        original_joint_results_unchanged=True,checked=checked,
        result_sha256={str(q):sha(q) for q in [DY/'IMAGE_RESULTS.json',DY/'MOTION_RESULTS.json',L2L/'RESULTS.json']})
    save(WORK/'FINAL_AUDIT.json',report)
    rows=[('Joint8000',read(PARENT/'RESULTS.json')['metrics']),('DyStream',im['metrics'])]
    with (WORK/'image_summary.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['model','psnr','ssim','fid','csim','cpbd'])
        for name,values in rows:writer.writerow([name,*[values[k] for k in ['psnr','ssim','fid','csim','cpbd']]])
    status=read(WORK/'MODEL_STATUS.json');status.update(DyStream='43_videos_image_and_motion_scored',Learning2Listen='4_models_x43_DECA_motion_scored')
    save(WORK/'MODEL_STATUS.json',status)
    combined=dict(schema='compare-vico43-available-models-v1',available_video_models=['DyStream'],available_motion_models=list(l['models']),
        model_status=status,dystream=dict(image=im,motion=mo),learning2listen=l,
        reference_joint=dict(image=read(PARENT/'RESULTS.json')['metrics'],motion=read(REF/'RESULTS.json')['metrics_table_scale']))
    save(WORK/'RESULTS.json',combined)
    save(WORK/'COMPLETE_AVAILABLE.json',dict(available_models_finished=True,all_eight_generators_finished=False,
        dystream_clips=43,l2l_outputs=172,blocked_generator_projects=6,at=time.time(),audit='FINAL_AUDIT.json'))
    report_markdown(combined,report)
    print(json.dumps({k:v for k,v in report.items() if k!='checked'},indent=2))


def report_markdown(result,audit):
    im=result['dystream']['image']['metrics'];joint=result['reference_joint']['image']
    lines=['# compare 参考工作 ViCo test43 实测','',
        '**当前可运行的参考模型已完成：DyStream视频43条；Learning2Listen四套人物模型各43条，共172条动作输出。另六项仍缺核心生成权重或生成代码，没有计为测试完成。**','',
        '[完整JSON](RESULTS.json) · [独立验收](FINAL_AUDIT.json) · [模型状态](MODEL_STATUS.json) · [画质CSV](image_summary.csv)','',
        '采用用户确认设置：DyStream target/listener自身音频为全零，真实speaker音频进入other分支，已有listener首帧、官方crop=False；5步／CFG默认值、seed137，EMA权重。DyStream输出动作／视频，不生成listener声音，MP4音轨仅为speaker上下文。Learning2Listen额外使用speaker视频动作，四套person-specific权重全测，不做按GT择优。','',
        '## 同一test43的画质与身份','',
        '| 模型 | PSNR ↑ | SSIM ↑ | 图像FID ↓ | CSIM ↑ | CPBD ↑ |','|---|---:|---:|---:|---:|---:|']
    for name,r in [('联合模型8000',joint),('DyStream listener适配',im)]:lines.append('| '+name+' | '+' | '.join(f'{r[k]:.6f}' for k in ['psnr','ssim','fid','csim','cpbd'])+' |')
    lines+=['','两组均原生512×512／25FPS，评分256×256／30FPS。DyStream官方推理原生每条少1帧，保持原始结果，评分映射以最近时间戳取帧并夹到末帧，不改变速度；最大时间误差 '+f"{audit['max_timestamp_error_s']:.4f}"+'秒。所有43条实际评分副本共16,288帧，已独立解码核对。256评分从官方返回的原始RGB直接抗混叠缩小，避免把原生MP4二次解码损失带入评分。','',
        '## BFM运动指标：联合模型与DyStream','',
        '同一Deep3DFaceRecon权重、原始系数和GT／speaker特征，exp64＋angle3＋trans3；pose为angle＋trans。FD与FID_delta_fm展示乘100，SND不乘100。±2秒lag、4秒窗口／1秒步长、逐片段等权；RTLCC42条、RWTLCC41条，其余43条。属于已明确参数的公开定义复现，不宣称严格复现CustomListener表1。','',
        '| 模型／指标 | exp | angle | trans | pose |','|---|---:|---:|---:|---:|']
    for label,m in [('联合8000',result['reference_joint']['motion']),('DyStream',result['dystream']['motion']['metrics_table_scale'])]:
        for k in ['FD','RTLCC','RWTLCC','FID_delta_fm','SND','V_D']:
            lines.append('| '+label+' '+k+' | '+' | '.join(f'{m[k][g]:.6f}' if g in m[k] else '—' for g in ['exp','angle','trans','pose'])+' |')
    lines+=['','DyStream生成视频漏检总数 '+str(audit['dy_missing_face_frames'])+' 帧，按同一官方前向填充策略处理，不丢弃视频；完整掩码／关键点／系数在 `dystream/motion/features/prediction/`。关键点复核要求实际网络输入像素一致、系数误差<1e−3；未用于对齐的局部关键点差异单独记录，不隐瞒为完全相同。','',
        '## Learning2Listen：独立DECA动作评测','',
        'DECA50维表情＋3维头部旋转向量＋3维下颌旋转向量，与上表BFM空间不同，没有平移系数，不混入同一排名。原生30FPS。没有作者的写实renderer，因此PSNR／SSIM／图像FID／CSIM／CPBD均不计算，也不拿3D示意当写实结果。','',
        '| 人物模型 | 表情L1 ↓ | 头旋转L1 ↓ | 下颌L1 ↓ | 表情Fréchet ↓ | 头旋转Fréchet ↓ |','|---|---:|---:|---:|---:|---:|']
    for n,r in result['learning2listen']['models'].items():
        m=r['metrics'];lines.append('| '+n+' | '+' | '.join(f'{v:.6f}' for v in [m['L1']['exp'],m['L1']['head_rotation'],m['L1']['jaw_rotation'],m['Frechet']['exp'],m['Frechet']['head_rotation']])+' |')
    lines+=['','完整L1／L2／Fréchet／差分Fréchet／SND／V-D／RTLCC／RWTLCC及逐片段结果见 [learning2listen/RESULTS.json](learning2listen/RESULTS.json)。四模型各43条，RTLCC各42条、RWTLCC各41条。','',
        '用户明确确认在独立适配器中修正官方循环音频切片`t`→`4*t`；网络、四套VQ／predictor权重及nucleus sampling不改。前置32帧标准化零历史，末端只补齐VQ计算窗口，输出裁回原始时段；不把初始GT或listener未来输入模型。DECA官方粗编码器严格加载，只提系数，不需要下载FLAME几何或渲染器。音频使用[作者issue2的更正代码](https://github.com/evonneng/learning2listen/issues/2)：16kHz、400窗／160步／512FFT、128维原始mel功率，Lanczos缩放到4T；不是MFCC或log-mel。','',
        '## 暂时不能生成的六项','',
        '| 项目 | 当前具体缺项 |','|---|---|',
        '| DIM | 未找到核心VQ speaker/listener与最终生成checkpoint的公开可用入口 |',
        '| ViCo baseline | listener/speaker driver checkpoint及配套训练统计未发布 |',
        '| MM2022 | 只发布基线改动，没有训练完成的核心driver checkpoint |',
        '| ReactFace | 核心best_checkpoint.pth没有公开链接，renderer/reference也仍需权限 |',
        '| OmniResponse | 官方推理生成代码、renderer、checkpoint尚未发布 |',
        '| SpeakerVid | 已发布的是数据处理／评分工具及其辅助权重，没有对话生成器 |','',
        '重新检查了六个仓库当前HEAD完整树、Releases和相关Issues／评论，未发现新增可用核心生成权重；当前HEAD与已部署版本一致。[本轮官方来源核查](../../compare/deployment/logs/current_generator_audit_20261106.json)。不将环境安装、辅助权重或help成功计为完整生成。','',
        '## 文件与复跑','',
        '- DyStream：`dystream/generated/{uuid}/generated.mp4`，`dystream/scoring/dystream/{uuid}.listener.mp4`；每条complete.json保存来源、原生帧数和完整时间映射。',
        '- Learning2Listen：`learning2listen/generated/{conan,fallon,stephen,trevor}/{uuid}.npz`，包含30FPS的T×56原始motion；配套逐条JSON。',
        '- 协议：[DyStream](dystream/protocol.json)、[DECA](learning2listen/DECA_PROTOCOL.json)、[L2L](learning2listen/protocol.json)。',
        '- 独立适配器：`compare/deployment/vico_dystream.py`、`vico_deca_extract.py`、`vico_l2l.py`、`vico_metrics.py`、`vico_finalize.py`。仅node8物理GPU7，未动0–6卡训练。','',
        '原联合模型的生成、评分和封存验收未修改；所有参考工作均使用同一43条正式test，不含ood42或train398。']
    (WORK/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (WORK/'README.md').write_text('\n'.join(lines)+'\n')


def read_probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(path)]))['streams'][0]


if __name__=='__main__':run()
