from pathlib import Path
import json,csv,hashlib,datetime,re
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;OLD=R/'experiments/p2_vico_full_eval_20261111'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
image=read(O/'image_eval/RESULTS.json');receipt=read(O/'image_eval/COMPLETE.json');motion=read(O/'MOTION_RESULTS.json')
assert receipt['passed'] and receipt['results_sha256']==sha(O/'image_eval/RESULTS.json')
assert image['clips']==43 and image['frames']==16288
assert motion['pairs']==43 and motion['shared_scored_frames']==16238 and motion['metric_parity']['passed']
old=read(R/'experiments/p2_vico43_unified_20261107/avatarforcing/IMAGE_RESULTS.json')
image_delta={k:image['metrics'][k]-v for k,v in old['metrics'].items()}
assert all(__import__('math').isfinite(x) for x in image_delta.values()),image_delta
for p,h in read(OLD/'COMPLETE.json')['files'].items():assert sha(p)==h,p
prior_motion=read(OLD/'unified_emoca/RESULTS.json')
# The identical common frame set must preserve earlier rows, not just rounded values.
with (OLD/'unified_motion_metrics.csv').open() as f:previous={r['method']:r for r in csv.DictReader(f)}
with (O/'motion_metrics.csv').open() as f:current=list(csv.DictReader(f))
for row in current:
    if row['method'] not in previous:continue
    for k,v in row.items():
        if k in previous[row['method']] and k!='method':assert float(v)==float(previous[row['method']][k]),(row['method'],k)
with (OLD/'image_metrics.csv').open() as f:
    reader=csv.DictReader(f);cols=reader.fieldnames;images=list(reader)
images.append(dict(method='Avatar Forcing (audio-only adaptation)',clips=43,frames=image['frames'],**image['metrics'],source=str(O/'image_eval/RESULTS.json'),source_sha256=sha(O/'image_eval/RESULTS.json')))
with (O/'image_metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(images)
reports={'FD':{'exp':16.64,'pose':.05},'rPCC':{'exp':.01,'pose':.01},'SID':{'exp':3.12,'pose':3.00},'Var':{'exp':2.80,'pose':.03}}
afm=motion['methods']['avatarforcing_audio_only']['metrics']
with (OLD/'report_vs_local_all.csv').open() as f:
    reader=csv.DictReader(f);cols=reader.fieldnames;comparison=list(reader)
for metric in ['FD','P_FD','MSE','SID','Var','rPCC']:
    for group in ['exp','pose']:
        comparison.append(dict(method='Avatar Forcing audio-only / unified EMOCA',metric=metric,group=group,reported=reports.get(metric,{}).get(group,''),local=afm[metric][group],difference='',relative_difference_percent='',comparability='descriptive_only_different_visual_condition_extractor_timebase',reported_dataset='ViCo Table4',local_dataset='ViCo test43 common 16238 frames',reported_space='SPECTRE (paper Table4)',local_space='EMOCA_v2_lr_mse_20',source='local AF Table4 reproduction sources; reference_mean_v3/CONTROLLED_REFERENCE_RESULTS.md',notes='Official weights; user_visual zero; original shared audio protocol. Not a claim of paper reproduction. Var is configured per-coefficient temporal population variance.'))
with (O/'report_vs_local_all.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(comparison)
result='''# Avatar Forcing加入当前ViCo统一评测（2026-11-11）

**已完成：43/43条、图像评分16,288帧、统一动作评分16,238个共同有效帧。** 官方生成权重及已生成视频通过SHA核验后复用；本次没有重新生成43条视频。新增执行五项官方图像评分，并把AF加入现有八行共同EMOCA动作评分，全部九行重新计算。其他八行数值不变。

## AF结果与Report并列

本轮是**音频共同输入适配**：listener首帧＋speaker音轨，listener音轨为零，user_visual为零latent；没有读取未来listener视频/音轨。官方完整多模态输入和论文Table4与此不同，论文采用SPECTRE，本表统一读出EMOCA，时间基准也不同。因此以下只并列展示，不根据数值差归咎于官方，也不宣称已经复现论文。

| 指标 | Report Exp | Report Pose | 本地Exp | 本地Pose |
|---|---:|---:|---:|---:|
'''
for k in ['FD','P_FD','MSE','SID','Var','rPCC']:
    rv=reports.get(k,{})
    result+=f"| {k} | {rv.get('exp','—')} | {rv.get('pose','—')} | {afm[k]['exp']:.6f} | {afm[k]['pose']:.6f} |\n"
result+='\nTable4来源及此前SPECTRE复现证据：[V3受控参考图结果](../p2_avatarforcing_table4_reference_mean_20261110_v3/CONTROLLED_REFERENCE_RESULTS.md)、[V4官方256 GT对照](../p2_avatarforcing_table4_canonical_gt_20261110_v4/RESULTS.md)。前者1/8按论文两位小数相同，后者0/8；这些是历史独立实验，不是本次EMOCA对比的通过率。最新[音源分离V5e](../p2_avatarforcing_table4_audio_preparation_cpu_20261110_v5e/README.md)已完成43对/86路并停止，尚未据此重新生成或评分。本次共同评测沿用已冻结音频，不把旧视频标成新分离音轨生成。\n\n## 图像指标同表\n\n| 方法 | PSNR ↑ | SSIM ↑ | FID ↓ | CSIM ↑ | CPBD ↑ |\n|---|---:|---:|---:|---:|---:|\n'
for x in images:result+='| '+x['method']+' | '+' | '.join(f"{float(x[k]):.6f}" for k in ['psnr','ssim','fid','csim','cpbd'])+' |\n'
result+='\nAF五项为本次重测，其他六行引用已封存结果；来源与SHA均在[图像CSV](image_metrics.csv)。图像FID是RGB分布指标，不等于动作FD。\n\n## 统一动作同表\n\n| 方法 | FD Exp ↓ | FD Pose ↓ | P-FD Exp ↓ | P-FD Pose ↓ | MSE Exp ↓ | MSE Pose ↓ |\n|---|---:|---:|---:|---:|---:|---:|\n'
for x in current:result+='| '+x['method']+' | '+' | '.join(f"{float(x[k]):.6f}" for k in ['FD_exp','FD_pose','P_FD_exp','P_FD_pose','MSE_exp','MSE_pose'])+' |\n'
result+='''
43条固定test，listener 1:T对speaker 0:T-1；九行与GT/speaker共同有限帧交集。沿用DIM发布函数，Var按此前确定的逐系数跨帧population方差再平均，ddof=0；原始拉平Var也保存在JSON。数学实现对照16个量最大差0，只证明评分实现一致，不证明论文完整私有协议一致。

[完整动作CSV](motion_metrics.csv)含SID/Var/rPCC；[逐片覆盖与原始结果](MOTION_RESULTS.json)；[全部Report与本地对照CSV](report_vs_local_all.csv)。旧表不同空间的原生行保留各自备注，不能混成同一排序。

## 执行、缓存与恢复

只在node11物理GPU7执行；没有停止GLARE或其他训练。运行脚本：[输入核验与共同动作评分](evaluate.py)、[官方图像评分封装](image_metrics.py)。图像评分复用官方预处理、PIQ FID、ArcFace与CPBD公式，FID/CSIM batch64，CPBD逐片CPU并行并按原帧数加权。已完成metric回执按锁复用。

FID原脚本每个小批次重新构建Inception；本次仅通过PIQ原生feature_extractor参数复用同一模型，其计算函数和模型不变。[一致性记录](FID_CACHE_PARITY.json)：真实GT与预测各64帧，两批特征最大差0。仅这些批次测得的时间不能推广为整批加速比例。重启只终止本次低效FID子进程；其他计算结果和进程保留。重测五项与既有AF评分的差值见[验收记录](FINAL_AUDIT.json)。FID从39.1527885932变为39.1409943196，差−0.0117942736（约−0.0301%），不声称跨运行完全一致。历史封装的batch为256，本轮为64并由gpu_guard关闭TF32、启用确定性cuDNN；这几项执行差异没有逐项消融，因此不将差值单独归因于缓存或任何一个设置。两批cache特征一致性仅在相同当前配置下成立。

项目根目录的恢复命令（已完成时图像阶段直接复用回执）：

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \\
  compare/.envs/vico_eval/bin/python experiments/p2_vico_avatarforcing_added_20261111/image_metrics.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \\
  compare/.envs/vico_more/bin/python experiments/p2_vico_avatarforcing_added_20261111/evaluate.py score
python experiments/p2_vico_avatarforcing_added_20261111/finalize.py
```

换权重、输入、特征或评分协议时使用新实验目录，不能直接复用本目录完成回执。历史评测文件未改写；[输入核验](INPUT_AUDIT.json)和[完成清单](COMPLETE.json)记录来源。

[Avatar Forcing独立交接文档](../../related/method_handoffs/avatarforcing.md) · [全部21个方法文档](../../related/method_handoffs/README.md)。
'''
(O/'README.md').write_text(result)
save(O/'FINAL_AUDIT.json',dict(passed=True,clips=43,image_frames=image['frames'],common_motion_frames=motion['shared_scored_frames'],image_difference_from_archived=image_delta,image_runtime=dict(batch=64,cuda_matmul_tf32=False,cudnn_tf32=False,cudnn_deterministic=True,cudnn_benchmark=False,archived_batch=256,changed_settings_causality='not isolated'),prior_eight_motion_rows_exact=True,prior_sealed_files_unchanged=True,metric_parity=motion['metric_parity'],inference_reused=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
files=[O/x for x in ['README.md','INPUT_AUDIT.json','SCORE_COMPLETE.json','MOTION_RESULTS.json','FINAL_AUDIT.json','FID_CACHE_PARITY.json','FID_CACHE_RESTART.json','image_metrics.csv','motion_metrics.csv','report_vs_local_all.csv','image_eval/RESULTS.json','image_eval/COMPLETE.json','evaluate.py','image_metrics.py','fid_cache.py','check_fid_cache.py','finalize.py']]
save(O/'COMPLETE.json',dict(passed=True,files={str(p):sha(p) for p in files},at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print(json.dumps(dict(completed=True,images=image['metrics'],image_delta=image_delta),indent=2))
