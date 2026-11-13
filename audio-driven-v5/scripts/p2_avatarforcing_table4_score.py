"""Frozen Table4 profile: shared metric kernels, SPECTRE K15/9 and feature-mean Var."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from p2_avatarforcing_table4_generate import ROOT,WORK,read,save,sha


def histogram_entropy(labels,k):
    h=np.bincount(labels,minlength=k).astype(np.float64);h/=h.sum()
    return float(-np.sum(h*np.log2(h+1e-6)))


def fd_definitions(gt,pred,author):
    mu1,cov1=author.calculate_activation_statistics(gt)
    mu2,cov2=author.calculate_activation_statistics(pred)
    standard=float(author.calculate_frechet_distance(mu1,cov1,mu2,cov2))
    delta=mu1-mu2;covterm=standard-float(delta@delta)
    return dict(standard_squared_l2=standard,
        paper_mean_norm_unsquared_l2=float(np.linalg.norm(delta)+covterm),
        paper_mean_norm_l1=float(np.abs(delta).sum()+covterm))


def coefficient_correlations(a,b):
    x=a.astype(np.float64)-np.mean(a,axis=0)
    y=b.astype(np.float64)-np.mean(b,axis=0)
    denom=np.sqrt(np.sum(x*x,axis=0)*np.sum(y*y,axis=0))
    valid=denom>1e-15
    values=np.full(a.shape[1],np.nan);values[valid]=np.sum(x*y,axis=0)[valid]/denom[valid]
    return values


def metrics(gt,pred,speaker):
    from vico_literature_metrics import load_author
    author=load_author();g,p,x=[np.concatenate(z) for z in [gt,pred,speaker]]
    primary={name:{} for name in ['FD','rPCC','SID','Var']};alternatives={};per_clip=[]
    for a,b in zip(gt,pred):
        per_clip.append({group:fd_definitions(a[:,sl],b[:,sl],author) for group,sl in [('exp',slice(6,56)),('pose',slice(0,6))]})
    for group,sl,k in [('exp',slice(6,56),15),('pose',slice(0,6),9)]:
        fd={name:float(np.mean([r[group][name] for r in per_clip])) for name in per_clip[0][group]}
        primary['FD'][group]=fd['standard_squared_l2']
        gc=float(np.corrcoef(g[:,sl].reshape(-1),x[:,sl].reshape(-1))[0,1])
        pc=float(np.corrcoef(p[:,sl].reshape(-1),x[:,sl].reshape(-1))[0,1])
        primary['rPCC'][group]=abs(gc-pc) if np.isfinite(gc) and np.isfinite(pc) else None
        cluster=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(g[:,sl])
        lab=[cluster.predict(a[:,sl]) for a in pred]
        primary['SID'][group]=histogram_entropy(np.concatenate(lab),k)
        primary['Var'][group]=float(np.var(p[:,sl],axis=0).mean())
        c1,c2=coefficient_correlations(g[:,sl],x[:,sl]),coefficient_correlations(p[:,sl],x[:,sl])
        valid=np.isfinite(c1)&np.isfinite(c2)
        alternatives[group]=dict(FD_equal_clips=fd,FD_all_frames=fd_definitions(g[:,sl],p[:,sl],author),
            SID_equal_clip_entropy=float(np.mean([histogram_entropy(a,k) for a in lab])),
            SID_GT_global=histogram_entropy(cluster.predict(g[:,sl]),k),
            Var_equal_clip_temporal=float(np.mean([np.var(a[:,sl],axis=0).mean() for a in pred])),
            Var_GT_pooled_per_coefficient=float(np.var(g[:,sl],axis=0).mean()),
            rPCC_mean_of_per_coefficient_difference=float(np.mean(np.abs(c1[valid]-c2[valid]))) if valid.any() else None,
            rPCC_valid_coefficients=int(valid.sum()),k=k,cluster_fit='GT only; random_state0 n_init auto')
    return dict(primary=primary,alternatives=alternatives,per_clip_fd=per_clip)


def score():
    complete=read(WORK/'EXTRACTION_COMPLETE.json');assert complete['passed'] and complete['streams']==129
    policy=read(WORK/'protocol.json');ids=[s['id'] for s in policy['samples']]
    features={};failure_audit={}
    for r in complete['records']:
        assert sha(r['path'])==r['sha256']
        with np.load(r['path']) as d:a=d['motion'].copy()
        assert a.shape==(r['frames'],56) and np.isfinite(a).all()
        features[r['id'],r['role']]=a
        failure_audit[r['id'],r['role']]=r['face_audit']
    gt=[features[u,'gt'] for u in ids];pred=[features[u,'avatarforcing'] for u in ids];speaker=[features[u,'speaker'] for u in ids]
    for g,p,x in zip(gt,pred,speaker):assert g.shape==p.shape==x.shape
    result=metrics(gt,pred,speaker)
    comparison=[]
    for metric,groups in result['primary'].items():
        for group,value in groups.items():
            target=policy['target'][metric][group]
            comparison.append(dict(metric=metric,group=group,measured=value,published=target,
                delta=value-target if value is not None else None,
                same_at_published_2dp=(f'{value:.2f}'==f'{target:.2f}') if value is not None else False))
    matched=sum(r['same_at_published_2dp'] for r in comparison)
    result.update(schema='avatarforcing-table4-recheck-result-v1',pairs=43,total_frames=sum(map(len,gt)),
        target_table=policy['target'],comparison=comparison,matched_cells_at_published_precision=matched,total_cells=8,
        numerical_table_match=matched==8,exact_private_run_confirmed=False,
        conclusion='Numerically consistent under executed reconstruction; private settings not proven' if matched==8 else 'Published Table4 not reproduced under the executed profile',
        protocol_sha256=sha(WORK/'protocol.json'),extraction_protocol_sha256=sha(WORK/'SPECTRE_PROTOCOL.json'),
        source_code_sha256=sha(__file__),
        coverage={role:dict(frames=sum(len(features[u,role]) for u in ids),
            interpolated_landmark_frames=sum(len(failure_audit[u,role]['missing_landmark_frames']) for u in ids),
            multi_face_frames=sum(failure_audit[u,role]['multi_face_frames'] for u in ids)) for role in ['speaker','gt','avatarforcing']},
        unresolved_assumptions=['Official checkpoint binding toTable4 not explicitly supplied','Table4 avatar-audio handling not explicit; zeros used',
            'Canonical ViCo crop versus authors source-video crop','Installed FAN tracker versus SPECTRE custom tracking fork',
            'FD mean-term notation and SID pooling not fully specified; alternatives kept'])
    save(WORK/'RESULTS.json',result)
    lines=['# Avatar Forcing Table4：官方权重端到端检查','',
        f"43条，{result['total_frames']}个25fps评分帧；8项中{matched}项在论文两位小数精度下一致。",'',
        '| 指标 | 分组 | 本地实测 | 论文Table4 | 差值 | 两位小数一致 |','|---|---|---:|---:|---:|---|']
    for r in comparison:
        value='N/A' if r['measured'] is None else f"{r['measured']:.6f}"
        delta='N/A' if r['delta'] is None else f"{r['delta']:+.6f}"
        lines.append(f"| {r['metric']} | {r['group']} | {value} | {r['published']:.2f} | {delta} | {'是' if r['same_at_published_2dp'] else '否'} |")
    lines+=['','## 执行口径','',
        '恢复真实speaker视觉和音频，avatar音频全零；官方权重及原生模型，seed137单次、10 NFE、CFG2／1、KV cache。SPECTRE exp50／pose6、25fps同时间戳、完整片段；SID K15／9，GT拟合的全局直方图熵，Var为逐系数时序方差的平均。',
        '主FD使用经过验证的标准Fréchet函数，逐片段等权；rPCC采用拉平后的相关差。公式／汇总歧义的替代结果保存在JSON，不按接近论文数字的程度替换主设置。',
        'SPECTRE保留官方残差加回、五帧时间卷积和边界复制，全部编码器状态严格加载。缺失关键点按官方插值，单列数量，不静默丢片段。','',
        '## 解释边界','',result['conclusion'],'',
        '若不一致，可能来自生成条件、公开checkpoint版本、实际有效测试、提取预处理或评分汇总，不能仅归因于评分器。若一致，则是本篇Avatar Forcing分支对齐的有力证据，仍不能自动证明DIM全部结果及EMOCA/K40/20分支已复现。','',
        '本检查不恢复已暂停的DIM best10，不修改原先source-only适配视频与成绩。','']
    (WORK/'RESULTS.md').write_text('\n'.join(lines))
    save(WORK/'COMPLETE.json',dict(passed_execution=True,numerical_table_match=matched==8,matched_cells=matched,
        results_sha256=sha(WORK/'RESULTS.json'),protocol_sha256=sha(WORK/'protocol.json'),strict_private_protocol_verified=False))
    save(WORK/'status.json',dict(stage='complete',matched_cells=matched,total_cells=8))


if __name__=='__main__':
    argparse.ArgumentParser().parse_args();score()
