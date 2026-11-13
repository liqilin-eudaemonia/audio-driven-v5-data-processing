"""Versioned per-coefficient variance readout; old results stay immutable."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(config,output):
    policy=read(config)
    assert policy['schema']=='vico-dim-variance-readout-v1'
    assert policy['variance_definition']=='pooled_per_coefficient_mean' and policy['ddof']==0
    assert policy['pooling']=='concatenate_scored_frames_across_clips_then_per_coefficient_variance_then_mean'
    assert policy['groups']=={'pose':[0,6],'exp':[6,56]}
    source_path=ROOT/policy['source_results'];source=read(source_path);folder=source_path.parent
    source_seal=read(folder/'COMPLETE.json')
    for path,h in source_seal['files'].items():assert sha(path)==h,path
    extraction=read(ROOT/policy['source_extraction']);methods=list(source['methods'])
    features={};feature_hashes={}
    for receipt in extraction['records']:
        if receipt['role'] not in ['gt',*methods]:continue
        assert sha(receipt['path'])==receipt['sha256']
        with np.load(receipt['path']) as data:features[receipt['id'],receipt['role']]=data['motion'].copy()
        feature_hashes[receipt['path']]=receipt['sha256']
    values={};variance_by_coefficient={}
    for role in ['gt',*methods]:
        clips=[features[c['id'],role][c['listener_frame_indices']] for c in source['coverage']]
        a=np.concatenate(clips)
        assert a.shape==(source['coverage_summary']['shared_scored_frames'],56) and np.isfinite(a).all()
        by_group={g:np.var(a[:,start:end],axis=0,ddof=0) for g,(start,end) in policy['groups'].items()}
        values[role]={g:float(v.mean()) for g,v in by_group.items()}
        variance_by_coefficient[role]={g:v.tolist() for g,v in by_group.items()}
    result=copy.deepcopy(source)
    result.update(schema='vico-emoca-per-coefficient-variance-readout-v1',
        source_result_sha256=sha(source_path),variance_policy=policy,
        publication_label='DIM released metrics with explicitly modified per-coefficient mean Var',
        released_variance_reference=dict(gt=copy.deepcopy(source['gt_metrics']['Var']),
            methods={m:copy.deepcopy(source['methods'][m]['metrics']['Var']) for m in methods}),
        variance_per_coefficient=variance_by_coefficient,
        author_variance_definition_confirmed=False)
    for method in methods:
        result['methods'][method]['metrics']['Var']=values[method]
        result['methods'][method]['metrics']['Var_GT']=values['gt']
        result['methods'][method]['aggregation']='FD/P-FD/MSE equal clips; SID GT-fitted global KMeans40/20; Var mean of per-coefficient pooled-frame population variances; rPCC pooled flattened correlation'
        for metric,groups in source['methods'][method]['metrics'].items():
            if metric not in ['Var','Var_GT']:assert result['methods'][method]['metrics'][metric]==groups
    result['gt_metrics']['Var']=values['gt'];result['gt_metrics']['Var_GT']=values['gt']
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=dict(config_sha256=sha(config),implementation_sha256=sha(__file__),source_sha256=sha(source_path))
    if (output/'COMPLETE.json').exists():
        complete=read(output/'COMPLETE.json');assert complete['identity']==identity
        for path,h in complete['files'].items():assert sha(path)==h,path
        print('Existing readout verified');return
    save(output/'RESULTS.json',result)
    lines=['# ViCo：逐系数方差取平均','',
        '用户指定采用第二种Var定义。对同一有效帧集合，将片段拼接后，分别计算每个系数的总体方差，再对50维exp／6维pose求平均；不先对每个视频求平均。',
        '全部43条、16,238个共同有效listener帧，与原结果完全相同。FD、P-FD、MSE、SID、rPCC、采样与其他设置保持原值。','',
        '| 数据／模型 | 新Var Exp | 新Var Pose | 原发布代码Var Exp | 原发布代码Var Pose |',
        '|---|---:|---:|---:|---:|']
    old_values={'gt':source['gt_metrics']['Var'],**{m:source['methods'][m]['metrics']['Var'] for m in methods}}
    for role in ['gt',*methods]:
        lines.append(f"| {role} | {values[role]['exp']:.6f} | {values[role]['pose']:.6f} | {old_values[role]['exp']:.6f} | {old_values[role]['pose']:.6f} |")
    lines+=['','公式：`mean_j Var_t(x[t,j])`，`ddof=0`。GT也使用完全相同的公式。',
        '主结果采用本次指定定义；原公式成绩保留为来源记录。这个选择尚不能证明DIM作者实际采用同一公式，论文引用表须注明该定义。',
        '原始封存成绩未修改；本次没有新生成、重提取或GPU分配。十次择优队列当前的暂停状态未改变。','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    (output/'variance.csv').write_text('source,var_exp,var_pose\n'+'\n'.join(f"{r},{v['exp']},{v['pose']}" for r,v in values.items())+'\n')
    save(output/'AUDIT.json',dict(passed=True,pairs=43,scored_frames=source['coverage_summary']['shared_scored_frames'],
        verified_feature_arrays=len(features),unchanged_metrics=['FD','P_FD','MSE','SID','rPCC'],
        feature_sha256=feature_hashes,original_results_unchanged=True,config=policy))
    save(output/'COMPLETE.json',dict(identity=identity,files={str(p):sha(p) for p in output.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
    print(json.dumps(values,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/p2_vico_dim_variance_per_coefficient_v1.json')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/p2_vico_dim_var_per_coefficient_20261108_v2')
    args=parser.parse_args();run(args.config,args.output)
