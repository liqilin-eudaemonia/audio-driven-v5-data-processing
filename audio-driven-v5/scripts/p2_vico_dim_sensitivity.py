"""Score frozen ViCo outputs under predeclared, evidence-based DIM alternatives.

CPU only. The old EMOCA features/results and vendor sources remain immutable.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_literature_metrics import evaluate

METRICS=['FD','P_FD','MSE','SID','Var','rPCC']
GROUPS={'exp':slice(6,56),'pose':slice(0,6)}


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def entropy(labels,k):
    counts=np.bincount(labels,minlength=k).astype(np.float64)
    if counts.sum()==0:raise ValueError('No observations')
    hist=counts/counts.sum()
    return float(-np.sum(hist*np.log2(hist+1e-6)))


def variance_statistics(clips):
    if not clips:raise ValueError('No clips')
    a=np.concatenate(clips)
    if not np.isfinite(a).all():raise ValueError('Invalid variance inputs')
    return dict(pooled_flatten=float(np.var(a)),
        pooled_per_coefficient_mean=float(np.var(a,axis=0).mean()),
        equal_clip_flatten=float(np.mean([np.var(x) for x in clips])),
        equal_clip_temporal_per_coefficient_mean=float(np.mean([np.var(x,axis=0).mean() for x in clips])))


def build_case(features,ids,methods,offset):
    """Keep listener t>=1 fixed; change speaker timestamps only, or sample scope."""
    if offset not in (0,1):raise ValueError('Unsupported offset')
    gt=[];speaker=[];pred={m:[] for m in methods};coverage=[]
    for uid in ids:
        arrays={r:features[uid,r] for r in ['gt','speaker',*methods]}
        if len({a.shape for a in arrays.values()})!=1:raise ValueError('Unequal stream shape')
        size=len(arrays['gt'])
        gi=np.arange(1,size);si=gi-offset
        valid=np.isfinite(arrays['speaker'][si]).all(1)
        for r in ['gt',*methods]:valid &= np.isfinite(arrays[r][gi]).all(1)
        gi=gi[valid];si=si[valid]
        if len(gi)<2:raise ValueError('Too few complete frames: '+uid)
        gt.append(arrays['gt'][gi]);speaker.append(arrays['speaker'][si])
        for m in methods:pred[m].append(arrays[m][gi])
        coverage.append(dict(id=uid,listener_frames=gi.tolist(),speaker_frames=si.tolist(),
            scored_frames=len(gi),excluded_frames=size-1-len(gi)))
    return gt,speaker,pred,coverage


def alternative_statistics(gt,pred):
    """Fit each cluster set once to GT, then use identical clusters for both methods."""
    output={name:{} for name in ['gt',*pred]}
    for group,sl in GROUPS.items():
        k=40 if group=='exp' else 20
        model=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(np.concatenate(gt)[:,sl])
        for name,clips in [('gt',gt),*pred.items()]:
            selected=[a[:,sl] for a in clips]
            label_clips=[model.predict(a) for a in selected]
            output[name][group]=dict(Var=variance_statistics(selected),
                SID=dict(global_histogram=entropy(np.concatenate(label_clips),k),
                    equal_clip_entropy=float(np.mean([entropy(labels,k) for labels in label_clips]))))
    return output


def compare_metrics(actual,expected):
    errors=[]
    for metric in METRICS:
        for group in GROUPS:
            a,b=actual[metric][group],expected[metric][group]
            if a is None or b is None:
                assert a is b
            else:errors.append(abs(a-b))
    if errors and max(errors)>1e-10:raise ValueError('Existing metric reproduction mismatch: '+str(max(errors)))
    return errors


def execute(config,output):
    config=Path(config).resolve();out=Path(output).resolve();policy=read(config)
    if policy['schema']!='vico-dim-inferred-protocol-v1':raise ValueError('Unexpected protocol')
    if policy['primary_case']!='official43_released_offset':raise ValueError('Primary choice must be explicit and frozen')
    source=ROOT/policy['source_work']
    if sha(source/'COMPLETE.json')!=policy['source_complete_sha256']:raise ValueError('Wrong source run')
    identity=dict(config_sha256=sha(config),implementation_sha256=sha(__file__),
        source_complete_sha256=sha(source/'COMPLETE.json'),
        scorer_sha256=sha(ROOT/'compare/deployment/vico_literature_metrics.py'))
    if (out/'COMPLETE.json').exists():
        complete=read(out/'COMPLETE.json')
        if complete['identity']!=identity:raise ValueError('Implementation/config changed; use a new directory')
        for path,h in complete['files'].items():
            if sha(path)!=h:raise ValueError('Output changed: '+path)
        print('Existing sensitivity run verified; no results rewritten');return
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists() and read(out/'protocol.json')['identity']!=identity:
        raise ValueError('Source drift: use a new output directory')
    save(out/'protocol.json',dict(identity=identity,policy=policy,created_at=time.time()))
    save(out/'status.json',dict(stage='checking_inputs',pid=os.getpid()))
    for path,h in read(source/'COMPLETE.json')['files'].items():
        if sha(path)!=h:raise ValueError('Old sealed artifact changed: '+path)
    original=read(source/'RESULTS.json');p=read(source/'protocol.json')
    methods=policy['methods'];assert methods==p['selected_methods']
    features={}
    for r in read(source/'EXTRACTION_COMPLETE.json')['records']:
        if sha(r['path'])!=r['sha256']:raise ValueError('Feature changed: '+r['path'])
        key=(r['id'],r['role'])
        if key in features:raise ValueError('Duplicate feature')
        with np.load(r['path']) as data:features[key]=data['motion'].copy()
        assert features[key].shape==(r['frames'],56)
    projection=read(source/'LOADER_SCOPE_DIAGNOSTIC.json')
    cases={};errors=[]
    for case in policy['cases']:
        cid=case['id']
        ids=p['ids'] if case['scope']=='official43' else projection['ids_with_repetitions']
        gt,speaker,pred,coverage=build_case(features,ids,methods,case['speaker_offset_frames'])
        values={m:evaluate(gt,a,speaker) for m,a in pred.items()}
        if cid==policy['primary_case']:
            for m in methods:errors+=compare_metrics(values[m]['metrics'],original['methods'][m]['metrics'])
        if case['scope']=='video_only_loader_projection':
            for m in methods:errors+=compare_metrics(values[m]['metrics'],projection['methods'][m]['metrics'])
        stats=alternative_statistics(gt,pred)
        for m in methods:
            for group in GROUPS:
                assert abs(stats[m][group]['SID']['global_histogram']-values[m]['metrics']['SID'][group])<1e-10
                assert stats[m][group]['Var']['pooled_flatten']==values[m]['metrics']['Var'][group]
        cases[cid]=dict(definition=case,rows=len(ids),unique_listener_ids=len(set(ids)),
            scored_frames=sum(c['scored_frames'] for c in coverage),coverage=coverage,
            methods=values,alternative_statistics=stats)
        save(out/'status.json',dict(stage='scoring',completed_cases=len(cases),total_cases=len(policy['cases']),case=cid))
        print('Scored',cid,'rows',len(ids),'frames',cases[cid]['scored_frames'],flush=True)
    base=cases[policy['primary_case']]
    # The offset experiment changes only the speaker side, keeping all listener observations.
    same=cases['official43_same_timestamp']
    for m in methods:
        for metric in ['FD','MSE','SID','Var']:
            assert base['methods'][m]['metrics'][metric]==same['methods'][m]['metrics'][metric]
    result=dict(identity=identity,primary_case=policy['primary_case'],cases=cases,
        interpretation='Evidence-based reconstruction and sensitivity, not proof of private published execution',
        selection_by_generated_scores=False,strict_published_table_equivalence=False,
        sampling=policy['sampling'],author_contact_required=False)
    save(out/'RESULTS.json',result)
    export(out,result)
    save(out/'FINAL_AUDIT.json',dict(passed=True,feature_streams=len(features),cases=len(cases),methods=methods,
        reproduced_existing_metric_scalars=len(errors),max_abs=max(errors),
        offset_changes_only_speaker=True,original_sealed_results_unchanged=True,
        no_new_gpu_execution=True,config_selected_by_generated_scores=False))
    save(out/'status.json',dict(stage='complete',cases=len(cases),strict_published_table_equivalence=False))
    files=[p for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json']
    save(out/'COMPLETE.json',dict(identity=identity,success=True,files={str(p):sha(p) for p in files}))


def export(out,result):
    primary=result['cases'][result['primary_case']]
    with (out/'metrics.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['case','method','rows','unique_listeners','frames','metric','group','value','delta_from_primary'])
        for cid,case in result['cases'].items():
            for method,row in case['methods'].items():
                for metric in METRICS:
                    for group in GROUPS:
                        value=row['metrics'][metric][group];base=primary['methods'][method]['metrics'][metric][group]
                        w.writerow([cid,method,case['rows'],case['unique_listener_ids'],case['scored_frames'],metric,group,value,
                            value-base if value is not None and base is not None else ''])
    with (out/'alternative_statistics.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['case','source','group','metric','definition','value'])
        for cid,case in result['cases'].items():
            for name,groups in case['alternative_statistics'].items():
                for group,metrics in groups.items():
                    for metric,definitions in metrics.items():
                        for definition,value in definitions.items():w.writerow([cid,name,group,metric,definition,value])
    lines=['# ViCo / DIM配置敏感性实测','',
        '主设置预先固定为正式test43、原始EMOCA、真实speaker及发布tester一帧偏移，指标沿发布实现。未按生成成绩选择设置。',
        '两版均为已封存的seed137单次输出，尚不是10次择优结果；本次仅CPU评分。','',
        '| 设置 | 方法 | FD Exp | FD Pose | P-FD Exp | P-FD Pose | rPCC Exp | rPCC Pose |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for cid,case in result['cases'].items():
        for method,row in case['methods'].items():
            vals=[row['metrics'][metric][group] for metric in ['FD','P_FD','rPCC'] for group in GROUPS]
            lines.append('| '+case['definition']['label']+' | '+method+' | '+' | '.join('N/A' if v is None else f'{v:.6f}' for v in vals)+' |')
    lines+=['','全部值、B−主设置差值与覆盖见[metrics.csv](metrics.csv)，各Var／SID定义的GT与两版输出见[alternative_statistics.csv](alternative_statistics.csv)。','',
        '## 推荐设置','',
        '- 完整43个listener UUID作为主范围。41记录／39个listener的加载器投影仅为已披露敏感性结果，缺作者HuBERT长度，不能称其为真实原表清单。',
        '- FD／P-FD／MSE、SID、Var、rPCC主结果沿固定DIM公开实现。SID用全局汇总；Var同时给逐系数时序定义作为敏感性表，不因接近论文GT而替换主公式。',
        '- 真实speaker用于评分，保留tester一帧偏移，并展示同时间戳的影响。',
        '- 保留原始尺度、指定EMOCA模型、共同有效帧掩码和全部失败覆盖；原生系数与RGB反提取差异随表说明。',
        '- 单次结果按单次报告。发布tester的10次GT择优是另一种采样协议，若补测必须另列，不能把当前单次成绩标为该协议。',
        '- 作者联系不作为推进前提；上述设置称为“依据论文及公开实现重建”，不称作者私有运行已确认。','']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'configs/p2_vico_dim_inferred_v1.json'))
    parser.add_argument('--output',default=str(ROOT/'experiments/p2_vico_dim_inferred_protocol_20261108/sensitivity_v1'))
    args=parser.parse_args();execute(args.config,args.output)
