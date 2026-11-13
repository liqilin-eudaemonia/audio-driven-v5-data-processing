"""Predeclared whole DIM metric implementation on the same SPECTRE features."""
import argparse
import time
from pathlib import Path
import sys

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_avatarforcing_table4_generate import read,save,sha


def prepare():
    dest=WORK/'DIM_STYLE_PROTOCOL.json'
    sources=[Path(__file__),ROOT/'compare/deployment/vico_literature_metrics.py',
        ROOT/'compare/Dyadic-Interaction-Modeling/code/metrics/eval_utils.py',
        ROOT/'compare/Dyadic-Interaction-Modeling/code/mymetrics.py']
    code={str(p):sha(p) for p in sources}
    if dest.exists():
        p=read(dest);assert p['code']==code
        return p
    assert not (WORK/'OFFICIAL_RESULTS.json').exists(), 'Declare before seeing full AF results'
    p=dict(schema='whole-published-DIM-math-SPECTRE-v1',created_unix=time.time(),code=code,
        generation_protocol_sha256=sha(WORK/'NATIVE_GENERATION_PROTOCOL.json'),
        feature_protocol_sha256=sha(WORK/'OFFICIAL_SPECTRE_PROTOCOL.json'),
        recipe=dict(FD='original equal-clip Frechet',rPCC='original pooled flattened residual correlation',
                    SID='original K40exp/K20pose, GT fit and pooled prediction entropy',
                    Var='original pooled flattened variance'),
        motivation=['AF Table4 explicitly inherits three DIM rows; use its whole released metric implementation as one coherent candidate',
                    'AF INFP* PoseSID3.20 exceeds log2(9), so D.5 K9 is not literally consistent with every Table4 own-run row'],
        feature_extractor='same raw SPECTRE pose6+exp50, same43 and timestamps; not EMOCA and not a DIM Table1 reproduction',
        no_target_based_cellwise_formula_selection=True,
        primary_AF_text_reconstruction_unchanged=True,
        author_AF_private_code_confirmed=False)
    save(dest,p);return p


def score():
    import numpy as np
    from vico_literature_metrics import evaluate,parity
    p=prepare()
    complete=read(WORK/'OFFICIAL_EXTRACTION_COMPLETE.json')
    assert complete['passed'] and complete['streams']==129
    generation=read(WORK/'NATIVE_GENERATION_PROTOCOL.json')
    ids=[s['id'] for s in generation['samples']]
    data={}
    for r in complete['records']:
        assert sha(r['path'])==r['sha256']
        with np.load(r['path']) as values:a=values['motion'].copy()
        assert a.shape==(r['frames'],56) and np.isfinite(a).all()
        data[r['id'],r['role']]=a
    gt=[data[u,'gt'] for u in ids];pred=[data[u,'avatarforcing'] for u in ids];sp=[data[u,'speaker'] for u in ids]
    proof=parity(gt,pred,sp)
    result=evaluate(gt,pred,sp)
    result['units']='Unscaled raw SPECTRE pose6 then exp50; DIM metric mathematics only'
    cells=[]
    for metric in ['FD','rPCC','SID','Var']:
        for group in ['exp','pose']:
            value=result['metrics'][metric][group];target=generation['target'][metric][group]
            cells.append(dict(metric=metric,group=group,measured=value,published=target,
                same_at_2dp=value is not None and f'{value:.2f}'==f'{target:.2f}'))
    matches=sum(c['same_at_2dp'] for c in cells)
    result.update(comparison=cells,matched_cells=matches,total_cells=8,numerical_table_match=matches==8,
        complete_published_math_parity=proof,protocol_sha256=sha(WORK/'DIM_STYLE_PROTOCOL.json'),
        extraction_complete_sha256=sha(WORK/'OFFICIAL_EXTRACTION_COMPLETE.json'),
        author_AF_private_code_confirmed=False,not_DIM_Table1_reproduction=True,
        interpretation='One predeclared complete published-code candidate, not an opportunistic mixture of per-cell formulas')
    save(WORK/'DIM_STYLE_RESULTS.json',result)
    lines=['# 同一SPECTRE系数上的DIM完整公开评分代码','',
        f'43条，8项中{matches}项与AF Table4两位小数一致。主AF文字重建结果仍单独保留，未按逐格接近程度改公式。','',
        '| 指标 | 分组 | 实测 | AF Table4 | 一致 |','|---|---|---:|---:|---|']
    for c in cells:
        value='N/A' if c['measured'] is None else f"{c['measured']:.6f}"
        lines.append(f"| {c['metric']} | {c['group']} | {value} | {c['published']:.2f} | {c['same_at_2dp']} |")
    lines+=['','这是同一43条原始视野生成视频／SPECTRE系数上的DIM完整数学实现。K40／20及拉平方差与AF附录K15／9、逐系数方差描述存在冲突，不能把它称作已取得AF私有评分代码；也不等于完成DIM的EMOCA十二项复现。','']
    (WORK/'DIM_STYLE_RESULTS.md').write_text('\n'.join(lines))
    print('Complete DIM-code candidate:',matches,'/8 AF cells; source-print parity passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','score']);args=parser.parse_args()
    prepare() if args.action=='prepare' else score()
