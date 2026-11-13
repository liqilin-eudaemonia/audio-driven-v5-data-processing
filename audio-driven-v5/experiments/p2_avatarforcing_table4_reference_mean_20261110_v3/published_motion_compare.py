"""Compare two public AF examples and local outputs under one feature frontend."""
import argparse
from pathlib import Path
import sys
import numpy as np

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
OUT=WORK/'published_motion_control_v2'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_avatarforcing_table4_generate import read,save,sha


def prepare():
    p=dict(schema='published-motion-comparison-v1',implementation_sha256=sha(__file__),
        published_feature_contract_sha256=sha(OUT/'PROTOCOL.json'),
        cohorts=['published_AF','local_single_frame_reference_v2','local_mean_reference_v3'],
        same_GT_and_speaker=True,clip_ids=['ClNUxPsUByg_000210_000221.0','Ew2z_sYABE0_000063_000083.0'],
        measurements=['standard Frechet per clip','temporal feature mean Var','flattened Var',
            'mean squared first temporal difference','pooled flattened rPCC against canonical speaker',
            'published versus canonical speaker feature mean squared error and coefficient correlation'],
        no_fitted_time_shift=True,full_interval=True,not_a_Table4_aggregate=True,
        limitations=['Qualitative examples need not be the exact outputs quantitatively evaluated',
            'Published encoding, titles, geometry and approximately2frame speaker delay remain confounds',
            'Speaker same-source control describes extraction/timeline sensitivity, not a correction fit',
            'No checkpoint or seed conclusion is established by a difference alone'])
    path=OUT/'COMPARISON_PROTOCOL.json'
    if path.exists():assert read(path)==p
    else:save(path,p)
    return p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','report']);a=parser.parse_args()
    protocol=prepare()
    if a.action=='prepare':return
    from vico_literature_metrics import load_author
    author=load_author();current=read(WORK/'FINAL_AUDIT.json');assert current['passed_execution_audit']
    prior=WORK.parent/'p2_avatarforcing_table4_framegrid_20261110_v2'
    assert read(prior/'FINAL_AUDIT.json')['passed_execution_audit']
    extracted=read(OUT/'EXTRACTION_COMPLETE.json');assert extracted['passed'] and extracted['streams']==4
    pub={}
    for r in extracted['records']:
        assert sha(r['path'])==r['sha256'] and sha(r['features'])==r['feature_sha256']
        assert r['protocol_sha256']==sha(OUT/'PROTOCOL.json')
        pub[r['id'],r['role']]=np.load(r['features'])['motion']
    def load_local(directory,role,uid):
        path=directory/'official_features'/role/(uid+'.npz');r=read(path.with_suffix('.json'))
        assert sha(path)==r['sha256'];return np.load(path)['motion']
    def corr(a,b):
        return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
    rows=[]
    for uid in protocol['clip_ids']:
        gt=load_local(WORK,'gt',uid);speaker=load_local(WORK,'speaker',uid)
        sources={'published_AF':pub[uid,'avatar'],'local_single_frame_reference_v2':load_local(prior,'avatarforcing',uid),'local_mean_reference_v3':load_local(WORK,'avatarforcing',uid)}
        assert all(x.shape==gt.shape==speaker.shape for x in sources.values())
        record=dict(id=uid,frames=len(gt),scores={},speaker_source_control={})
        for group,sl in [('exp',slice(6,56)),('pose',slice(0,6))]:
            g=gt[:,sl];x=speaker[:,sl];published_speaker=pub[uid,'speaker'][:,sl]
            c1=x-x.mean(0);c2=published_speaker-published_speaker.mean(0)
            den=np.sqrt((c1*c1).sum(0)*(c2*c2).sum(0));valid=den>1e-15
            record['speaker_source_control'][group]=dict(mse=float(np.mean((x-published_speaker)**2)),
                mean_coefficient_correlation=float(np.mean((c1*c2).sum(0)[valid]/den[valid])),valid_coefficients=int(valid.sum()),
                temporal_Var_ratio=float(np.var(published_speaker,axis=0).mean()/np.var(x,axis=0).mean()))
            for name,values in sources.items():
                p=values[:,sl]
                record['scores'].setdefault(name,{})[group]=dict(FD=float(author.calculate_frechet_distance(*author.calculate_activation_statistics(g),*author.calculate_activation_statistics(p))),
                    temporal_feature_mean_Var=float(np.var(p,axis=0).mean()),flattened_Var=float(np.var(p)),
                    first_difference_energy=float(np.mean(np.diff(p,axis=0)**2)),rPCC=abs(corr(g,x)-corr(p,x)))
        rows.append(record)
    save(OUT/'COMPARISON.json',dict(rows=rows,protocol_sha256=sha(OUT/'COMPARISON_PROTOCOL.json'),
        published_source_sha256=sha(OUT/'EXTRACTION_COMPLETE.json'),current_audit_sha256=sha(WORK/'FINAL_AUDIT.json'),
        full_table4_result=read(WORK/'OFFICIAL_RESULTS.json')['comparison'],not_a_Table4_aggregate=True))
    lines=['# 作者两条演示与本地输出：同前端对照','',
        '使用相同的官方SPECTRE前端和完整时间区间，不拟合偏移；两条定性展示不能替代Table4整批成绩。','',
        '| ID | 输出 | 分组 | FD | 时序逐系数方差 | 拉平方差 | 一阶差分能量 | rPCC |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        for name,groups in row['scores'].items():
            for group,m in groups.items():lines.append(f"| {row['id']} | {name} | {group} | {m['FD']:.6f} | {m['temporal_feature_mean_Var']:.6f} | {m['flattened_Var']:.6f} | {m['first_difference_energy']:.6f} | {m['rPCC']:.6f} |")
    lines+=['','同源speaker的系数相关／方差比及全部来源SHA保存在JSON。发布视频压缩、标题、裁剪及约两帧时序偏移仍会影响系数；差异不能单独证明公开权重缺少DPO，或某个随机种子正确。','']
    (OUT/'COMPARISON.md').write_text('\n'.join(lines))
    print('Published/local comparison complete',flush=True)


if __name__=='__main__':main()
