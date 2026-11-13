"""Score frozen EMOCA arrays and export literature / local rows with provenance."""
import csv
from pathlib import Path
import time

import numpy as np

from vico_literature_protocol import ROOT,WORK,OLD,read,save,sha
from vico_literature_metrics import evaluate,parity


def score():
    protocol=read(WORK/'protocol.json')
    complete=read(WORK/'EXTRACTION_COMPLETE.json');assert complete['success'] and complete['tasks']==172
    assert read(WORK/'ENCODER_PARITY.json')['passed'] and read(WORK/'MEDIA_PARITY.json')['passed']
    expected={(r['id'],r['role']):r for r in protocol['tasks']}
    features={};records={}
    for r in complete['records']:
        key=r['id'],r['role'];assert key in expected and key not in features
        assert r['video_sha256']==expected[key]['video_sha256']
        assert sha(r['path'])==r['sha256']
        features[key]=np.load(r['path'])['motion'];records[key]=r
        assert features[key].shape==(expected[key]['expected_frames'],56)
    assert set(features)==set(expected)
    ids=protocol['ids'];methods=protocol['selected_methods'];masks={};coverage=[]
    for uid in ids:
        # Match the released tester's t0 speaker -> t1 listener convention.
        mask=np.isfinite(features[uid,'speaker'][:-1]).all(1)
        for role in ['gt',*methods]:mask &= np.isfinite(features[uid,role][1:]).all(1)
        assert int(mask.sum())>=2,(uid,int(mask.sum()))
        masks[uid]=mask
        coverage.append(dict(id=uid,total_frames=len(mask),scored_frames=int(mask.sum()),
            listener_frame_indices=(np.where(mask)[0]+1).tolist(),speaker_frame_indices=np.where(mask)[0].tolist()))
    gt=[features[u,'gt'][1:][masks[u]] for u in ids]
    speaker=[features[u,'speaker'][:-1][masks[u]] for u in ids]
    results={}
    for method in methods:
        pred=[features[u,method][1:][masks[u]] for u in ids]
        result=evaluate(gt,pred,speaker)
        for uid,row in zip(ids,result['per_clip']):row.update(id=uid,scored_frames=int(masks[uid].sum()))
        results[method]=result
    actual_parity=parity(gt,[features[u,methods[0]][1:][masks[u]] for u in ids],speaker)
    actual_parity['input_space']='actual EMOCA_v2_lr_mse_20 arrays, all43 pairs'
    save(WORK/'EMOCA_METRIC_PARITY.json',actual_parity)
    self_result=evaluate(gt,gt,speaker)
    for name in ['FD','P_FD','MSE','rPCC']:
        assert max(abs(v) for v in self_result['metrics'][name].values() if v is not None)<1e-5,(name,self_result['metrics'][name])
    save(WORK/'GT_SELF_CHECK.json',dict(passed=True,metrics=self_result['metrics'],pairs=43))
    published=next(r for r in read(WORK/'LITERATURE_ROWS.json')['rows'] if r['method']=='GT / DIM')
    anchor=dict(published_dim_gt=published['metrics'],measured_gt=self_result['metrics'],
        interpretation='Diagnostic only: rounded published anchor and unknown effective IDs/crop can differ; no rescaling to force agreement')
    save(WORK/'GT_ANCHOR.json',anchor)
    # Declared before EMOCA scores: sensitivity to the visible loader's keys and
    # video length filter. Author HuBERT lengths are unavailable; therefore this
    # projection must never be described as the author's exact retained test.
    mapping=read(WORK/'TEST_MANIFEST.json')['pairs']
    projection=[r['released_mapping_listener'] for r in mapping
        if 5<=len(features[r['released_mapping_listener'],'gt'])<=1024]
    projection_result={}
    for method in methods:
        projected_gt=[features[u,'gt'][1:][masks[u]] for u in projection]
        projected_pred=[features[u,method][1:][masks[u]] for u in projection]
        projected_speaker=[features[u,'speaker'][:-1][masks[u]] for u in projection]
        projection_result[method]=evaluate(projected_gt,projected_pred,projected_speaker)
    save(WORK/'LOADER_SCOPE_DIAGNOSTIC.json',dict(
        name='Metadata audio-key projection plus video length5..1024 only',
        source_rows=len(projection),unique_listener_ids=len(set(projection)),ids_with_repetitions=projection,
        author_hubert_lengths_available=False,exact_author_test=False,primary_scope_unchanged=True,
        methods=projection_result,
        reason='Released preprocessing can overwrite listener identity per audio key; repeated loader rows are retained in this diagnostic'))
    missing={}
    for role in ['speaker','gt',*methods]:
        role_rows=[records[u,role] for u in ids]
        missing[role]=dict(total_frames=sum(r['frames'] for r in role_rows),
            missing_frames=sum(len(r['missing_frames']) for r in role_rows),
            affected_pairs=sum(bool(r['missing_frames']) for r in role_rows))
    coverage_summary=dict(pairs_retained=43,roles=missing,
        candidate_frames_after_t0=sum(c['total_frames'] for c in coverage),
        shared_scored_frames=sum(c['scored_frames'] for c in coverage),
        shared_excluded_frames=sum(c['total_frames']-c['scored_frames'] for c in coverage),
        policy='Failures are counted here; motion metrics use the same valid frame mask for both models; no pair is silently dropped')
    save(WORK/'COVERAGE.json',coverage_summary)
    result=dict(schema='vico-emoca-dim-released-metrics-v1',at=time.time(),pairs=43,
        methods=results,gt_metrics=self_result['metrics'],coverage=coverage,coverage_summary=coverage_summary,source_protocol_sha256=sha(WORK/'protocol.json'),
        extraction_protocol_sha256=sha(WORK/'EXTRACTION_PROTOCOL.json'),
        encoder_checkpoint_sha256=read(WORK/'ENCODER.json')['checkpoint_sha256'],
        readout='RGB re-extraction into EMOCA, not native driver output',
        sampling='one fixed already-sealed output per pair; no GT-based candidate selection',
        publication_label='Measured with named EMOCA-v2 model and DIM released metric implementation; disclosed deviations',
        strict_published_table_reproduction=False,cross_source_superiority_claim_allowed=False,
        literature=read(WORK/'LITERATURE_ROWS.json')['rows'])
    save(WORK/'RESULTS.json',result)
    return result


def export(result):
    folder=WORK/'paper_tables';folder.mkdir(exist_ok=True)
    groups=['exp','pose'];metrics=['FD','P_FD','MSE','SID','Var','rPCC']
    with (folder/'all_metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['origin','method','metric','group','value','result_source'])
        for row in result['literature']:
            for k in metrics:
                for g in groups:w.writerow(['literature',row['method'],k,g,row['metrics'][k][g],row['original_result_source']])
        for name,row in result['methods'].items():
            for k in metrics:
                for g in groups:w.writerow(['local_emoca_rgb',name,k,g,row['metrics'][k][g],'this work; fixed43 / one output'])
    with (folder/'ours_per_clip.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','uuid','frames','metric','group','value'])
        for name,row in result['methods'].items():
            for r in row['per_clip']:
                for k in ['FD','P_FD','MSE']:
                    for g in groups:w.writerow([name,r['id'],r['scored_frames'],k,g,r[k][g]])
    with (folder/'gt_anchor.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['origin','metric','group','value'])
        published_gt=next(r for r in result['literature'] if r['method']=='GT / DIM')
        for k in ['SID','Var']:
            for g in groups:
                w.writerow(['DIM published GT',k,g,published_gt['metrics'][k][g]])
                w.writerow(['Our EMOCA extraction of GT',k,g,result['gt_metrics'][k][g]])
    with (folder/'coverage.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['role','total_frames','missing_frames','affected_pairs'])
        for role,row in result['coverage_summary']['roles'].items():
            w.writerow([role,row['total_frames'],row['missing_frames'],row['affected_pairs']])
    lines=['# ViCo：EMOCA与DIM发布指标的对齐实测','',
        '**评价组成已对齐，发表时的完整执行协议仍有未公开项。** 本地结果不能标为DIM原表的严格复现，也不据此自动计算跨来源提升率。','',
        '[逐项协议对照](ALIGNMENT.md) · [完整JSON](RESULTS.json) · [全部指标CSV](paper_tables/all_metrics.csv) · [每条FD/P-FD/MSE](paper_tables/ours_per_clip.csv)','',
        '| 来源 | 方法 | FD Exp ↓ | FD Pose ↓ | MSE Exp ↓ | MSE Pose ↓ | SID Exp | SID Pose | Var Exp | Var Pose | rPCC Exp ↓ | rPCC Pose ↓ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def display(v,digits):return 'N/A' if v is None else f'{v:.{digits}f}'
    selected=['FD','MSE','SID','Var','rPCC']
    for r in result['literature']:
        vals=[display(r['metrics'][k][g],2) for k in selected for g in groups]
        lines.append('| 引用：'+r['original_result_source']+' | '+r['method']+' | '+' | '.join(vals)+' |')
    for name,row in result['methods'].items():
        vals=[display(row['metrics'][k][g],5) for k in selected for g in groups]
        lines.append('| 本地：RGB→EMOCA | '+name+' | '+' | '.join(vals)+' |')
    gt_values=['N/A' if k in ['FD','MSE','rPCC'] else display(result['gt_metrics'][k][g],5) for k in selected for g in groups]
    lines.append('| 本地：GT提取锚点 | GT / our EMOCA | '+' | '.join(gt_values)+' |')
    c=result['coverage_summary']
    lines += ['','## 人脸检出与实际评分覆盖','',
        f"全部{c['pairs_retained']}条pair均保留；去除首帧后候选{c['candidate_frames_after_t0']}帧，共同评分{c['shared_scored_frames']}帧，共同掩码排除{c['shared_excluded_frames']}帧。",'',
        '| 流 | 原始帧数 | 未检出人脸帧数 | 涉及pair数 |','|---|---:|---:|---:|']
    for role,r in c['roles'].items():lines.append(f"| {role} | {r['total_frames']} | {r['missing_frames']} | {r['affected_pairs']} |")
    lines += ['','缺失帧不赋予零误差、不插值；这些帧没有进入动作距离，失败数单独报告。原生系数基线不存在同样的RGB检测步骤，比较时需要保留该差异。',
        '[覆盖CSV](paper_tables/coverage.csv) · [逐帧有效范围JSON](RESULTS.json)','']
    if (WORK/'diagnostics/missing_face_first_case.png').exists():
        lines += ['[首个未检出案例的原始帧截图](diagnostics/missing_face_first_case.png) · [对应视频与帧号](diagnostics/MISSING_FACE_FIRST_CASE.json)','']
    lines+=['','## 本次执行及差异','',
        '- 使用DIM预处理明确指定的EMOCA_v2_lr_mse_20；pose6+exp50，原始系数，无统计重标定。',
        '- 固定43个listener UUID，不含OOD；两组预测共用同一有效帧掩码。listener去首帧，与speaker前T−1帧对应，沿发布tester。',
        '- FD/P-FD/MSE逐片段平均；SID/Var/rPCC按照固定版本DIM代码执行。P-FD见完整CSV。',
        '- 原生动作系数与RGB反提取有差异；我们为后者，不伪称原生输出。',
        '- 我们固定单次样本；DIM公开tester的十次GT择优是否用于发表主表未确认，未对我们进行GT择优。',
        '- 发布加载器的重复audio-key覆盖、长度筛选及常量speaker问题尚不能代表论文实际运行。本次保留真实speaker和完整UUID，不把猜测写成已确认。',
        '- GT自比较、GT统计锚点和逐指标作者代码对照均单独保存；与发表GT不一致时如实报告，不反推缩放系数。','',
        '[公开加载器范围敏感性诊断](LOADER_SCOPE_DIAGNOSTIC.json)：另算audio-key覆盖和视频5–1024帧限制的投影；没有作者HuBERT长度，不能称其为作者的实际test。本表仍保留预先固定的完整43条。','',
        '[GT统计对照CSV](paper_tables/gt_anchor.csv)：同时保留DIM发表的GT数值和本次GT提取统计，不通过缩放使其相等。','',
        '## 原始出处','',
        '[DIM Table1及补充A](../../writing/17_DIM_ECCV2024.pdf)；[INFP Table3、§5.2.2](../../writing/35_INFP_CVPR2025.pdf)；[Avatar Forcing Table4](../../writing/43_Avatar_Forcing_CVPR2026.pdf)。',
        'RLHG/L2L/DIM旧行源于DIM；INFP原版与Avatar Forcing重实现的INFP*分行，未报告的格子为N/A。','']
    lines += ['[正式发表版本逐行／逐格核验](LITERATURE_CELL_PROVENANCE.json) · [结果来源BibTeX](paper_tables/result_sources.bib)。',
        'DIM的正式ECCV版本Table1在PDF第10页，本地预印本在第11页；引用行数值一致。INFP Table3在第8页，Avatar Forcing Table4在第6页。引用结果来源时，还应同时引用对应方法原文。','']
    published_gt=next(r for r in result['literature'] if r['method']=='GT / DIM')['metrics']
    gt=result['gt_metrics']
    lines += ['## 对齐程度的实测结论','',
        f"GT表情Var：本次{gt['Var']['exp']:.5f}，论文{published_gt['Var']['exp']:.2f}；GT姿态Var：本次{gt['Var']['pose']:.5f}，论文{published_gt['Var']['pose']:.2f}。",
        'GT统计并未完全对齐；文献行可以按来源引用，我们的行应标为“按DIM公开EMOCA及指标设置实测，存在已披露差异”。当前证据不足以称为完全相同发表协议下的严格排名，不能只凭数值计算跨来源提升率。',
        '这个差异来自GT侧，无法通过更换我们的生成权重来解决；也不通过调尺度、改裁剪或按分数选择子集强行匹配。作者实际有效清单、GT特征或完整预处理配置是进一步定位所需的证据。','']
    (WORK/'RESULTS.md').write_text('\n'.join(lines))
    # Two explicit blocks; no best-value highlighting or pooled ranking.
    from export_vico43_sft2752_paper_tables import table
    headers=['Method','FD Exp','FD Pose','MSE Exp','MSE Pose','SID Exp','SID Pose','Var Exp','Var Pose','rPCC Exp','rPCC Pose']
    columns=['FD','MSE','SID','Var','rPCC']
    short_names={'ELP reimplementation':'ELP (DIM reimpl.)','L2L + CANDOR':'L2L + CANDOR',
        'DIM + CANDOR':'DIM + CANDOR','INFP original':'INFP',
        'INFP* / AvatarForcing reimplementation':'INFP* (AF reimpl.)','GT / DIM':'GT (DIM)'}
    literature_rows=[[short_names.get(r['method'],r['method']),*[display(r['metrics'][k][g],2) for k in columns for g in groups]] for r in result['literature']]
    labels={'joint':'Ours / base Qwen','joint_sft2752':'Ours / SFT2752'}
    ours_rows=[[labels[m],*[display(r['metrics'][k][g],5) for k in columns for g in groups]] for m,r in result['methods'].items()]
    ours_rows.append(['GT / our EMOCA',*['N/A' if k in ['FD','MSE','rPCC'] else display(result['gt_metrics'][k][g],5) for k in columns for g in groups]])
    (folder/'literature_reference.tex').write_text(table('Published ViCo reference values. Original result sources: DIM Table1, INFP Table3 and Avatar Forcing Table4. '
        'RLHG/L2L/DIM are inherited results; original INFP and Avatar Forcing INFP* are distinct. '
        'Unreported cells remain N/A. This block is not locally rerun.', 'tab:vico-literature-reference',headers,literature_rows,True))
    c=result['coverage_summary']
    (folder/'ours_emoca.tex').write_text(table('Our fixed43 ViCo outputs evaluated with EMOCA-v2-lr-mse-20 and the DIM released metric implementation. '
        'RGB re-extraction, one fixed sample, actual speaker motion and disclosed valid-frame masks. '
        f"All43 pairs retained; {c['shared_scored_frames']}/{c['candidate_frames_after_t0']} candidate frames evaluated after excluding listener t0; {c['shared_excluded_frames']} frames excluded by the common face-validity mask. "
        'Published effective test scope and candidate-selection settings are unconfirmed; this block is not strict original-table reproduction and supports no cross-source ranking.',
        'tab:vico-ours-emoca',headers,ours_rows,True))
    for stem in ['literature_reference','ours_emoca']:
        path=folder/(stem+'.tex');text=path.read_text()
        text=text.replace(r'% Requires \usepackage{booktabs}',r'% Requires \usepackage{booktabs,graphicx}')
        text=text.replace(r'\begin{tabular}',r'\resizebox{\textwidth}{!}{%'+'\n'+r'\begin{tabular}',1)
        text=text.replace(r'\end{tabular}',r'\end{tabular}'+'\n}',1)
        path.write_text(text)
    save(WORK/'EXPORTS.json',dict(files={str(p):sha(p) for p in folder.iterdir() if p.suffix in ['.csv','.tex','.bib']},
        literature_and_local_blocks_separate=True,unreported_cells_preserved=True))


def verify_existing():
    """Recompute all metrics from sealed arrays without changing any artifact."""
    for path,h in read(WORK/'COMPLETE.json')['files'].items():assert sha(path)==h,path
    result=read(WORK/'RESULTS.json');protocol=read(WORK/'protocol.json');features={}
    for r in read(WORK/'EXTRACTION_COMPLETE.json')['records']:
        assert sha(r['path'])==r['sha256']
        features[r['id'],r['role']]=np.load(r['path'])['motion']
    gt=[];speaker=[];pred={m:[] for m in protocol['selected_methods']}
    for uid,c in zip(protocol['ids'],result['coverage']):
        assert c['id']==uid
        mask=np.isfinite(features[uid,'speaker'][:-1]).all(1)
        for role in ['gt',*protocol['selected_methods']]:mask &= np.isfinite(features[uid,role][1:]).all(1)
        assert c['listener_frame_indices']==(np.where(mask)[0]+1).tolist()
        assert c['speaker_frame_indices']==np.where(mask)[0].tolist()
        gt.append(features[uid,'gt'][1:][mask]);speaker.append(features[uid,'speaker'][:-1][mask])
        for method in pred:pred[method].append(features[uid,method][1:][mask])
    errors=[]
    for method,arrays in pred.items():
        actual=evaluate(gt,arrays,speaker);expected=result['methods'][method]
        for key,groups in expected['metrics'].items():
            for group,value in groups.items():
                if value is None:assert actual['metrics'][key][group] is None
                else:errors.append(abs(actual['metrics'][key][group]-value))
        for a,b in zip(actual['per_clip'],expected['per_clip']):
            for key in ['FD','P_FD','MSE']:
                for group in ['exp','pose']:errors.append(abs(a[key][group]-b[key][group]))
    assert max(errors)<1e-10,max(errors)
    print('Sealed full43 numerical recomputation passed:',len(errors),'scalars; max_abs=',max(errors))


if __name__=='__main__':
    sys_path=ROOT/'scripts'
    import sys
    sys.path.insert(0,str(sys_path))
    if (WORK/'COMPLETE.json').exists():verify_existing()
    else:export(score())
