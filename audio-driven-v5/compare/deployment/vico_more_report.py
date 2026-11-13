"""Publish only completed baselines; keep transfer/readout limitations explicit."""
import copy
import csv
import html
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from vico_more_common import ROOT, WORK, DECA, read, save, sha
from vico_more_motion import aggregate

PREVIOUS=ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'
PAGE=ROOT/'data_previews/listen_benchmark_review/vico_more_20261107'
URL='http://localhost:8876/listen_benchmark_review/vico_more_20261107/'
LABELS=dict(joint='Ours / base Qwen',joint_sft2752='Ours / SFT2752',dystream='DyStream adapted',
    avatarforcing='Avatar Forcing adapted',rlhg='RLHG local retraining',listenformer='ListenFormer local retraining')


def comparison():
    old=read(PREVIOUS/'COMPARISON.json')
    live=read(WORK/'status.json') if (WORK/'status.json').exists() else {}
    images=copy.deepcopy(old['images'])
    deca={k:copy.deepcopy(old['deca']['models'][k]) for k in images}
    native={};bfm={};status={};files={str(PREVIOUS/'COMPARISON.json'):sha(PREVIOUS/'COMPARISON.json')}
    for method in ['rlhg','listenformer']:
        out=WORK/method
        if (out/'COMPLETE.json').exists():
            complete=read(out/'COMPLETE.json');assert complete['success']
            for p,h in complete['files'].items():assert sha(p)==h,p
            m=read(out/'MOTION_RESULTS.json')
            images[method]=read(out/'RESULTS.json')['metrics'];deca[method]=m['deca_video']
            native[method]=m['bfm_native'];bfm[method]=m['bfm_video']
            status[method]='complete';files[str(out/'COMPLETE.json')]=sha(out/'COMPLETE.json')
        else:
            markers=['FAILED.json','NOT_SCHEDULED.json','INCOMPLETE.json']
            status[method]=next((read(out/n) for n in markers if (out/n).exists()),'preparing')
            if status[method]=='preparing':
                status[method]=dict(stage=live.get('methods',{}).get(method,'preparing'),
                    training=read(out/'status.json') if (out/'status.json').exists() else None)
    proxy=read(WORK/'dualtalk/METRICS.json')
    rows=[dict(id=r['id'],metrics={k:v for k,v in r.items() if k!='id'}) for r in proxy['records']]
    dual=aggregate(rows)
    dual.update(official_demo_parity='exact',checkpoint='official public',self_audio='zero',
        speaker_motion='DECA transferred to official exp-jaw-neck input',rgb_metrics=None,
        comparability='Exploratory coefficient-transfer proxy; author calls pose[0:3] neck, DECA calls it global head rotation; geometric equivalence unverified',
        formal_common_rgb_table=False,head_distance_claim_allowed=False)
    bfm_video={method:dict(metrics=m['metrics_raw'],coverage=m['coverage_clips'],clips=m['clips'],
        per_clip=[dict(id=r['id'],metrics=r['metrics_raw'],audit=r['timing_audit']) for r in m['per_clip']])
        for method,m in old['bfm'].items()}
    bfm_video.update(bfm)
    paired={}
    from vico_unified_evaluate import cluster_bootstrap
    ours={r['id']:r['metrics'] for r in deca['joint_sft2752']['per_clip']}
    for method in ['rlhg','listenformer']:
        if method not in deca:continue
        other={r['id']:r['metrics'] for r in deca[method]['per_clip']};assert set(ours)==set(other)
        paired[method]={}
        for metric,groups in deca['joint_sft2752']['metrics'].items():
            paired[method][metric]={}
            for group in groups:
                ids=list(ours)
                diffs=[None if ours[u][metric][group] is None or other[u][metric][group] is None
                       else ours[u][metric][group]-other[u][metric][group] for u in ids]
                paired[method][metric][group]=cluster_bootstrap(ids,diffs)
    result=dict(schema='vico-additional-baselines-report-v1',pairs=43,source_dialogues=10,
        images=images,deca_video=deca,bfm_video=bfm_video,bfm_video_new=bfm,bfm_native_new=native,dualtalk_transfer_proxy=dual,
        learning2listen_native={k:v for k,v in old['deca']['models'].items() if k.startswith('learning2listen/')},
        status=status,paired_sft2752_minus_new=paired,read_only_sources=files,
        limitations=['Local adaptation/retraining; not author-paper table reproduction or matched training budgets.',
            'RLHG/ListenFormer additionally receive speaker motion and listener t0 coefficients.',
            'DECA56, BFM70 native, BFM70 RGB readout and DualTalk transfer proxy are kept separate.',
            'DualTalk uses zero self audio, native25fps, common30fps timestamp mapping; no calibrated neck/head geometry claim.',
            'No human appropriateness ratings; reference distance alone does not establish better reactions.',
            'No OVI or GLARE this round. PCH remains contingent on a complete defensible implementation.'])
    save(WORK/'COMPARISON.json',result)
    return result


def paper(result):
    folder=WORK/'paper_tables';folder.mkdir(exist_ok=True)
    names=list(result['images'])
    def number(v):return '--' if v is None else f'{v:.5f}'
    headers=['Method','PSNR','SSIM','FID','CSIM','CPBD','Exp L1','Head L1']
    rows=[]
    selections=read(WORK/'TRAINING_SELECTION.json')
    selection_note=' '.join(f"{LABELS[m]} completed {s['epochs_completed']} epochs; the evaluated validation-selected checkpoint is epoch {s['selected_epoch']} (step {s['selected_step']})."
                            for m,s in selections.items() if m in result['images'])
    for method in names:
        m=result['images'][method];d=result['deca_video'][method]['metrics']
        rows.append([LABELS[method],*[number(m[k]) for k in ['psnr','ssim','fid','csim','cpbd']],
            number(d['L1']['exp']),number(d['L1']['head_rotation'])])
    with (folder/'rgb_common.csv').open('w') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(rows)
    from export_vico43_sft2752_paper_tables import table
    (folder/'rgb_common.tex').write_text(table(
        'Local evaluation on all 43 official ViCo test pairs (10 source videos). '
        'Image scoring: native or timestamp-mapped 256x256/30fps; official frame weighting and global FID. '
        'Expression/head L1 are clip-averaged common DECA video readouts. '
        'New RLHG/ListenFormer rows appear only after complete official epoch budgets, validation-only checkpoint selection and renderer/scoring acceptance. '
        'Additional speaker motion, training data and transfer adaptations differ; these are not matched-budget author-paper reproductions. '+selection_note,
        'tab:vico-more-rgb',headers,rows,True))
    native_rows=[]
    for method,m in result['bfm_native_new'].items():
        v=m['metrics'];native_rows.append([LABELS[method],*[number(v['FD'][g]) for g in ['exp','angle','trans']],
            number(v['V_D']['exp']),number(v['V_D_GT']['exp'])])
    (folder/'bfm_native.tex').write_text(table('Native BFM70 driver coefficients, unscaled; separate from RGB re-extraction and DECA. '
        'Variance is reported with its GT reference and is not ranked by magnitude.',
        'tab:vico-more-native',['Method','Exp L1','Angle L1','Trans L1','Exp variance','GT variance'],
        native_rows or [['No completed new retraining','--','--','--','--','--']],True))
    bfm_rows=[]
    for method,m in result['bfm_video'].items():
        v=m['metrics'];bfm_rows.append([LABELS[method],*[number(v['FD'][g]) for g in ['exp','angle','trans']],
            number(v['RTLCC']['exp']),number(v['RWTLCC']['exp']),number(v['V_D']['exp'])])
    (folder/'bfm_video.tex').write_text(table('Common BFM70 RGB re-extraction, all unscaled values; native driver output remains separate. '
        'L1 coverage is43, complete-lag RTLCC42 and complete-window RWTLCC41 pairs. '
        'Correlation-curve differences are lower-is-better distances; variance has no monotonic quality ranking.',
        'tab:bfm-video',['Method','Exp L1','Angle L1','Trans L1','RTLCC diff','RWTLCC diff','Exp variance'],bfm_rows,True))
    # A proxy row must never quietly enter the calibrated DECA/RGB main table.
    d=result['dualtalk_transfer_proxy']['metrics']
    dual_rows=[['DualTalk, exploratory transfer',number(d['L1']['exp']),number(d['L1']['jaw_rotation']),'N/A','N/A']]
    (folder/'dualtalk_proxy.tex').write_text(table('Exploratory DualTalk official-checkpoint transfer, zero self audio plus speaker DECA motion. '
        'The original demo core matches exactly, but author FLAME neck rotation and DECA global head rotation have unverified geometric equivalence. '
        'These coefficient proxy distances support no common-head or RGB superiority claim and are not pooled with the main table.',
        'tab:dualtalk-proxy',['Method','Exp proxy L1','Jaw proxy L1','Head claim','RGB'],dual_rows,True))
    l2l_rows=[]
    for method,m in result['learning2listen_native'].items():
        d=m['metrics'];l2l_rows.append([method.split('/')[-1],number(d['L1']['exp']),number(d['L1']['head_rotation']),
            number(d['V_D']['exp']),number(d['V_D_GT']['exp'])])
    (folder/'l2l_native.tex').write_text(table('Previously sealed Learning2Listen native DECA coefficient transfer, all43 pairs. '
        'Four person-specific checkpoints are shown individually; none is chosen using ViCo GT. '
        'Native motion readout and different conditioning/training prevent pooling with common RGB readout.',
        'tab:l2l-native',['L2L checkpoint','Exp L1','Head L1','Exp variance','GT variance'],l2l_rows,True))
    # Keep common RGB motion amplitude visible beside reference distances.
    amplitude=[]
    groups=['exp','head_rotation','jaw_rotation']
    reference=result['deca_video']['joint_sft2752']['metrics']['V_D_GT']
    amplitude.append(['Listener GT',*[f'{reference[g]:.8f}' for g in groups],'100.0'])
    for method,m in result['deca_video'].items():
        variance=m['metrics']['V_D'];gt=m['metrics']['V_D_GT']
        amplitude.append([LABELS[method],*[f'{variance[g]:.8f}' for g in groups],f"{100*variance['exp']/gt['exp']:.1f}"])
    (folder/'rgb_motion_amplitude.tex').write_text(table('Common DECA RGB readout, clip-averaged temporal population variance. '
        'The expression ratio is relative to GT. Both under-motion and over-motion can be inappropriate; variance is not ranked by magnitude. '
        'This table must accompany reference-distance comparisons; no human appropriateness rating has been collected.',
        'tab:rgb-motion-amplitude',['Method','Exp variance','Head variance','Jaw variance','Exp/GT (percent)'],amplitude,True))
    conditions=[]
    for method in result['images']:
        if method in ['rlhg','listenformer']:
            t=read(WORK/method/'TRAIN_PROTOCOL.json');done=read(WORK/method/'TRAIN_COMPLETE.json')
            selected=selections[method]['selected_epoch']
            conditions.append([LABELS[method],'speaker audio + BFM speaker motion + listener t0',
                f"local ViCo: {done['epochs']} epochs, batch {t['global_batch']}; evaluated val-selected epoch {selected}; compiled_loss={t['compiled_loss']}",'RGB, BFM and DECA re-extraction'])
        else:
            inputs={'joint':'speaker audio + listener t0 + base-Qwen predicted conditions + Whisper',
                'joint_sft2752':'speaker audio + listener t0 + SFT2752 predicted conditions + Whisper',
                'dystream':'speaker audio + listener t0; zero self audio',
                'avatarforcing':'speaker audio + listener t0; zero self audio and user visual'}[method]
            training='fixed combined8000; trained with base-Qwen H' if method.startswith('joint') else 'previously sealed public checkpoint transfer'
            conditions.append([LABELS[method],inputs,training,'RGB, BFM and DECA re-extraction'])
    conditions += [['L2L, all four checkpoints','speaker features + listener t0, native transfer',
        'public person-specific checkpoints','native DECA coefficients'],
        ['DualTalk transfer proxy','speaker audio and motion; zero self audio',
         'official public checkpoint','FLAME coefficient proxy; no calibrated head or RGB claim']]
    with (folder/'conditions.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','conditions','training','readout']);w.writerows(conditions)
    with (folder/'deca_per_clip.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','uuid','metric','group','value'])
        for method,data in result['deca_video'].items():
            for r in data['per_clip']:
                for metric,groups in r['metrics'].items():
                    for g,v in groups.items():w.writerow([method,r['id'],metric,g,v])
    with (folder/'all_motion_per_clip.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['space','readout','method','uuid','source_video_id','metric','group','unscaled_value'])
        tables=[('DECA56','video',result['deca_video']),('DECA56','native',result['learning2listen_native']),
            ('BFM70','video',result['bfm_video']),('BFM70','native',result['bfm_native_new']),
            ('FLAME-to-DECA proxy','exploratory',{'dualtalk':result['dualtalk_transfer_proxy']})]
        for space,readout,models in tables:
            for method,m in models.items():
                for row in m['per_clip']:
                    for metric,values in row['metrics'].items():
                        for group,value in values.items():
                            if method=='dualtalk' and group=='head_rotation':continue
                            w.writerow([space,readout,method,row['id'],row['id'].rsplit('_',2)[0],metric,group,value])
    with (folder/'paired_sft2752_minus_new.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['reference','metric','group','mean_difference','ci95_low','ci95_high','clips','sources'])
        for method,m in result['paired_sft2752_minus_new'].items():
            for metric,values in m.items():
                for group,value in values.items():
                    w.writerow([method,metric,group,value['mean'],*(value['ci95'] or [None,None]),value['clips'],value['source_groups']])
    tex=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{booktabs,graphicx,float}
\begin{document}
\section*{Additional ViCo baselines: measured results}
All prior results are read-only inputs. Incomplete training is excluded from formal rows.
\input{rgb_common.tex}\clearpage
\input{rgb_motion_amplitude.tex}\clearpage
\input{bfm_video.tex}\clearpage
\input{bfm_native.tex}\clearpage
\input{l2l_native.tex}\clearpage
\input{dualtalk_proxy.tex}
\end{document}
'''
    # Keep preview headings above their tables; exported paper fragments retain
    # their normal two-column float placement.
    for stem in ['rgb_common','rgb_motion_amplitude','bfm_video','bfm_native','l2l_native','dualtalk_proxy']:
        fragment=(folder/(stem+'.tex')).read_text().replace(r'\begin{table*}[t]',r'\begin{table}[H]').replace(
            r'\end{table*}',r'\end{table}')
        (folder/('preview_'+stem+'.tex')).write_text(fragment)
        tex=tex.replace('{'+stem+'.tex}','{preview_'+stem+'.tex}')
    (folder/'preview.tex').write_text(tex)
    with (folder/'build.log').open('w') as log:
        subprocess.run(['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode','preview.tex'],
            cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
    import pymupdf
    pdf=pymupdf.open(folder/'preview.pdf')
    assert len(pdf)>=6
    outside=[]
    for p in pdf:
        assert p.get_text().strip()
        outside += [(p.number,list(b[:4])) for b in p.get_text('blocks')
            if b[0]<0 or b[1]<0 or b[2]>p.rect.width+.5 or b[3]>p.rect.height+.5]
    assert not outside,outside
    assert 'Overfull' not in (folder/'build.log').read_text()
    save(folder/'PDF_LAYOUT_AUDIT.json',dict(passed=True,pages=len(pdf),out_of_page_blocks=outside,
        overfull_warnings=0,pdf_sha256=sha(folder/'preview.pdf')))
    save(WORK/'PAPER_TABLES.json',dict(pdf=str(folder/'preview.pdf'),pages=len(pdf),
        layout_audit_sha256=sha(folder/'PDF_LAYOUT_AUDIT.json'),
        files={str(p):sha(p) for p in folder.iterdir() if p.suffix in ['.tex','.pdf','.csv']},
        no_unfinished_training_rows=True))


def review(result):
    PAGE.mkdir(parents=True,exist_ok=True)
    if not (PAGE/'status.json').exists():
        (PAGE/'status.json').symlink_to(WORK/'status.json')
    previous=read(ROOT/'data_previews/listen_benchmark_review/vico43_sft2752_20261107/data.json')
    prefix='/listen_benchmark_review/vico43_sft2752_20261107/'
    cases=[];curve_files={};available=set(result['images']);preview=[]
    for method in ['rlhg','listenformer']:
        if method not in available and (WORK/method/'RENDER_COMPLETE.json').exists():
            recs=read(WORK/method/'RENDER_COMPLETE.json')['records']
            assert len(recs)==43
            for r in recs:assert sha(r['path'])==r['sha256']
            available.add(method);preview.append(method)
    l2l_receipts={method:{r['id']:r['prediction_sha256'] for r in m['per_clip']}
                  for method,m in result['learning2listen_native'].items()}
    for c0 in previous['cases']:
        c={k:copy.deepcopy(c0[k]) for k in ['id','duration','speaker','gt','speaker_audio','score']}
        uid=c['id']
        for k,v in c['score'].items():
            if not v.startswith('/'):c['score'][k]=prefix+v
        for method in ['rlhg','listenformer']:
            if method not in available:continue
            dest=PAGE/method;source=WORK/method/'scoring'/method
            if not dest.exists():dest.symlink_to(source,target_is_directory=True)
            c['score'][method]=f'{method}/{uid}.listener.mp4'
        pred=np.load(WORK/'dualtalk/predictions'/(uid+'.npz'))['motion']
        gt=np.load(DECA/'features/gt'/(uid+'.npz'))['motion']
        # Three raw expression channels for transparent coefficient inspection.
        c['dualtalk_curves']=dict(time=(np.arange(len(pred))/30).round(4).tolist(),
            prediction=pred[:,:3].round(6).tolist(),gt=gt[:,:3].round(6).tolist())
        group_slices={'expression':slice(0,3),'head':slice(50,53),'jaw':slice(53,56)}
        native=dict(id=uid,time=c['dualtalk_curves']['time'],
            ground_truth={g:gt[:,sl].round(6).tolist() for g,sl in group_slices.items()},models={})
        native['models']['dualtalk']=dict(label='DualTalk / exploratory FLAME transfer',
            readout='Uncalibrated coefficient transfer; head geometry unverified',
            groups={g:pred[:,group_slices[g]].round(6).tolist() for g in ['expression','jaw']})
        for method in result['learning2listen_native']:
            variant=method.split('/')[-1];path=DECA/'generated'/variant/(uid+'.npz')
            assert sha(path)==l2l_receipts[method][uid]
            motion=np.load(path)['motion'];assert motion.shape==gt.shape and np.isfinite(motion).all()
            native['models'][method]=dict(label='L2L / '+variant,readout='Native DECA coefficient transfer; original person-specific checkpoint',
                groups={g:motion[:,sl].round(6).tolist() for g,sl in group_slices.items()},prediction_sha256=sha(path))
        curve_path=PAGE/'motion_curves'/(uid+'.json');save(curve_path,native)
        curve_files[str(curve_path)]=sha(curve_path)
        c['native_motion_url']='motion_curves/'+uid+'.json'
        c['per_clip_deca']={method:next(r['metrics'] for r in m['per_clip'] if r['id']==uid)
                            for method,m in result['deca_video'].items()}
        cases.append(c)
    labels=dict(LABELS)
    selection=read(WORK/'TRAINING_SELECTION.json')
    for method,entry in selection.items():
        if method in available:labels[method]+=f" / val epoch {entry['selected_epoch']} (trained {entry['epochs_completed']})"
    for method in preview:labels[method]+=' / 视频已生成，评分中'
    data=dict(cases=cases,labels=labels,comparison=result,url=URL,preview_video_methods=preview,training_selection=selection)
    save(PAGE/'data.json',data)
    notice='；'.join(LABELS[m] for m in preview)+'：视频已生成，评分中；当前数值表暂不含这些方法。' if preview else ''
    (PAGE/'index.html').write_text(HTML.replace('<p id="preview_note"></p>','<p id="preview_note">'+html.escape(notice)+'</p>'))
    save(WORK/'REVIEW.json',dict(url=URL,cases=43,video_methods=len(available),formal_video_methods=len(result['images']),
        preview_video_methods=preview,
        dualtalk='raw coefficient curves; no photorealistic video',l2l_native_variants=4,
        native_motion_pairs=43,native_motion_files=curve_files,
        html_sha256=sha(PAGE/'index.html'),data_sha256=sha(PAGE/'data.json')))


def result_text(result):
    selections={}
    for method in ['rlhg','listenformer']:
        if not (WORK/method/'TRAIN_COMPLETE.json').exists():continue
        import torch
        out=WORK/method;best=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        validation=[json.loads(line) for line in (out/'validation.jsonl').read_text().splitlines() if line.strip()]
        winner=min(validation,key=lambda row:row['score'])
        assert best['epoch']==winner['epoch'] and abs(best['validation_score']-winner['score'])<1e-12
        assert best['protocol_sha256']==sha(out/'TRAIN_PROTOCOL.json')
        selections[method]=dict(epochs_completed=read(out/'TRAIN_COMPLETE.json')['epochs'],
            selected_epoch=best['epoch'],selected_step=best['steps'],validation_score=best['validation_score'],
            selection='minimum logged validation46 free-generation L1; no test metrics',
            best_sha256=sha(out/'best.pt'),validation_log_sha256=sha(out/'validation.jsonl'))
    save(WORK/'TRAINING_SELECTION.json',selections)
    rows=['# ViCo43：新增基线比较结果','',
        f'[全部43条同步检查]({URL}) · [完整JSON](COMPARISON.json) · [论文PDF](paper_tables/preview.pdf) · [逐条完整动作CSV](paper_tables/all_motion_per_clip.csv)','',
        '固定43个正式test pair、10个来源；全部保留，一次生成。原生动作、视频反提取及DualTalk迁移代理分表，不能混合数值或推断统一排名。所有旧结果只读复用。','',
        '| 方法 | PSNR ↑ | SSIM ↑ | FID ↓ | CSIM ↑ | CPBD ↑ | 表情L1 ↓ | 头旋转L1 ↓ | 表情方差／GT |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method,m in result['images'].items():
        d=result['deca_video'][method]['metrics'];values=[m[k] for k in ['psnr','ssim','fid','csim','cpbd']]
        values += [d['L1']['exp'],d['L1']['head_rotation']]
        rows.append('| '+LABELS[method]+' | '+' | '.join(f'{v:.6f}' for v in values)+f" | {100*d['V_D']['exp']/d['V_D_GT']['exp']:.1f}% |")
    rows += ['','图像指标沿用官方帧加权和集合FID；动作距离／方差逐pair等权。幅度过小或过大都可能不合适，低参考距离不等于反馈更自然；尚无人工适当性评分。','',
        '## 已完成的新训练及权重选择','']
    if not selections:rows.append('新增训练仍在运行，尚无正式新训练行。')
    for method,m in selections.items():
        rows.append(f"- {LABELS[method]}：完成{m['epochs_completed']}轮；验证集选择第{m['selected_epoch']}轮／{m['selected_step']}步权重。test43未参与统计和选权重。")
    for method in ['listenformer','rlhg']:
        if method not in result['images']:continue
        motion=result['deca_video'][method]['metrics']
        ratio=motion['V_D']['exp']/motion['V_D_GT']['exp']
        if ratio<.1:
            rows += ['',f"{LABELS[method]} 的共同视频读出表情方差仅为GT的{100*ratio:.1f}%，存在明显动作不足。较低的参考距离不能单独作为优于其他方法的证据；当前保留验证集选中的权重，不根据test结果换权重。"]
    rows += ['','## 与SFT2752的来源聚类配对差','',
        '差值为SFT2752−新增方法；距离指标为负表示SFT2752较低。95%区间按来源video_id聚类，10000次、seed137；区间跨0时不作稳定优劣结论。方差只辅助解释。','',
        '| 对照 | 指标 | 平均差 | 95%区间 | pair／来源 |','|---|---|---:|---|---|']
    for method,m in result['paired_sft2752_minus_new'].items():
        for metric,group in [('L1','exp'),('L1','head_rotation'),('Frechet','exp'),('Delta_Frechet','exp'),('V_D','exp')]:
            value=m[metric][group];ci=value['ci95']
            rows.append(f"| {LABELS[method]} | {metric}/{group} | {value['mean']:.7f} | [{ci[0]:.7f}, {ci[1]:.7f}] | {value['clips']}／{value['source_groups']} |")
    rows += ['','## 其他读出及适用边界','',
        'L2L四个人物权重全部逐条提供原生曲线与指标，不按test挑选人物；DualTalk提供官方权重的43条动作迁移代理，neck／global head几何尚未验证，因此不进入共同头部或RGB优劣主张。',
        'RLHG／ListenFormer额外接收speaker动作和listener首帧系数，并在ViCo本地重训；我们的生成器与两种既有视频baseline训练条件不同。本表不是等训练预算或作者论文原表复现。',
        'CSV空白表示不适用（N/A），覆盖数保留在JSON；RTLCC/RWTLCC分别仅支持42／41条，不填充短片段。',
        '[条件与训练口径CSV](paper_tables/conditions.csv) · [BFM视频反提取表](paper_tables/bfm_video.tex) · [独立BFM原生表](paper_tables/bfm_native.tex) · [L2L表](paper_tables/l2l_native.tex) · [DualTalk代理表](paper_tables/dualtalk_proxy.tex) · [方差表](paper_tables/rgb_motion_amplitude.tex)','']
    (WORK/'RESULTS.md').write_text('\n'.join(rows))


def coverage(result):
    (WORK/'paper_tables').mkdir(exist_ok=True)
    rows=[]
    for method in result['images']:
        rows.append(dict(method_or_variant=method,status='complete',rgb_pairs=43,
            native_motion_pairs=43 if method in result['bfm_native_new'] else 0,
            readout='RGB plus common DECA/BFM re-extraction',reason=''))
    for method in result['learning2listen_native']:
        rows.append(dict(method_or_variant=method,status='complete_transfer',rgb_pairs=0,native_motion_pairs=43,
            readout='native DECA coefficients',reason='Original person-specific checkpoint; no photorealistic video'))
    rows.append(dict(method_or_variant='dualtalk',status='complete_exploratory_transfer',rgb_pairs=0,native_motion_pairs=43,
        readout='FLAME coefficient proxy',reason='Neck/global-head equivalence unverified; no calibrated head/RGB ranking'))
    for method in ['rlhg','listenformer']:
        if method in result['images']:continue
        status=result['status'][method]
        rows.append(dict(method_or_variant=method,status=status if isinstance(status,str) else status.get('stage','not_completed'),
            rgb_pairs=0,native_motion_pairs=0,readout='not a formal result',
            reason=status.get('reason',json.dumps(status,ensure_ascii=False)) if isinstance(status,dict) else status))
    pch=read(WORK/'PCH_FEASIBILITY.json')
    rows.append(dict(method_or_variant='pch',status=pch['status'],rgb_pairs=0,native_motion_pairs=0,
        readout='not a formal result',reason=pch['reason']))
    for method in ['ovi','glare']:
        rows.append(dict(method_or_variant=method,status='deferred_by_user',rgb_pairs=0,native_motion_pairs=0,
            readout='not run this round',reason='Explicit user scope'))
    for row in rows:
        method=row['method_or_variant']
        row['preview_rgb_pairs']=43 if method in ['rlhg','listenformer'] and method not in result['images'] and (WORK/method/'RENDER_COMPLETE.json').exists() else 0
    save(WORK/'METHOD_COVERAGE.json',dict(pairs=43,rows=rows,
        counting='Checkpoint/planner variants are separate rows, not separate new methods'))
    with (WORK/'paper_tables/method_coverage.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ViCo43 新增基线</title><style>body{font:16px system-ui;margin:24px;background:#10151e;color:#e5e9f0}a{color:#8fcaff}button,select,input{font:inherit;margin:8px;padding:6px;max-width:100%;box-sizing:border-box}select{min-width:0}#case{width:min(100%,720px)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}video{width:100%;background:#000}figure{margin:0}canvas{width:100%;height:250px;background:#fff}table{border-collapse:collapse;display:block;overflow-x:auto}td,th{padding:8px;border-bottom:1px solid #456}.note{background:#253143;padding:16px}pre{white-space:pre-wrap;font:13px monospace}p,pre{overflow-wrap:anywhere}#refresh{display:none}</style>
<h1>ViCo43：新增基线实测</h1><p class="note">旧结果保持封存。只有完整训练、生成和评分验收完成的方法才进入视频评分表。DualTalk 已完成官方权重推理，单独展示动作系数；FLAME neck 与 DECA global head 几何对应尚未核实，不据此宣称头部指标优劣。人工适当性评分尚未开展。</p>
<div id="summary"></div><p id="status"></p><p id="preview_note"></p><button id="refresh" onclick="location.reload()">新结果已到，刷新查看</button><select id="case"></select><button id="play">同步播放／暂停</button><button id="reset">回到开头</button><input id="seek" type="range" min="0" value="0" step="0.05"><span id="time"></span>
<audio id="audio" controls preload="metadata"></audio><div id="videos" class="grid"></div><h2>本片段：共同视频动作读出</h2><div id="clip_metrics"></div>
<h2>原生动作逐条查看</h2><select id="motion_model"><option value="dualtalk">DualTalk：迁移代理</option><option value="learning2listen/conan">L2L：conan</option><option value="learning2listen/fallon">L2L：fallon</option><option value="learning2listen/stephen">L2L：stephen</option><option value="learning2listen/trevor">L2L：trevor</option></select><select id="motion_group"></select><p id="curve_status"></p><p>红／绿／蓝为三个通道；实线为模型、虚线为 GT，横轴为秒。表情仅展示前 3 个系数。原生动作与视频反提取分开；DualTalk 只作迁移诊断，几何尚未确认的头旋转不展示。</p><canvas id="curves" width="1200" height="260"></canvas><pre id="details"></pre>
<script>let data,current,currentNative,players=[];const curveCache=new Map(),$=id=>document.getElementById(id);function fmt(x){return x===null?'N/A':Number(x).toFixed(5)}
function plot(){if(!currentNative||currentNative.id!==current.id)return;let method=$('motion_model').value,group=$('motion_group').value,m=currentNative.models[method];if(!m||!m.groups[group])return;let ctx=$('curves').getContext('2d'),d={prediction:m.groups[group],gt:currentNative.ground_truth[group]},values=d.prediction.flat().concat(d.gt.flat()),lo=Math.min(...values),hi=Math.max(...values);if(hi===lo)hi=lo+1;ctx.clearRect(0,0,1200,260);ctx.fillStyle='#111';ctx.font='14px sans-serif';ctx.fillText(lo.toFixed(3)+' … '+hi.toFixed(3),8,18);for(let k=0;k<3;k++){for(let key of ['prediction','gt']){ctx.strokeStyle=['#c22','#198342','#2462ce'][k];ctx.setLineDash(key==='gt'?[5,4]:[]);ctx.beginPath();d[key].forEach((a,i)=>{let x=45+i/(currentNative.time.length-1)*1135,y=235-(a[k]-lo)/(hi-lo)*205;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}}ctx.setLineDash([]);ctx.fillStyle='#111';ctx.fillText('0 s',42,256);ctx.fillText(current.duration.toFixed(2)+' s',1100,256);$('curve_status').textContent=m.label+' · '+m.readout;$('curves').dataset.caseId=current.id;$('curves').dataset.motionModel=method;$('curves').dataset.motionGroup=group}
function motionOptions(){if(!currentNative)return;let previous=$('motion_group').value,m=currentNative.models[$('motion_model').value];$('motion_group').replaceChildren();for(let g of Object.keys(m.groups))$('motion_group').add(new Option({expression:'表情系数 0–2',head:'头旋转三轴',jaw:'下颌旋转三轴'}[g],g));if(m.groups[previous])$('motion_group').value=previous;plot()}
async function loadCurves(c){$('curves').removeAttribute('data-case-id');$('curve_status').textContent='加载本片段动作…';let d=curveCache.get(c.id);try{if(!d){let response=await fetch(c.native_motion_url);if(!response.ok)throw Error('动作文件读取失败');d=await response.json();curveCache.set(c.id,d)}if(current.id!==c.id)return;currentNative=d;motionOptions()}catch(e){if(current.id===c.id)$('curve_status').textContent=String(e)}}
function metricTable(c){let t=document.createElement('table'),h=t.insertRow();['模型','表情 L1 ↓','头旋转 L1 ↓','下颌 L1 ↓','表情方差','GT 方差'].forEach(x=>{let th=document.createElement('th');th.textContent=x;h.append(th)});for(let [method,m]of Object.entries(c.per_clip_deca)){let r=t.insertRow();[data.labels[method],fmt(m.L1.exp),fmt(m.L1.head_rotation),fmt(m.L1.jaw_rotation),fmt(m.V_D.exp),fmt(m.V_D_GT.exp)].forEach(x=>r.insertCell().textContent=x)}$('clip_metrics').replaceChildren(t)}
function select(){players.forEach(v=>v.pause());$('audio').pause();current=data.cases[$('case').selectedIndex];$('videos').replaceChildren();for(let [key,url]of [['speaker',current.speaker],['gt',current.gt],...Object.entries(current.score)]){let fig=document.createElement('figure'),label=document.createElement('figcaption'),v=document.createElement('video');label.textContent=data.labels[key]||key;v.src=url;v.muted=true;v.playsInline=true;v.preload='metadata';fig.append(label,v);$('videos').append(fig)}players=[...document.querySelectorAll('video')];$('audio').src=current.speaker_audio;$('seek').max=current.duration;$('seek').value=0;$('details').textContent=JSON.stringify({pair:current.id,duration:current.duration,video_methods:Object.keys(current.score),audio:'原始speaker；baseline无生成listener音轨'},null,2);metricTable(current);loadCurves(current)}
$('motion_model').onchange=motionOptions;$('motion_group').onchange=plot;
$('play').onclick=()=>{let a=$('audio');if(a.paused){players.forEach(v=>{v.currentTime=a.currentTime;v.play().catch(()=>{})});a.play().catch(()=>{})}else{a.pause();players.forEach(v=>v.pause())}};function seek(t){$('audio').currentTime=t;players.forEach(v=>v.currentTime=t);$('seek').value=t}$('reset').onclick=()=>seek(0);$('seek').oninput=()=>seek(Number($('seek').value));$('case').onchange=select;$('audio').ontimeupdate=()=>{let t=$('audio').currentTime;$('seek').value=t;$('time').textContent=t.toFixed(2)+' s';players.forEach(v=>{if(Math.abs(v.currentTime-t)>.12)v.currentTime=t})};$('audio').onended=()=>players.forEach(v=>v.pause());
fetch('data.json').then(r=>r.json()).then(d=>{data=d;data.cases.forEach((c,i)=>$('case').add(new Option((i+1)+' · '+c.id,i)));let t=document.createElement('table');let head=t.insertRow();['方法','PSNR ↑','SSIM ↑','FID ↓','CSIM ↑','CPBD ↑'].forEach(x=>{let h=document.createElement('th');h.textContent=x;head.append(h)});for(let [k,m]of Object.entries(data.comparison.images)){let row=t.insertRow();[data.labels[k],...['psnr','ssim','fid','csim','cpbd'].map(x=>fmt(m[x]))].forEach(x=>row.insertCell().textContent=x)}$('summary').append(t);select();live()}).catch(e=>$('status').textContent=String(e));function live(){fetch('status.json',{cache:'no-store'}).then(r=>r.json()).then(s=>{let labels={train:'训练',generate:'生成',render:'渲染',extract:'提取评分特征',motion_scores:'动作评分',image_scores:'画质评分',complete:'完成',not_scheduled:'预算不足，未安排',incomplete:'未完成',failed:'失败，见说明'},parts=['特征 '+s.features_complete+'/'+s.features_total];for(let [method,stage]of Object.entries(s.methods)){let p=s.training_progress&&s.training_progress[method],key=stage.split(':')[0],label=(data&&data.labels[method])||method;if(key==='train'&&p)parts.push(label+' 第 '+p.epoch+'/'+p.epochs+' 轮，预计训练剩 '+Math.ceil(p.estimated_remaining_h*60)+' 分钟');else parts.push(label+'：'+(labels[key]||(stage.startsWith('waiting')?'等待优先实验完成后评估预算':stage)));if(data&&stage==='complete'&&!data.comparison.images[method])$('refresh').style.display='inline-block'}$('status').textContent=parts.join('；')}).catch(()=>{})}live();setInterval(live,30000);</script></html>'''


def main():
    if (WORK/'COMPLETE.json').exists():
        complete=read(WORK/'COMPLETE.json')
        for p,h in complete['files'].items():assert sha(p)==h,p
        print('Already complete; sealed report verified; no files rewritten',flush=True)
        return
    sys.path.insert(0,str(ROOT/'scripts'))
    result=comparison();result_text(result);coverage(result);paper(result);review(result)
    lines=['# ViCo新增基线：运行与结果','',f'[43条同步检查入口]({URL}) · [结果及解释](RESULTS.md) · [比较JSON](COMPARISON.json) · [论文表格PDF](paper_tables/preview.pdf) · [实时状态](status.json)','',
        '截止：2026-11-08 北京时间12:00（04:00 UTC），用户已确认。ListenFormer原定00:00 UTC冻结；RLHG按实测后处理耗时的3倍（至少90分钟）预留生成、评分与验收时间，完整500轮预算另加15%余量，详见rlhg/RUN_BUDGET.json。',
        '用户已确认暂时仅用node11 GPU7，优先DualTalk＋ListenFormer。现有31卡及AV＋Feature接续队列不改动；先完成ListenFormer，RLHG只在剩余时间足够跑满500轮时再启动。OVI、GLARE不做本轮。','',
        'DualTalk：官方完整权重严格加载、932个状态键逐值核验、官方demo核心输出误差0；43/43条已生成。原生25fps动作→30fps时间戳映射，self音轨置零，额外speaker动作输入。仅动作迁移代理；原实现称neck、DECA称global head，对应几何尚未验收，所以不进入共同RGB／头部优劣主张。','',
        'RLHG：完整500epoch、有效batch16。ListenFormer：完整200epoch、batch128、100epoch阶段切换。351train/46validation；test43不参与统计或选权重。完整候选训练预算不足时保留未完成原因，不拿短训充正式结果。','',
        '两方法共用BFM73／audio45缓存；音频与官方函数真实样本逐值一致，人脸对齐像素一致。训练器日志同步／padding掩码适配已核验前向、loss与全部梯度。源码与兼容层分开；PIRender EMA严格加载且真实训练片段前向通过。','',
        '| 方法 | 当前状态 |','|---|---|']
    for method,status in result['status'].items():lines.append('| '+method+' | '+str(status).replace('|','/')+' |')
    lines+=['','[批准设计](DESIGN.md) · [源代码版本](REPOSITORIES.json) · [协议](protocol.json) · [控制器状态](CONTROLLER_STATE.json)',
        '[持续执行入口](../../compare/deployment/vico_more_controller.py) · [训练／推理](../../compare/deployment/vico_more_train.py) · [缓存](features/protocol.json)','',
        '旧SFT2752、DyStream、Avatar Forcing、L2L及全部训练输入保持原样。视觉相似度、动作距离和动作幅度共同报告；尚未收集反应适当性人工评分。','']
    lines += ['[完整逐条指标CSV](paper_tables/all_motion_per_clip.csv) · [配对区间CSV](paper_tables/paired_sft2752_minus_new.csv) · [条件与训练口径](paper_tables/conditions.csv) · [全部方法完成范围](METHOD_COVERAGE.json)',
        '[训练／验证与test来源时间隔离](SPLIT_ISOLATION_AUDIT.json) · [PCH可行性与未列正式行的原因](PCH_FEASIBILITY.json)','',
        '恢复：在项目根目录运行 `python compare/deployment/vico_more_controller.py`。已有控制器持锁时不重复启动；已经封存完成时仅校验文件，不重新跑模型。各阶段日志与实际退出状态保存在 `jobs/<stage>/`。','']
    (WORK/'README.md').write_text('\n'.join(lines))
    print(URL,flush=True)


if __name__=='__main__':main()
