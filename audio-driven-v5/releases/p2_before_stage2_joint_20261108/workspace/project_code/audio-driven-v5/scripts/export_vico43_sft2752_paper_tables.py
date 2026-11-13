"""Export measured ViCo tables, with protocols kept distinct and explicit."""
from pathlib import Path
import csv
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_avatarforcing import read,save,sha

WORK=ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'
METHODS=[('dystream','DyStream (adapted)'),('avatarforcing','Avatar Forcing (adapted)'),
    ('joint','Ours, base Qwen'),('joint_sft2752','Ours, SFT2752 Qwen')]


def table(caption,label,headers,rows,wide=False):
    env='table*' if wide else 'table'
    return '\n'.join([r'% Requires \usepackage{booktabs}',f'\\begin{{{env}}}[t]',
        r'\centering\small',f'\\caption{{{caption}}}',f'\\label{{{label}}}',
        r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',
        r'\toprule',' & '.join(headers)+r' \\',r'\midrule',
        *[' & '.join(row)+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}',f'\\end{{{env}}}',''])


def export():
    result=read(WORK/'COMPARISON.json');folder=WORK/'paper_tables';folder.mkdir(exist_ok=True)
    paired_images=read(WORK/'PAIRED_IMAGE_SCORES.json')
    assert paired_images['complete'] and paired_images['clips']==43
    error={name:max(abs(paired_images['frame_weighted'][short][metric]-result['images'][name][metric])
        for metric in ['psnr','ssim']) for short,name in [('base','joint'),('sft','joint_sft2752')]}
    assert max(error.values())<1e-10,error
    save(WORK/'PAIRED_IMAGE_AUDIT.json',dict(passed=True,clips=43,
        baseline_global_max_error=error['joint'],sft_global_max_error=error['joint_sft2752'],
        comparison_sha256=sha(WORK/'COMPARISON.json')))
    with (folder/'deca_per_clip.csv').open('w') as file:
        writer=csv.writer(file);writer.writerow(['method','uuid','source_video_id','frames','metric','group','value'])
        for name,model in result['deca']['models'].items():
            for row in model['per_clip']:
                for metric,groups in row['metrics'].items():
                    for group,value in groups.items():writer.writerow([name,row['id'],row['id'].rsplit('_',2)[0],row['frames'],metric,group,value])
    with (folder/'bfm_per_clip.csv').open('w') as file:
        writer=csv.writer(file);writer.writerow(['method','uuid','metric','group','unscaled_value'])
        for name,model in result['bfm'].items():
            for row in model['per_clip']:
                for metric,groups in row['metrics_raw'].items():
                    for group,value in groups.items():writer.writerow([name,row['id'],metric,group,value])
    with (folder/'paired_sft_minus_reference.csv').open('w') as file:
        writer=csv.writer(file);writer.writerow(['reference','metric','group','mean_difference','ci95_low','ci95_high','pairs','sources','draws','seed'])
        for name,metrics in result['paired_sft_minus_reference'].items():
            for metric,groups in metrics.items():
                for group,value in groups.items():writer.writerow([name,metric,group,value['mean'],*(value['ci95'] or [None,None]),value['clips'],value['source_groups'],value.get('replicates'),value.get('seed')])
    image_rows=[];bfm_rows=[];amplitude=[]
    for key,label in METHODS:
        image=result['images'][key];motion=result['deca']['models'][key]['metrics']
        image_rows.append([label,f"{image['psnr']:.3f}",f"{image['ssim']:.4f}",f"{image['fid']:.3f}",
            f"{image['csim']:.4f}",f"{image['cpbd']:.4f}",f"{motion['L1']['exp']:.5f}",
            f"{motion['L1']['head_rotation']:.5f}",f"{motion['Frechet']['exp']:.5f}"])
        m=result['bfm'][key]['metrics_raw']
        bfm_rows.append([label,*[f'{m[a][b]:.6f}' for a,b in [('FD','exp'),('FD','angle'),('FD','trans'),('RTLCC','exp'),('RWTLCC','exp')]]])
        v=motion['V_D'];gt=motion['V_D_GT']
        amplitude.append([label,*[f'{v[g]:.8f}' for g in ['exp','head_rotation','jaw_rotation']],f"{100*v['exp']/gt['exp']:.1f}\\%"])
    header=['Method',r'PSNR (dB) $\uparrow$',r'SSIM $\uparrow$',r'FID $\downarrow$',r'CSIM $\uparrow$',r'CPBD $\uparrow$',
        r'Exp. L1 $\downarrow$',r'Head L1 $\downarrow$',r'Exp. Fr\'{e}chet $\downarrow$']
    caption=('Local transfer evaluation on all 43 official ViCo test pairs (10 source videos, seed 137, one sample per pair). '
        'Both Ours rows use the identical step-8000 generator; only the Qwen planner, its descriptions, hidden conditions and Gemma conditions change. '
        'The generator was trained with base-Qwen hidden conditions. DyStream and Avatar Forcing use disclosed source-only listener adaptations, '
        'with zero self audio and no future listener frames. Image metrics use official frame weighting (FID is set-level); '
        'DECA expression-50 and head-axis-angle-3 distances are clip-averaged. All distances and sharpness must be interpreted alongside motion amplitude and appropriateness.')
    (folder/'main_comparison.tex').write_text(table(caption,'tab:vico43-planner',header,image_rows,True))
    (folder/'bfm_supplement.tex').write_text(table(
        'Independent BFM70 readout (expression-64, angle-3 and translation-3). Values are unscaled; no factor of 100 is applied. '
        'RTLCC and RWTLCC are speaker--prediction versus speaker--GT correlation-curve differences, not correlation coefficients; lower is better. '
        'Coverage is 43 pairs for L1, 42 for RTLCC and 41 for RWTLCC. BFM values are not pooled with DECA or EMOCA.',
        'tab:vico43-bfm',['Method',r'Exp. L1 $\downarrow$',r'Angle L1 $\downarrow$',r'Trans. L1 $\downarrow$',
        r'RTLCC diff. $\downarrow$',r'RWTLCC diff. $\downarrow$'],bfm_rows,True))
    gt=result['deca']['models']['joint']['metrics']['V_D_GT']
    amplitude.insert(0,['Listener GT',*[f'{gt[g]:.8f}' for g in ['exp','head_rotation','jaw_rotation']],r'100.0\%'])
    (folder/'motion_amplitude.tex').write_text(table(
        'DECA temporal variance and expression-variance ratio relative to GT. Higher variance is not automatically better: '
        'both under-motion and over-motion can be inappropriate. No human appropriateness scores have been collected.',
        'tab:vico43-amplitude',['Method','Expression variance','Head variance','Jaw variance','Exp./GT'],amplitude,True))
    paired=result['paired_sft_minus_reference']['joint'];rows=[]
    for m,g in [('L1','exp'),('L1','head_rotation'),('L1','jaw_rotation'),('Frechet','exp'),('Frechet','head_rotation'),('RTLCC','exp'),('RWTLCC','exp')]:
        v=paired[m][g];ci=v['ci95'];name=(m+'/'+g).replace('_',r'\_')
        rows.append([name,f"${v['mean']:.6f}$",f"$[{ci[0]:.6f}, {ci[1]:.6f}]$",str(v['clips']),str(v['source_groups'])])
    (folder/'paired_planner_difference.tex').write_text(table(
        'SFT2752 minus base-Qwen paired DECA differences. Negative distance differences favor SFT. '
        r'95\% intervals use 10,000 bootstrap draws resampling entire source-video clusters, seed 137. '
        'These exploratory intervals are conditional on fixed checkpoints and one generation seed, not training or sampling variability.',
        'tab:vico43-paired',['Metric/group','Mean difference',r'95\% interval','Pairs','Sources'],rows,True))
    l2l=[]
    for key,model in result['deca']['models'].items():
        if not key.startswith('learning2listen/'):continue
        m=model['metrics'];l2l.append(['L2L '+key.split('/')[-1],f"{m['L1']['exp']:.5f}",
            f"{m['L1']['head_rotation']:.5f}",f"{m['L1']['jaw_rotation']:.5f}",f"{m['Frechet']['exp']:.5f}"])
    assert len(l2l)==4
    (folder/'l2l_transfer_reference.tex').write_text(table(
        'Supplementary Learning2Listen transfer reference using four author-released person-specific checkpoints, '
        'without retraining on ViCo. These models additionally consume speaker motion and return native DECA motion, '
        'whereas video methods are read out by video re-extraction. No photorealistic L2L video renderer is evaluated. '
        'This table is separate from the common-input video comparison and from published EMOCA results.',
        'tab:vico43-l2l-transfer',['Transferred checkpoint',r'Exp. L1 $\downarrow$',r'Head L1 $\downarrow$',
        r'Jaw L1 $\downarrow$',r'Exp. Fr\'{e}chet $\downarrow$'],l2l,True))
    complete_rows=[];complete_paired=[]
    for metric,groups in result['deca']['models']['joint_sft2752']['metrics'].items():
        for group in groups:
            label=(metric+'/'+group).replace('_',r'\_')
            coverage=[result['deca']['models'][key]['coverage'][metric][group] for key,_ in METHODS]
            coverage_text=str(coverage[0]) if len(set(coverage))==1 else '/'.join(map(str,coverage))
            complete_rows.append([label,*[f"${result['deca']['models'][key]['metrics'][metric][group]:.8f}$" for key,_ in METHODS],coverage_text])
            v=result['paired_sft_minus_reference']['joint'][metric][group];ci=v['ci95']
            complete_paired.append([label,f"${v['mean']:.8f}$",f"$[{ci[0]:.8f}, {ci[1]:.8f}]$",str(v['clips']),str(v['source_groups'])])
    (folder/'deca_complete_appendix.tex').write_text(table(
        'Complete DECA readout, with every metric and parameter group retained. Distance and correlation-curve differences are lower-is-better; '
        'V\\_D and V\\_D\\_GT are temporal variances, without a higher-is-better ranking. Coverage is the valid pair count '
        '(in column order if unequal). Full precision is provided in the accompanying CSV.',
        'tab:vico43-deca-complete',['Metric/group','DyStream','Avatar Forcing','Ours base','Ours SFT2752','Pairs'],complete_rows,True))
    (folder/'paired_complete_appendix.tex').write_text(table(
        'All SFT2752-minus-base DECA differences, including variance and dynamics metrics. Negative differences favor SFT for distance metrics; '
        'variance differences do not have a universal preferred sign. Source-cluster bootstrap intervals use the same fixed seed and 10,000 draws as the main paired table.',
        'tab:vico43-paired-complete',['Metric/group','Mean difference',r'95\% interval','Pairs','Sources'],complete_paired,True))
    before=result['deca']['models']['joint']['metrics'];after=result['deca']['models']['joint_sft2752']['metrics']
    expr=paired['L1']['exp'];dynamics=paired['Delta_Frechet']['exp'];head=paired['L1']['head_rotation']
    assert all(result['images']['joint_sft2752'][metric]<result['images']['joint'][metric] for metric in ['psnr','ssim','csim','cpbd','fid'])
    assert after['V_D']['exp']>before['V_D']['exp'] and after['L1']['exp']>before['L1']['exp']
    assert dynamics['ci95'][1]<0<expr['ci95'][0] and head['ci95'][0]<0<head['ci95'][1]
    assert all(result['paired_sft_minus_reference'][name]['L1'][group]['ci95'][1]<0
        for name in ['dystream','avatarforcing'] for group in ['exp','head_rotation'])
    observation=(f"On all 43 official ViCo test pairs, replacing the base Qwen planner with SFT2752 under an unchanged step-8000 generator "
        f"increased the ratio of mean expression variance to GT from {100*before['V_D']['exp']/before['V_D_GT']['exp']:.1f}% "
        f"to {100*after['V_D']['exp']/after['V_D_GT']['exp']:.1f}%. FID decreased from {result['images']['joint']['fid']:.3f} "
        f"to {result['images']['joint_sft2752']['fid']:.3f}. Expression Delta-Frechet decreased from {before['Delta_Frechet']['exp']:.5f} "
        f"to {after['Delta_Frechet']['exp']:.5f} (paired difference {dynamics['mean']:.5f}, 95% interval "
        f"[{dynamics['ci95'][0]:.5f}, {dynamics['ci95'][1]:.5f}]). Framewise expression L1 increased from {before['L1']['exp']:.5f} "
        f"to {after['L1']['exp']:.5f} (difference {expr['mean']:.5f}, 95% interval [{expr['ci95'][0]:.5f}, {expr['ci95'][1]:.5f}]); "
        f"PSNR, SSIM, CSIM and CPBD were also lower. The head-rotation L1 difference was {head['mean']:.5f} "
        f"with interval [{head['ci95'][0]:.5f}, {head['ci95'][1]:.5f}], which includes zero. "
        "These results show a trade-off between motion amplitude/dynamics and framewise correspondence. "
        "The generator was trained with base-Qwen hidden conditions; SFT replaces both predicted descriptions and hidden conditions at inference. "
        "Intervals resample source-video clusters and are conditional on fixed checkpoints and generation seed 137. Human-rated appropriateness was not measured.")
    comparison=(f"Under the disclosed source-only transfer protocol, the SFT variant achieved expression/head-rotation L1 "
        f"of {after['L1']['exp']:.5f}/{after['L1']['head_rotation']:.5f}, compared with "
        f"{result['deca']['models']['dystream']['metrics']['L1']['exp']:.5f}/{result['deca']['models']['dystream']['metrics']['L1']['head_rotation']:.5f} "
        f"for DyStream and {result['deca']['models']['avatarforcing']['metrics']['L1']['exp']:.5f}/"
        f"{result['deca']['models']['avatarforcing']['metrics']['L1']['head_rotation']:.5f} for Avatar Forcing. "
        "The source-cluster intervals for all four paired L1 differences were below zero. Training budgets are unmatched; "
        "this is a common-input transfer comparison with the stated adaptations.")
    (folder/'observations.txt').write_text(observation+'\n\n'+comparison+'\n')
    (folder/'observations.tex').write_text(observation.replace('%',r'\%')+'\n\n'+comparison+'\n')
    preview=folder/'preview.tex'
    preview.write_text('\n'.join([r'\documentclass[10pt]{article}',
        r'\usepackage[margin=1cm,includefoot]{geometry}',r'\usepackage{booktabs}',r'\begin{document}',
        *[r'\input{'+n+r'.tex}\clearpage' for n in ['main_comparison','bfm_supplement','motion_amplitude',
            'paired_planner_difference','l2l_transfer_reference','deca_complete_appendix','paired_complete_appendix']],
        r'\section*{Measured observations}',r'\input{observations.tex}',r'\end{document}','']))
    with (folder/'preview_build.log').open('w') as log:
        subprocess.run(['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode',preview.name],
            cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    files=list(folder.glob('*.tex'))+list(folder.glob('*.csv'))+[folder/'observations.txt',folder/'preview.pdf',folder/'preview_build.log',WORK/'PAIRED_IMAGE_AUDIT.json']
    save(WORK/'PAPER_TABLES.json',dict(source_comparison_sha256=sha(WORK/'COMPARISON.json'),
        exporter_sha256=sha(Path(__file__)),files={str(p):sha(p) for p in files},latex_compilation_passed=True,
        model_selection='All four video arms reported; no per-clip winner selection; no human ratings fabricated'))


if __name__=='__main__':export()
