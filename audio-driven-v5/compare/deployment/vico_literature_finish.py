"""Observe the actual owned extractor; score and verify only after successful exit."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from vico_literature_protocol import ROOT,WORK,OLD,DIM,EMOCA,read,save,sha,commit
from vico_more_common import remote,GPU_UUID
from vico_literature_sources import verify_sources,write_bibliography


def verify():
    p=read(WORK/'protocol.json');ex=read(WORK/'EXTRACTION_COMPLETE.json');result=read(WORK/'RESULTS.json')
    assert result['pairs']==43 and not result['strict_published_table_reproduction']
    assert not result['cross_source_superiority_claim_allowed']
    assert read(WORK/'DEVICE.json')['uuid']==GPU_UUID
    assert read(WORK/'ENCODER_PARITY.json')['passed'] and read(WORK/'MEDIA_PARITY.json')['passed']
    assert read(WORK/'EMOCA_METRIC_PARITY.json')['passed'] and read(WORK/'EMOCA_METRIC_PARITY.json')['compared_scalars']==16
    assert read(WORK/'GT_SELF_CHECK.json')['passed']
    references=read(WORK/'LITERATURE_CELL_PROVENANCE.json')
    assert references['passed'] and references['rows']==9
    for row in references['row_proofs']:
        assert sha(row['published_pdf'])==row['published_pdf_sha256'] and row['exact_numeric_row_matched']
    for name,h in p['source_files'].items():assert sha(name)==h,name
    assert commit(DIM)==p['dim_commit'] and commit(EMOCA)==p['emoca_commit']
    for repo in [DIM,EMOCA]:subprocess.run(['git','diff','--exit-code','--quiet'],cwd=repo,check=True)
    encoder=read(WORK/'ENCODER.json');assert sha(encoder['checkpoint'])==encoder['checkpoint_sha256']
    assert sha(encoder['config'])==encoder['config_sha256']
    assert encoder['full_encoder_state_strict'] and sum(encoder['encoder_tensors'].values())==644
    assert sha(OLD/'COMPLETE.json')==p['original_complete_sha256']
    for path,h in read(OLD/'COMPLETE.json')['files'].items():assert sha(path)==h,path
    features={}
    assert len(ex['records'])==172
    for r in ex['records']:
        assert sha(r['path'])==r['sha256']
        a=np.load(r['path']);assert a['motion'].shape==(r['frames'],56)
        assert np.array_equal(a['frame_indices'],np.arange(r['frames']))
        assert np.array_equal(np.isfinite(a['motion']).all(1),a['valid'])
        assert int(a['valid'].sum())==r['valid_frames']
        assert (r['id'],r['role']) not in features
        features[r['id'],r['role']]=a['motion']
    for t in p['tasks']:assert sha(t['video'])==t['video_sha256']
    assert {x['id'] for x in result['coverage']}==set(p['ids'])
    errors=[]
    for method,row in result['methods'].items():
        assert row['clips']==43 and {r['id'] for r in row['per_clip']}==set(p['ids'])
        for uid,coverage,per in zip(p['ids'],result['coverage'],row['per_clip']):
            assert uid==coverage['id']==per['id']
            gi=np.asarray(coverage['listener_frame_indices']);si=np.asarray(coverage['speaker_frame_indices'])
            assert np.array_equal(gi,si+1)
            assert len(gi)==coverage['scored_frames']==per['scored_frames']
            pred=features[uid,method][gi];gt=features[uid,'gt'][gi]
            for name,sl in [('pose',slice(0,6)),('exp',slice(6,56))]:
                errors.append(abs(float(np.square(gt[:,sl]-pred[:,sl]).mean())-per['MSE'][name]))
    assert max(errors)<1e-12
    expected={}
    for row in result['literature']:
        for k,groups in row['metrics'].items():
            for g,v in groups.items():expected['literature',row['method'],k,g]=v
    for method,row in result['methods'].items():
        for k in ['FD','P_FD','MSE','SID','Var','rPCC']:
            for g,v in row['metrics'][k].items():expected['local_emoca_rgb',method,k,g]=v
    seen=set()
    for row in csv.DictReader((WORK/'paper_tables/all_metrics.csv').open()):
        key=tuple(row[k] for k in ['origin','method','metric','group']);assert key not in seen;seen.add(key)
        v=expected[key]
        if v is None:assert row['value']==''
        else:assert float(row['value'])==v
    assert seen==set(expected)
    assert len(read(WORK/'ALIGNMENT.json')['rows'])==10
    assert read(WORK/'TEST_MANIFEST.json')['author_effective_test_match_confirmed'] is False
    diagnostic=read(WORK/'LOADER_SCOPE_DIAGNOSTIC.json')
    assert diagnostic['primary_scope_unchanged'] and diagnostic['exact_author_test'] is False
    assert set(diagnostic['methods'])==set(p['selected_methods'])
    save(WORK/'FINAL_AUDIT.json',dict(passed=True,at=time.time(),pairs=43,feature_streams=172,
        independent_mse_scalars=len(errors),max_abs=max(errors),all_encoder_tensors=644,
        source_hashes_verified=True,old_sealed_results_unchanged=True,
        literature_vs_local_labels_verified=True,unreported_cells_preserved=True,
        unpublished_protocol_differences_disclosed=True,strict_original_table_claim_allowed=False,
        results_sha256=sha(WORK/'RESULTS.json')))


def paper_preview():
    import pymupdf
    folder=WORK/'paper_tables'
    tex='\\documentclass[10pt]{article}\n\\usepackage[a4paper,margin=15mm]{geometry}\n\\usepackage{booktabs,graphicx,float}\n\\begin{document}\n'
    for stem in ['literature_reference','ours_emoca']:
        content=(folder/(stem+'.tex')).read_text().replace(r'\begin{table*}[t]',r'\begin{table}[H]').replace(r'\end{table*}',r'\end{table}')
        (folder/('preview_'+stem+'.tex')).write_text(content)
        tex+='\\input{preview_'+stem+'.tex}\n\\clearpage\n'
    tex+='\\end{document}\n';(folder/'preview.tex').write_text(tex)
    with (folder/'build.log').open('w') as log:
        subprocess.run(['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode','preview.tex'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert 'Overfull' not in (folder/'build.log').read_text()
    pdf=pymupdf.open(folder/'preview.pdf');assert len(pdf)==2
    for page in pdf:
        assert page.get_text().strip()
        for b in page.get_text('blocks'):assert b[0]>=0 and b[1]>=0 and b[2]<=page.rect.width+.5 and b[3]<=page.rect.height+.5
    save(WORK/'PAPER_PREVIEW.json',dict(passed=True,pages=2,pdf_sha256=sha(folder/'preview.pdf')))


def publish():
    save(WORK/'IMPLEMENTATION_FILES.json',dict(files={str(p):sha(p) for p in sorted(
        (ROOT/'compare/deployment').glob('vico_literature_*.py'))}))
    lines=['# ViCo文献协议对齐与EMOCA实测','',
        '**执行完成：固定43条，172组EMOCA特征，两组我们模型评分。** 原论文未公开的执行差异已披露，尚不能称严格复现原表。','',
        '[结果与解释](RESULTS.md) · [逐项协议对照](ALIGNMENT.md) · [完整JSON](RESULTS.json) · [论文表格PDF](paper_tables/preview.pdf) · [全部数值CSV](paper_tables/all_metrics.csv)','',
        '[测试清单与audio-key差异](TEST_MANIFEST.json) · [文献结果来源](LITERATURE_ROWS.json) · [固定协议](protocol.json) · [执行设计](DESIGN.md)','',
        '[官方权重](DOWNLOADS.json) · [编码器严格加载](ENCODER.json) · [真实前向对照](ENCODER_PARITY.json) · [解码与batch对照](MEDIA_PARITY.json) · [官方指标全43条核验](EMOCA_METRIC_PARITY.json) · [GT自比较](GT_SELF_CHECK.json) · [GT统计锚点](GT_ANCHOR.json) · [最终验收](FINAL_AUDIT.json)','',
        '复算：项目根目录执行 `env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=\'\' compare/.envs/vico_more/bin/python compare/deployment/vico_literature_report.py`。正式封存后先验证COMPLETE.json，不覆盖旧结果。','',
        '代码：[清单与来源](../../compare/deployment/vico_literature_protocol.py) · [EMOCA提取](../../compare/deployment/vico_literature_emoca.py) · [原指标适配](../../compare/deployment/vico_literature_metrics.py) · [评分与导出](../../compare/deployment/vico_literature_report.py) · [接续与验收](../../compare/deployment/vico_literature_finish.py)。','',
        '新发现的ReactFace用户资源见[来源记录](REACTFACE_USER_RESOURCES.json)，不替代EMOCA，不声称已取得其生成器。','']
    (WORK/'README.md').write_text('\n'.join(lines))
    for index in [ROOT/'README.md',ROOT.parent/'OmniResponse/evaluation/README.md']:
        text=index.read_text()
        prefix='**ViCo文献协议对齐实测（2026-11-08）：**'
        relative='experiments/p2_vico_literature_protocol_20261108/README.md' if index==ROOT/'README.md' else '../../audio-driven-v5/experiments/p2_vico_literature_protocol_20261108/README.md'
        entry=prefix+' 指定EMOCA_v2_lr_mse_20，固定43条，联合8000/base与SFT2752两版输出按DIM发布指标补评。DIM→INFP→Avatar Forcing引用行逐项溯源，测试名单／采样／原生与视频读出差异明确披露；不称原表严格复现，不混算排名。[结果、协议与验收]('+relative+')。'
        if prefix not in text:index.write_text(entry+'\n\n'+text)
    save(WORK/'status.json',dict(stage='complete_with_disclosed_protocol_gaps',at=time.time(),pairs=43,methods=2,
        exact_published_table_equivalence=False))
    runtime={'GOAL_PROGRESS.json','LIVE_OBSERVATION.json','FINISH_LAUNCHED.json','finish_runtime.json',
        'EXTRACTION_LAUNCHED.json','SMOKE_LAUNCHED.json','SMOKE_MEDIA_LAUNCHED.json'}
    files=[p for p in WORK.iterdir() if p.suffix in ['.json','.md','.csv'] and p.name not in runtime|{'COMPLETE.json'}]
    files += [p for p in (WORK/'paper_tables').iterdir() if p.suffix in ['.pdf','.tex','.csv','.bib']]
    save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),scope='Paper/code alignment, literature provenance and real EMOCA re-evaluation with disclosed gaps',
        strict_published_table_reproduction=False,unsealed_runtime_files=sorted(runtime),files={str(p):sha(p) for p in files}))


def main():
    if (WORK/'COMPLETE.json').exists():
        for p,h in read(WORK/'COMPLETE.json')['files'].items():assert sha(p)==h,p
        print('Complete; receipts verified');return
    save(WORK/'finish_runtime.json',dict(pid=os.getpid(),started=time.time()))
    job=WORK/'jobs/emoca_extract'
    while not (job/'exit.json').exists():
        if (job/'running.json').exists():
            r=read(job/'running.json');pid=r['pid'];sp=r['supervisor_pid']
            code=f'from pathlib import Path;p=Path("/proc/{pid}/cmdline");q=Path("/proc/{sp}/cmdline");print(bool(p.exists() and b"vico_literature_emoca.py" in p.read_bytes()) or bool(q.exists() and b"emoca_extract" in q.read_bytes()))'
            # Observation/SSH errors are not terminal and never restart a worker.
            try:
                alive=remote(['python','-c',code]).strip()
                save(WORK/'LIVE_OBSERVATION.json',dict(at=time.time(),worker_pid=pid,supervisor_pid=sp,live=alive=='True'))
                if alive!='True' and not (job/'exit.json').exists():raise RuntimeError('Owned extractor and supervisor exited without receipt')
            except RuntimeError as e:
                if str(e).startswith('Owned'):raise
                save(WORK/'OBSERVATION_ERROR.json',dict(at=time.time(),error=str(e)))
        time.sleep(30)
    assert read(job/'exit.json')['returncode']==0,read(job/'exit.json')
    save(WORK/'LIVE_OBSERVATION.json',dict(at=time.time(),live=False,stage='owned extractor exited',
        returncode=0,ended=read(job/'exit.json')['ended']))
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    verify_sources();write_bibliography()
    subprocess.run([str(ROOT/'compare/.envs/vico_more/bin/python'),str(ROOT/'compare/deployment/vico_literature_report.py')],cwd=ROOT,env=env,check=True)
    paper_preview();verify();publish()
    print('Alignment, actual EMOCA evaluation and exports verified',flush=True)


if __name__=='__main__':main()
