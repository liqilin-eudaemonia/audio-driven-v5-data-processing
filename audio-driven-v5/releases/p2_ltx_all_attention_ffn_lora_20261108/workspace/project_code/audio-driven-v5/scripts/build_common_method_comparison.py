"""Publish auditable local comparisons separately from published table values."""
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'experiments/common_method_comparison_20261106'
OUT = ROOT / 'data_previews/listen_benchmark_review/common_method_comparison_20261106'
DECA = WORK / 'vico_deca'
URL = 'http://localhost:8876/listen_benchmark_review/common_method_comparison_20261106/'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def published():
    """Transcribe DIM table1 only, and verify every numeric row against the PDF."""
    source = ROOT / 'writing/17_DIM_ECCV2024.pdf'
    try:
        import pymupdf
    except ModuleNotFoundError:
        # Headless model environments need not install a PDF renderer. Reuse the
        # already verified transcription only while the source PDF is unchanged.
        cached=read(WORK/'published_reference.json')
        assert cached['source_pdf_sha256']==sha(source) and cached['page']==11 and cached['table']==1
        text='\n'.join(row['method']+'\n'+' '.join(f'{row["metrics"][m][g]:.2f}' for m in ['FD','P-FD','MSE','SID','Var','rPCC'] for g in ['exp','pose']) for row in cached['rows'])
    else:
        with pymupdf.open(source) as doc:
            text = doc[10].get_text().split('Table 2:')[0]
    labels = ['Random', 'Nearest Audio', 'Nearest Motion', 'ELP∗[66]', 'RLHG [84]', 'L2L [50]', 'L2L† [50]', 'DIM-Listener †']
    numbers = [
        '72.88 0.12 75.82 0.12 2.05 0.03 4.68 3.83 1.49 0.02 0.09 0.17',
        '65.77 0.10 68.84 0.10 1.77 0.03 4.81 3.93 1.49 0.02 0.07 0.11',
        '42.41 0.06 45.33 0.06 1.27 0.02 4.62 3.74 1.49 0.02 0.08 0.12',
        '47.17 0.08 47.48 0.08 0.98 0.02 1.76 1.66 1.49 0.02 0.09 0.01',
        '39.02 0.07 40.18 0.07 0.86 0.01 3.62 3.17 1.52 0.02 0.08 0.02',
        '33.93 0.06 35.88 0.06 0.93 0.01 2.77 2.66 0.83 0.02 0.06 0.08',
        '31.03 0.06 33.02 0.06 0.87 0.01 3.53 3.05 0.83 0.02 0.06 0.09',
        '23.88 0.06 24.39 0.06 0.70 0.01 3.71 2.35 1.53 0.02 0.06 0.03',
    ]
    metrics = ['FD', 'P-FD', 'MSE', 'SID', 'Var', 'rPCC']
    rows = []
    for label, line in zip(labels, numbers):
        assert re.search(re.escape(label) + r'\s*' + re.escape(line), text), f'PDF table transcription mismatch: {label}'
        values = list(map(float, line.split()))
        rows.append(dict(method=label, metrics={m: dict(exp=values[2*i], pose=values[2*i+1]) for i, m in enumerate(metrics)},
                         origin='reported by DIM authors; not executed locally', candor_pretrained='†' in label))
    return dict(source_pdf=str(source), source_pdf_sha256=sha(source), page=11, table=1,
        url='https://github.com/Boese0601/Dyadic-Interaction-Modeling', rows=rows,
        coefficient_space='EMOCA; differs from local DECA and BFM protocols',
        policy='Reference table only; no local ours row, cross-table rank or improvement percentage.',
        notes=['†: CANDOR pretraining; ∗: ELP reimplementation by DIM authors.',
               'Exact effective test manifest, main-table speaker setting and candidate selection remain unconfirmed.'])


def video_rows():
    sources, rows = {}, []
    joint = ROOT / 'experiments/p2_vico256_combined8000_20261106/RESULTS.json'
    dy = ROOT / 'experiments/p2_compare_vico43_20261106/dystream/IMAGE_RESULTS.json'
    for label, path in [('联合模型8000', joint), ('DyStream listener适配', dy)]:
        d = read(path)
        sources[str(path)] = sha(path)
        rows.append(dict(dataset='ViCo test43', method=label, arm='combined8000' if path == joint else 'dystream',
            clips=43, metrics=d['metrics'], coverage={k: dict(pairs=43, frames=16288) for k in d['metrics']},
            source=str(path), source_sha256=sha(path),
            geometry='生成512×512/25fps → 评分256×256/30fps',
            training_provenance='原始训练来源保留；训练数据与测试集的去重尚未在本次汇总中核实',
            conditioning='speaker音频＋listener首帧；DyStream自身音频为零'))
    for dataset, parent, target, reference_sha in [
        ('ResponseNet fixed200', 'p2_responsenet_random200_twelveway_20261105', 'p2_compare_responsenet_random200_20261106',
         '8279171c39501a644c75c36b2e5fe110036270a28d9f38f0e097503ba2c61146'),
        ('Seamless fixed200', 'p2_seamless200_tenway_24gpu_20261105', 'p2_compare_seamless200_20261106',
         'd4bda633eefebfd0b0a1b055684a749d86a6b28ec7c9d1a032e4a151b0468b07')]:
        path = ROOT / 'experiments' / parent / 'RESULTS.json'
        parent_protocol=ROOT / 'experiments' / parent / 'protocol.json'
        dataset_key='responsenet' if dataset.startswith('ResponseNet') else 'seamless'
        spec=read(parent_protocol)['datasets'][dataset_key]
        assert spec['references_sha256']==reference_sha
        assert sha(Path(spec['prepared'])/'references.jsonl')==reference_sha
        sources[str(parent_protocol)]=sha(parent_protocol)
        sources[str(path)] = sha(path)
        original = read(path)
        # Include every frozen arm, never choose a checkpoint using test scores.
        for name, d in original.items():
            assert d['requested_pairs'] == 200
            rows.append(dict(dataset=dataset, method='我们 / ' + d['arm'], arm=d['arm'], clips=200,
                checkpoint=d['checkpoint'], metrics={k: v for k, v in d['metrics'].items() if k != 'utmos'},
                coverage=d['coverage'], status=d['status'], source=str(path), source_key=name, source_sha256=sha(path),
                geometry='沿用该固定测试清单的正式评分几何', references_sha256=reference_sha,
                conditioning='沿用已封存的speaker音频＋listener首帧与规划器输入协议',
                training_provenance='checkpoint路径已列出；未基于测试结果选择最佳arm'))
        path = ROOT / 'experiments' / target / 'dystream/RESULTS.json'
        d = read(path)
        assert d['references_sha256'] == reference_sha and d['requested_pairs'] == 200
        sources[str(path)] = sha(path)
        rows.append(dict(dataset=dataset, method='DyStream listener适配', arm='dystream', clips=200,
            metrics=d['metrics'], coverage=d['coverage'], status=d['status'], source=str(path), source_sha256=sha(path),
            geometry='512×512/25fps生成，沿用同一固定清单的正式评分几何', references_sha256=reference_sha,
            conditioning='speaker音频＋listener首帧；listener自身音频为零；无生成listener声音',
            training_provenance='官方预训练权重，未在本轮重训'))
    # Later completed experiments contain the user-selected combined8000 model
    # and two additional Whisper arms; do not omit them from the consolidated view.
    for experiment in ['p2_seamless200_whisper_20261105','p2_combined_whisper_eval_20261106']:
        path=ROOT/'experiments'/experiment/'RESULTS.json'
        protocol_path=path.parent/'protocol.json'
        protocol=read(protocol_path)
        sources[str(path)]=sha(path);sources[str(protocol_path)]=sha(protocol_path)
        for key,d in read(path).items():
            dataset_key=key.rsplit('/',1)[1]
            dataset='Seamless fixed200' if dataset_key=='seamless' else 'ResponseNet fixed200'
            if any(r['dataset']==dataset and r['arm']==d['arm'] for r in rows):continue
            spec=protocol['datasets'][dataset_key]
            assert sha(Path(spec['prepared'])/'references.jsonl')==spec['references_sha256']
            assert d['requested_pairs']==200
            rows.append(dict(dataset=dataset,method='我们 / '+d['arm'],arm=d['arm'],clips=200,
                checkpoint=d['checkpoint'],metrics={k:v for k,v in d['metrics'].items() if k!='utmos'},
                coverage=d['coverage'],status=d['status'],source=str(path),source_key=key,source_sha256=sha(path),
                references_sha256=spec['references_sha256'],geometry='沿用同一固定清单的正式评分几何',
                conditioning='已封存speaker音频＋listener首帧与原始规划器',
                training_provenance='原始训练配置与checkpoint保留；未按测试分数选择'))
    dataset_order={'ViCo test43':0,'ResponseNet fixed200':1,'Seamless fixed200':2}
    rows.sort(key=lambda r:(dataset_order[r['dataset']],0 if r['arm'] in ['combined8000','base_event_feature_whisper_8000'] else 1 if r['arm']=='dystream' else 2))
    return rows, sources


def table(headers, rows):
    def cell(value):
        return html.escape('N/A' if value is None else f'{value:.6f}' if isinstance(value, float) else str(value))
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + cell(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + cell(x) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def video_pairs():
    datasets={}
    for dataset,experiment in [('responsenet','p2_compare_responsenet_random200_20261106'),('seamless','p2_compare_seamless200_20261106')]:
        dy=ROOT/'experiments'/experiment/'dystream';protocol=read(dy/'protocol.json');cases=[]
        for item in protocol['samples']:
            uid=item['id'];joint=ROOT/'experiments/p2_combined_whisper_eval_20261106/generated/base_event_feature_whisper_8000'/dataset/uid
            other=dy/'generated'/uid
            sources=dict(joint=joint/'generated.mp4',dystream=other/'generated.mp4',speaker=Path(item['speaker']),
                         listener_audio=joint/'listener.wav',first_frame=Path(item['image']))
            sources['gt']=Path(item['reference']['gt']) if dataset=='responsenet' else WORK/'gt_preview/seamless'/(uid+'.mp4')
            receipt=read(joint/'complete.json');dy_receipt=read(other/'complete.json')
            expected=dict(joint=receipt['files']['generated.mp4']['sha256'],dystream=dy_receipt['native_sha256'],
                          listener_audio=receipt['files']['listener.wav']['sha256'],first_frame=item['image_sha256'])
            if dataset=='seamless':expected['gt']=read(sources['gt'].with_suffix('.json'))['preview_sha256']
            urls={};hashes={}
            for role,path in sources.items():
                assert path.is_file(),path
                digest=sha(path)
                if role in expected:assert digest==expected[role],path
                hashes[role]=digest
                link=OUT/'media'/dataset/uid/(role+path.suffix)
                link.parent.mkdir(parents=True,exist_ok=True)
                if link.is_symlink():assert link.resolve()==path.resolve()
                elif link.exists():raise FileExistsError(link)
                else:link.symlink_to(path.resolve())
                urls[role]=str(link.relative_to(OUT))
            cases.append(dict(id=uid,duration=item['duration'],frames=item['frames'],fps=item['fps'],urls=urls,sha256=hashes))
        assert len(cases)==len({c['id'] for c in cases})==200
        datasets[dataset]=dict(cases=cases,references_sha256=protocol['references_sha256'])
    result=dict(schema='native-video-comparison-fixed200-v1',datasets=datasets,
        models=['combined8000','dystream'],selection='all200 reference pairs, original order, no metric-based selection',
        resolution='both native512x512/25fps; score geometries differ by dataset',
        audio='DyStream has no generated listener audio; separate speaker and combined listener WAV controls')
    save(WORK/'video_pairs.json',result);save(OUT/'video_pairs.json',result)
    shutil.copyfile(ROOT/'scripts/common_method_video_pairs.html',OUT/'video_pairs.html')
    return result


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sources = video_rows()
    literature = published()
    common = read(DECA / 'RESULTS.json') if (DECA / 'COMPLETE.json').exists() else None
    if common:
        assert read(DECA / 'FINAL_AUDIT.json')['passed']
        sources[str(DECA / 'RESULTS.json')] = sha(DECA / 'RESULTS.json')
    active_file=WORK/'ACTIVE_L2L_RUN.json'
    retrain_dir=Path(read(active_file)['path']) if active_file.exists() else WORK/'l2l_retrained'
    retrained=None
    quality=read(retrain_dir/'QUALITY_GATE.json') if (retrain_dir/'QUALITY_GATE.json').exists() else {}
    if (retrain_dir/'COMPLETE.json').exists() and quality.get('passed'):
        assert read(retrain_dir/'FINAL_AUDIT.json')['passed']
        retrained=read(retrain_dir/'RESULTS.json')
        sources[str(retrain_dir/'RESULTS.json')]=sha(retrain_dir/'RESULTS.json')
        if common:
            common['models']['learning2listen/retrained']=retrained
    pilot_path=WORK/'l2l_training_pilot/RESULTS.json'
    pilot=read(pilot_path) if (pilot_path.parent/'COMPLETE.json').exists() else None
    if pilot:sources[str(pilot_path)]=sha(pilot_path)
    data = dict(schema='common-method-comparison-v1', local_video_results=rows, common_deca=common,
        published_reference=literature, l2l_training_feasibility=pilot, l2l_retrained=retrained, source_sha256=sources,
        claims=dict(published_tables_comparable_to_local=False, controlled_training_comparison=False,
                    no_test_checkpoint_selection=True, existing_generation_reused=True),
        notes=['统一测试材料不等于训练条件一致；各模型训练来源需明确列出。',
               '保留全部既有arm；不同数据集的指标分别展示，不合并平均。',
               'L2L原生动作与视频反提取动作的输出路径不同。',
               'Seamless情绪和ΔResp存在支持不足，逐项覆盖数随结果保存。'])
    provenance=read(WORK/'TRAINING_PROVENANCE.json') if (WORK/'TRAINING_PROVENANCE.json').exists() else None
    data['training_provenance']=provenance
    if provenance:
        sources[str(WORK/'TRAINING_PROVENANCE.json')]=sha(WORK/'TRAINING_PROVENANCE.json')
    for row in rows:
        row['primary_comparison']=row['arm'] in ['combined8000','base_event_feature_whisper_8000','dystream']
        if row['arm'] in ['combined8000','base_event_feature_whisper_8000'] and provenance:
            row['training_provenance']='Seamless训练缓存285638条／500.000285h；与当前Seamless200无同ID／同source-pair交集；未核实所有人物和外部预训练重叠'
    save(WORK / 'RESULTS.json', data)
    save(WORK / 'published_reference.json', literature)
    with (WORK / 'video_metrics.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'method', 'arm', 'checkpoint', 'metric', 'value', 'coverage_json', 'source', 'source_sha256'])
        for r in rows:
            for metric, value in r['metrics'].items():
                w.writerow([r['dataset'], r['method'], r['arm'], r.get('checkpoint', ''), metric, value,
                            json.dumps(r['coverage'], separators=(',', ':')), r['source'], r['source_sha256']])
    if common:
        with (WORK / 'common_deca_metrics.csv').open('w') as f:
            w = csv.writer(f)
            w.writerow(['method', 'readout', 'metric', 'group', 'raw_value', 'coverage_clips'])
            for name, model in common['models'].items():
                for metric, groups in model['metrics'].items():
                    for group, value in groups.items():
                        w.writerow([name, model['readout'], metric, group, value, model['coverage'][metric][group]])
    body = ['<h1>方法比较：统一实测与文献参考</h1>',
        '<p>复用已完成的推理；ViCo 43条、ResponseNet 200条、Seamless 200条。全部既有模型版本均保留，不按测试成绩挑选。</p>',
        '<nav><a href="../p2_vico43_compare_20261106/">ViCo逐条视频与动作</a> · <a href="data.json">完整数据／覆盖数</a> · <a href="video_metrics.csv">视频指标CSV</a> · <a href="published_reference.json">文献引用来源</a></nav>',
        '<p class="note">以下是同测试材料的模型比较。训练数据、额外输入与预训练规模尚未统一，不能据此称为同训练条件的比较。</p>']
    if provenance:
        body.append('<p>当前联合8000训练缓存：285,638条／约500h，全部标记train；与Seamless固定200条无同ID／同source-pair交集。此检查不等于人物、源对话及全部外部预训练去重。<a href="training_provenance.json">训练来源核查</a>。每个测试集优先列出预先选定的联合8000与DyStream，其余arm完整保留。</p>')
    body.append('<p><a href="video_pairs.html">打开ResponseNet／Seamless全部400对实际推理视频</a>：联合8000与DyStream同步播放，支持speaker音频及联合模型生成声音；按原始清单顺序，不挑选案例。</p>')
    for dataset in dict.fromkeys(r['dataset'] for r in rows):
        subset = [r for r in rows if r['dataset'] == dataset]
        keys = ['psnr', 'ssim', 'fid', 'csim', 'cpbd'] if dataset.startswith('ViCo') else ['psnr', 'ssim', 'lpips', 'fid', 'fvd', 'emo_acc', 'delta_resp']
        body += ['<h2>' + html.escape(dataset) + '</h2>', table(['方法／checkpoint版本'] + keys + ['状态'],
            [[r['method']] + [r['metrics'].get(k) for k in keys] + [r.get('status', 'complete')] for r in subset])]
        if dataset.startswith('Seamless'):
            body.append('<p>DyStream画质200/200，情绪／ΔResp199/200。其他arm的逐项覆盖保留在完整数据中，不能将partial理解为全部200条均有该指标。</p>')
    body.append('<h2>ViCo共同DECA动作空间</h2>')
    if common:
        columns = [('L1', 'exp'), ('L1', 'head_rotation'), ('L1', 'jaw_rotation'), ('Frechet', 'exp'),
                   ('RTLCC', 'exp'), ('RWTLCC', 'exp'), ('V_D', 'head_rotation')]
        body += [table(['方法', '输出路径'] + [k + '/' + g for k, g in columns],
            [[name, r['readout']] + [r['metrics'][k][g] for k, g in columns] for name, r in common['models'].items()]),
            '<p>原始DECA表情50＋头旋转3＋下颌3，无平移；RTLCC42/43、RWTLCC41/43，其余43/43。L2L为四个人物模型迁移测试，无GT择优。V-D须结合GT方差解读。</p>',
            '<p><a href="common_deca_metrics.csv">共同动作指标CSV</a> · <a href="common_deca.json">全部逐片段动作评分</a></p>']
    else:
        body.append('<p>统一DECA提取／评分进行中，完成且验收后发布。原有BFM与L2L结果仍可在逐条页面查看。</p>')
    body += ['<h2>文献参考：DIM论文Table 1</h2>',
        '<p class="note">下表全部为DIM作者报告的结果，未在本地运行；使用EMOCA，未确认与本地划分和采样一致。本表不加入我们的分数、不跨表排名或计算提升比例。†表示CANDOR预训练，∗表示DIM作者重实现ELP。</p>',
        table(['论文中的方法'] + [m + '/' + g for m in ['FD', 'P-FD', 'MSE', 'SID', 'Var', 'rPCC'] for g in ['exp', 'pose']],
              [[r['method']] + [r['metrics'][m][g] for m in ['FD', 'P-FD', 'MSE', 'SID', 'Var', 'rPCC'] for g in ['exp', 'pose']] for r in literature['rows']]),
        '<p>逐行数值已与本地DIM PDF第11页Table 1核对，PDF摘要保存在来源文件。</p>',
        '<h2>L2L重训适配</h2><p>官方train398有1条与test共享speaker时间段，新增训练适配隔离该条；其余397条按源对话划分为train351、validation46，test43和OOD42隔离。小规模验证与最终重训基线分别记录；未经完整训练和测试的产物不加入正式排名。</p>']
    if pilot:
        body.append('<p>两阶段训练可行性验证已通过：train8／validation4，VQ及predictor各100步，训练与验证损失均有限。试训只验证流程，独立于正式重训。<a href="l2l_training_feasibility.json">试训记录与损失</a></p>')
    if retrained:
        body.append('<p>保守397条全量特征、两阶段训练及test43评分已完成验收，已加入上方共同DECA表。采用本地训练设置，不能标为DIM原表的L2L基线。<a href="l2l_retrained.json">本地重训完整结果</a></p>')
    elif quality.get('status')=='nonconverged_diagnostic':
        body.append('<p>前轮训练流程执行完毕，但VQ验证最优停在第1步，未通过训练有效性检查，已从正式比较中移除；原权重和分数保留为诊断记录。修正训练流程后再评估。</p>')
    elif (retrain_dir/'LAUNCH.json').exists():
        state=read(retrain_dir/'status.json') if (retrain_dir/'status.json').exists() else {'stage':'全量特征提取'}
        body.append('<p id="trainingProgress">保守397条正式重训已启动：'+html.escape(str(state.get('stage')))+'。完成验证集选权重、test43与验收后才加入上方实测表。</p>')
        body.append('<script>async function progress(){try{const r=await fetch("training_progress.json",{cache:"no-store"});if(!r.ok)return;const d=await r.json(),el=document.getElementById("trainingProgress");if(d.complete){el.textContent="正式重训与test43已完成，请刷新页面查看新增结果。";return}if(d.stage==="paused_after_vq"){el.textContent="已按你的要求在VQ完成后暂停，GPU7已释放；predictor与test43未启动。";return}if(d.stage==="failed"){el.textContent="重训出现错误："+(d.failure||JSON.stringify(d.failures))+"；尚未完成正式基线。";return}el.textContent="正式重训："+(d.stage==="features"?"全量特征 "+d.feature_pairs+"/397":d.stage+" · "+d.training.step+"步")+" · node11 GPU7 · 更新 "+d.updated_utc}catch(e){}}progress();setInterval(progress,15000)</script>')
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>方法比较</title><style>body{font:15px/1.6 system-ui;margin:24px auto;padding:0 16px;max-width:1700px;color:#25312e}a{color:#245d83}h1{font-size:25px}h2{margin-top:30px;font-size:19px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:12px}th,td{text-align:right;border-bottom:1px solid #ddd;padding:7px}th:first-child,td:first-child{text-align:left}th{background:#f1f4f2}.note{background:#faf3e5;padding:12px}nav{margin:15px 0}</style>' + ''.join(body) + '</html>'
    (OUT / 'index.html').write_text(document)
    for name, dest in [('RESULTS.json', 'data.json'), ('video_metrics.csv', 'video_metrics.csv'), ('published_reference.json', 'published_reference.json')]:
        shutil.copyfile(WORK / name, OUT / dest)
    if common:
        shutil.copyfile(DECA / 'RESULTS.json', OUT / 'common_deca.json')
        shutil.copyfile(WORK / 'common_deca_metrics.csv', OUT / 'common_deca_metrics.csv')
    if pilot:shutil.copyfile(pilot_path, OUT/'l2l_training_feasibility.json')
    if retrained:shutil.copyfile(retrain_dir/'RESULTS.json',OUT/'l2l_retrained.json')
    if provenance:shutil.copyfile(WORK/'TRAINING_PROVENANCE.json',OUT/'training_provenance.json')
    pairs=video_pairs()
    for path, expected in sources.items():
        assert sha(path) == expected
    save(WORK / 'PUBLISHED.json', dict(url=URL, video_rows=len(rows), common_deca_complete=bool(common),
        video_pairs=sum(len(d['cases']) for d in pairs['datasets'].values()),
        pdf_table_rows_verified=len(literature['rows']), source_hashes_verified=len(sources),
        data_sha256=sha(OUT / 'data.json'), html_sha256=sha(OUT / 'index.html')))
    print(URL)


if __name__ == '__main__':
    main()
