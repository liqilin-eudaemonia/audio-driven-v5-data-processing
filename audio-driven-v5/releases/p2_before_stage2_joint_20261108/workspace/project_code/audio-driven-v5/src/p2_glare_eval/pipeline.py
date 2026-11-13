"""Strict paired-video evaluation with resumable independent detection artifacts."""
import os
from pathlib import Path
import subprocess
import sys
from .common import ROOT, read, sha, digest, atomic_json, checked_files, code_identity, job_key, artifact_valid
from .media import probe, aligned
from .upstream import verify_sources, metrics
from .scoring import score_seed, aggregate_seeds, fid_identity

DEFAULT_DETECTOR_PYTHON = ROOT / '.venv-p2-glare-detector/bin/python'
DEFAULT_RUNTIME = ROOT / '.eval_models/detector_runtime.json'


def prepare_pairs(manifest, output, detector_runtime=DEFAULT_RUNTIME):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    data = read(manifest)
    if data.get('schema') != 'p2-glare-pairs-v1' or set(data)-{'schema','samples','annotation_provenance'}:
        raise ValueError('Expected p2-glare-pairs-v1 manifest')
    if not isinstance(data.get('samples'),list) or not data['samples']:
        raise ValueError('Nonempty evaluation samples required')
    rows=[]; ids=set(); groups={}; gt_hashes={}; needs_detector=False
    allowed={'sample_id','sample_seed','gt_video','generated_video','gt_csv','generated_csv','generation_error'}
    for raw in data['samples']:
        if set(raw)-allowed or not isinstance(raw.get('sample_id'),str) or not raw['sample_id'].strip():
            raise ValueError('Invalid pair fields/sample ID')
        seed=raw.get('sample_seed')
        if type(seed) is not int or seed<0:
            raise ValueError('Explicit nonnegative integer sample_seed required')
        key=(raw['sample_id'],seed)
        if key in ids: raise ValueError('Duplicate sample ID / seed')
        ids.add(key);groups.setdefault(seed,set()).add(raw['sample_id'])
        row=dict(raw);row['files']={}
        if 'generation_error' in raw and (not isinstance(raw['generation_error'],str) or not raw['generation_error'].strip()):
            raise ValueError('Generation failure requires a nonempty error message')
        for name in ('gt_video','generated_video','gt_csv','generated_csv'):
            value=raw.get(name)
            if value is None:
                if name=='gt_video':raise ValueError('GT video is required')
                if name=='generated_video' and not raw.get('generation_error'):raise ValueError('Missing generation requires an explicit failure')
                continue
            path=(manifest.parent/value).resolve()
            if not path.is_file():raise FileNotFoundError(path)
            if name.endswith('csv') and path.suffix.lower()!='.csv':raise ValueError('Independent annotations must be CSV, never probe.pt')
            row[name]=str(path);row['files'][str(path)]=sha(path)
        if row.get('generation_error') and row.get('generated_video'):
            raise ValueError('Failed generation cannot also supply a successful video')
        old=gt_hashes.setdefault(raw['sample_id'],row['files'][row['gt_video']])
        if old!=row['files'][row['gt_video']]:raise ValueError('GT changed between generation seeds')
        if not row.get('generation_error') and (not row.get('gt_csv') or not row.get('generated_csv')):
            needs_detector=True
        rows.append(row)
    if any(v!=next(iter(groups.values())) for v in groups.values()):
        raise ValueError('Every seed must cover the exact same planned sample set')
    provenance=data.get('annotation_provenance')
    if any(r.get('gt_csv') or r.get('generated_csv') for r in rows) and (not isinstance(provenance,str) or not provenance.strip()):
        raise ValueError('External CSVs require an annotation_provenance description')
    if provenance and any(x in provenance.lower() for x in ('training_head','probe.pt','event_head')):
        raise ValueError('Training event heads are not independent generated-video annotations')
    lock=verify_sources()
    protocol=dict(schema='p2-glare-evaluation-v1', metric_protocol_id='glare-public-four-reactions-v1',
        detector_protocol_id='glare-public-detector-v1', upstream=lock, code=code_identity(),
        detector_policy=read(ROOT/'configs/p2_glare_detector_v1.json'),
        detector_runtime=read(detector_runtime) if needs_detector else None, fid_runtime=fid_identity(),
        samples=rows,seeds=sorted(groups), sample_count=len(gt_hashes),
        external_annotation_provenance=provenance, reference_kind='independent detector reference, not human ground truth',
        evaluation_scope='six visual reactions; no audio score; not a speaker-appropriateness or E0/E1 causal conclusion',
        input_manifest_sha256=sha(manifest))
    atomic_json(output/'protocol.json',protocol,immutable=True)
    return protocol


def load_protocol(output):
    p=read(Path(output)/'protocol.json')
    if p['schema']!='p2-glare-evaluation-v1' or p['code']!=code_identity() or p['upstream']!=verify_sources():
        raise ValueError('Evaluation code/source changed; create a fresh evaluation version')
    if p['fid_runtime']!=fid_identity():raise ValueError('FID runtime changed after protocol freeze')
    for row in p['samples']:checked_files(row['files'])
    return p


def detector_env():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='2',TF_ENABLE_ONEDNN_OPTS='0',
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2')


def evaluate(output, *, detector_python=DEFAULT_DETECTOR_PYTHON, device='cpu', batch_size=16,
             detection_only=False, workers=4):
    output=Path(output).resolve();p=load_protocol(output)
    if workers<1:raise ValueError('Positive detection worker count required')
    records=[];failures=[]
    # A single evaluator owns a workspace; parallelize through separate shards/workspaces.
    import fcntl
    with (output/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from concurrent.futures import ThreadPoolExecutor
        import json
        videos={}
        for row in p['samples']:
            if row.get('generation_error'):continue
            for side in ('gt','generated'):
                if not row.get(side+'_csv'):
                    video=row[side+'_video'];videos[row['files'][video]]=video
        if videos:
            current=json.loads(subprocess.check_output([str(detector_python),str(ROOT/'scripts/p2_glare_eval.py'),
                'detector-runtime'],text=True,env=detector_env()))
            if current!=p['detector_runtime']:raise ValueError('Detector runtime changed after protocol freeze')
        def ensure_detection(item):
            hashed,video=item;cache=output/'detections'/hashed
            ident=dict(video_sha256=hashed,policy=p['detector_policy'],runtime=p['detector_runtime'],
                       upstream=p['upstream'],code=p['code'])
            try:
                if not artifact_valid(cache,ident):
                    cache.mkdir(parents=True,exist_ok=True)
                    with (cache/'worker.log').open('a') as log:
                        subprocess.run([str(detector_python),str(ROOT/'scripts/p2_glare_eval.py'),'detect-one',
                            '--video',video,'--output',str(cache),'--protocol',str(output/'protocol.json')],
                            check=True,stdout=log,stderr=subprocess.STDOUT,env=detector_env())
                    if not artifact_valid(cache,ident):raise ValueError('Detector did not publish a valid complete artifact')
                return hashed,None
            except Exception as error:
                reason=str(error)
                if (cache/'failed.json').exists():reason+='; '+read(cache/'failed.json').get('error','')
                return hashed,reason
        with ThreadPoolExecutor(max_workers=workers) as pool:
            detector_errors=dict(pool.map(ensure_detection,videos.items()))
        geometries={}
        for i,row in enumerate(p['samples'],1):
            try:
                if row.get('generation_error'):raise ValueError('Generation failed: '+row['generation_error'])
                for side in ('gt','generated'):
                    video=row[side+'_video'];hashed=row['files'][video]
                    if hashed not in geometries:geometries[hashed]=probe(video)
                gt=geometries[row['files'][row['gt_video']]];gen=geometries[row['files'][row['generated_video']]];aligned(gt,gen)
                record=dict(row,geometry=gt)
                for side in ('gt','generated'):
                    csv_path=row.get(side+'_csv')
                    if not csv_path:
                        video=row[side+'_video'];hashed=row['files'][video];cache=output/'detections'/hashed
                        if detector_errors[hashed] is not None:raise ValueError(detector_errors[hashed])
                        csv_path=str(cache/'scores.csv')
                    annotation=metrics()['common'].read_annotation(Path(csv_path))
                    if annotation.frame_count!=gt['frame_count']:raise ValueError('CSV/video frame count mismatch')
                    record[side+'_csv']=csv_path
                    record[side+'_csv_sha256']=sha(csv_path)
                records.append(record)
            except Exception as error:
                failures.append(dict(sample_id=row['sample_id'],sample_seed=row['sample_seed'],error_type=type(error).__name__,error=str(error)))
            atomic_json(output/'progress.json',dict(stage='detection',processed=i,planned=len(p['samples']),
                valid=len(records),failed=len(failures),last_sample=row['sample_id']))
        atomic_json(output/'detected_pairs.json',records)
        report=dict(schema='p2-glare-report-v1',protocol_sha256=digest(p),metric_protocol_id=p['metric_protocol_id'],
            detector_protocol_id=p['detector_protocol_id'],status='incomplete' if failures else 'detected',
            planned_pairs=len(p['samples']),valid_pairs=len(records),failed_pairs=len(failures),failures=failures,
            metrics=None,per_seed={},human_verified=False,reference_kind=p['reference_kind'],scope=p['evaluation_scope'])
        if not failures and not detection_only:
            try:
                if device.startswith('cuda'):
                    from .generation import check_idle_device
                    check_idle_device(device)
                per_seed={str(seed):score_seed([r for r in records if r['sample_seed']==seed],device,batch_size) for seed in p['seeds']}
                report.update(status='complete',per_seed=per_seed,aggregation=aggregate_seeds(per_seed))
                report['metrics']=report['aggregation']['mean']
                report['scoring_settings']=dict(device=device,batch_size=batch_size,fid_runtime=fid_identity())
            except Exception as error:
                report.update(status='scoring_failed',scoring_error=dict(type=type(error).__name__,message=str(error)))
        atomic_json(output/'results.json',report)
        write_report(report,output/'report.md')
        return report


def run(work,checkpoint,evaluation_output,*,devices=('cuda:0',),wait=False,wait_timeout=86400,**score_options):
    from .generation import export_pairs
    work=Path(work).resolve()
    if not devices or len(devices)!=len(set(devices)):raise ValueError('Distinct generation devices required')
    children=[]
    try:
        for index,device in enumerate(devices):
            command=[sys.executable,str(ROOT/'scripts/p2_glare_eval.py'),'generate','--work',str(work),
                '--checkpoint',str(Path(checkpoint).resolve()),'--device',device,'--shards',str(len(devices)),
                '--shard-index',str(index),'--wait-timeout',str(wait_timeout)]
            if wait:command.append('--wait')
            log=(work/f'generation_worker_{index:03d}.log').open('a')
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            log.close();children.append(child)
        codes=[child.wait() for child in children]
        atomic_json(work/'generation_workers.json',dict(devices=list(devices),returncodes=codes))
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:child.wait()
    pairs=work/'pairs.json';export_pairs(work,pairs)
    prepare_pairs(pairs,evaluation_output)
    return evaluate(evaluation_output,**score_options)


def write_report(result,path):
    lines=['# GLARE 四项反应评测','',f"状态：{result['status']}。计划 {result['planned_pairs']} 对，检测有效 {result['valid_pairs']} 对，失败 {result['failed_pairs']} 对。",'',
        '使用公开代码固定协议，自动检测参考一致性；不是论文原始实验的完整复现，也不代表人类适当性判断。','',
        '| 指标 | 结果 | 方向 |','| --- | ---: | --- |']
    for key,direction in (('R-F1','↑'),('R-tIoU','↑'),('R-ATD','↓，公式 ×100'),('R-FID','↓')):
        value=(result.get('metrics') or {}).get(key)
        lines.append(f'| {key} | {value} | {direction} |')
    lines+=['','完整逐类计数、匹配事件、反应帧数、各 seed 结果和失败原因见 results.json。',
            '某必要样本失败时不公布完整集合的四项数值；某 seed 的指标为 null 时，该指标跨 seed 均值也为 null。']
    Path(path).write_text('\n'.join(lines)+'\n')
