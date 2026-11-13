"""Real optimized measurement + accepted crop + exact label reuse + export smoke."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_quality_pipeline import source_job,export_node,measurement_path
from listen_benchmark.j_annotation import read,write
from listen_visual_quality.cli import load_policy
from listen_visual_quality.decision import evidence_index
from listen_visual_quality.rules import analyze_focus

if __name__=='__main__':
    os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[2]})
    base=Path(__file__).parent/'combined_smoke';old=base/'old';work=base/'quality';target=base/'annotation'
    original=ROOT/'experiments/listen_j_annotation_node17_20261029'
    rows=read(ROOT/'experiments/visual_quality_holdout120_20261030/selection.json')['rows']
    selected=[rows[i] for i in (0,1,2,19,24,38,48,51)]
    write(old/'jobs/canary.json',{'job_id':'canary','rows':selected,'source_result_sha256':'multi-source-validation-only','excluded':[]})
    for branch in ('vlm','nod'):
        (old/branch).mkdir(parents=True,exist_ok=True)
        if not (old/branch/'results').exists():(old/branch/'results').symlink_to(original/branch/'results',target_is_directory=True)
    policy=load_policy(ROOT/'configs/listen_visual_quality_v1.json')
    evidence=evidence_index(read(ROOT/'configs/listen_visual_quality_reviewed_evidence_20261030.json'))
    cfg={'old_work':str(old),'bounded_detector_threads':True,'sequential_decode':True,
         'annotation_work':str(target),'hostname':os.uname().nodename,'node':'smoke','scope_clips':len(selected)}
    receipt=source_job(work,cfg,'canary',policy,evidence,'combined-smoke-v1')
    expected={r['id']:r for r in read(ROOT/'experiments/visual_quality_holdout120_20261030/data.json')['results']}
    for row in selected:
        got=read(measurement_path(work,row['sample_id']));r=expected[row['sample_id']]
        r['focus']=analyze_focus(r['focus_samples'],r['source']['duration_s'],policy)
        for key in ('raw_measurements','focus_samples','focus','crop'):assert got[key]==r[key],(row['sample_id'],key)
    write(work/'quality/completed/canary.json',receipt)
    write(work/'queues/quality/state.json',{'failed':{}})
    (work/'ready.jsonl').write_text('{"job_id":"canary","count":8}\n')
    for name in ('config.json','selected.json','resource_tuning.json'):shutil.copy2(original/name,work/('original_'+name))
    report=export_node(work,cfg)
    assert report['decisions']=={'retain':4,'exclude':4,'review':0},report
    for name in ('selected.json','resource_tuning.json'):assert read(target/name)==read(original/name)
    write(base/'validation.json',{'passed':True,'receipt':receipt,'export':report,'all_eight_measurements_exact':True})
    print(json.dumps(report))
