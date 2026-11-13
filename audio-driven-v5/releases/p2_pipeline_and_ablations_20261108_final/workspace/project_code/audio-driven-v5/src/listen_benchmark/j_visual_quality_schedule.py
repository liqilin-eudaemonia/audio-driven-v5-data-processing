"""Coordinate early CPU visual filtering and later Omni admission.

The immutable measurement contract is unchanged. Per-source locks serialize
all writers; early nod screening populates the same measurement cache without
altering the caption/quality journals or their accounting.
"""
from pathlib import Path
import importlib.metadata,json,subprocess,os
from listen_benchmark.j_annotation import locked,read,write
from listen_benchmark.j_visual_gate import measure_job
from listen_visual_quality.io import canonical_hash


def guarded_measure_job(work,jid):
    work=Path(work)
    with locked(work/'measurement_locks'/f'{jid}.lock'):
        return measure_job(work,jid)


def ensure_nod_quality(config,job_path):
    work=Path(config['work']);cfg=read(work/'config.json')
    if cfg['contract_sha256']!=config['contract_sha256']:raise ValueError('Nod visual quality contract mismatch')
    expected=cfg.get('contract',{}).get('packages',{})
    if expected and any(importlib.metadata.version(n)!=v for n,v in expected.items()):
        # The nod runtime may use a different PyAV build. Never mix that build
        # into frozen visual measurements; run in the verified CPU environment.
        root=Path(__file__).resolve().parents[2]
        command=[config['runtime_python'],str(root/'scripts/listen_j_visual_quality_early.py'),'--config',str(Path(cfg['omni_work'])/'config.json'),'--job',str(job_path)]
        proc=subprocess.run(command,capture_output=True,text=True,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GLOG_minloglevel='2'))
        return json.loads(proc.stdout)

    job_path=Path(job_path).resolve();job=read(job_path);jid=job['job_id']
    omni=Path(cfg['omni_work']);link=omni/'jobs'/f'{jid}.json'
    with locked(omni/'producer-step.lock'):
        if link.exists():
            if link.resolve()!=job_path:raise ValueError('Conflicting J source')
        else:link.symlink_to(job_path)
    # Early receipts deliberately do not complete or increment another queue.
    with locked(work/'measurement_locks'/f'{jid}.lock'):
        early=work/'early/completed'/f'{jid}.json'
        receipt=read(early) if early.exists() else measure_job(work,jid)
        if receipt['source_result_sha256']!=job['source_result_sha256'] or receipt['contract_sha256']!=config['contract_sha256']:raise ValueError('Early quality receipt mismatch')
        out={d['sample_id']:d for d in receipt['decisions']}
        if len(out)!=len(job['rows']) or set(out)!={r['sample_id'] for r in job['rows']}:raise ValueError('Incomplete early quality receipt')
        for row in job['rows']:
            if out[row['sample_id']]['source_sha256']!=canonical_hash(row['source']):raise ValueError('Early quality row changed')
        if not early.exists():write(early,receipt)
        return out
