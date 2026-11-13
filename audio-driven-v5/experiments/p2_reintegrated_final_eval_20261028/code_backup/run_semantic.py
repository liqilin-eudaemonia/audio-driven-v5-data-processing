"""Bounded co-resident label integration; native validation, no training edits."""
import json, os, signal, subprocess, sys, time, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import p2_guided_chain.backend  # Resolve package before native script inserts scripts/.
sys.path.append(str(ROOT / 'scripts'))
from p2_restore_full50h import recover_job
from listen_benchmark.semantic_prompt import digest

def read(p):
    return json.loads(p.read_text())

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.partial')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')
    t.replace(p)

def train_safe():
    state = read(ROOT / 'experiments/p2_full50h_duration384_20261028/training_queue/state.json')
    job = state['jobs']['av_feature']
    return job['status'] == 'running' and job['stage'] == 1 and job['step'] < 1050

def stop(p):
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()

def main():
    import fcntl
    lock = (HERE / 'semantic.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = [json.loads(l) for l in (HERE / 'semantic/inputs.jsonl').read_text().splitlines()]
    assert len(jobs) == 24 and len({j['job_id'] for j in jobs}) == 24
    # Pilot plus two bounded shards. Failed shards are retried individually so
    # one overlong request cannot prevent other samples from receiving a proposal.
    groups = [jobs[:1], jobs[1:13], jobs[13:]]
    runs = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='7', PYTHONHASHSEED='0',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',
               PYTORCH_ALLOC_CONF='expandable_segments:True')
    deadline = time.time() + 1800
    while groups:
        if not train_safe() or time.time() > deadline:
            raise RuntimeError('Yield GPU before training handoff; rerun after resource review')
        group = groups.pop(0)
        idx = len(runs)
        work = HERE / 'semantic' / f'run_{idx:03d}'
        inputs = HERE / 'semantic' / f'input_{idx:03d}.jsonl'
        inputs.write_text(''.join(json.dumps(j, ensure_ascii=False) + '\n' for j in group))
        free = int(subprocess.check_output(['nvidia-smi', '--id=7', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < 80000:
            raise RuntimeError(f'Insufficient GPU headroom: {free} MiB')
        write(HERE / 'semantic/status.json', dict(status='running', shard=idx, shard_samples=len(group), gpu=7, pid=os.getpid(), started=time.time(), model_review='record_only'))
        cmd = ['/private/software/conda/envs/vllm/bin/python', str(ROOT / 'scripts/run_listener_semantic_prompt.py'), '--inputs', str(inputs), '--output', str(work), '--batch', '2', '--memory', '.46', '--attempts', '3']
        with (HERE / 'semantic' / f'run_{idx:03d}.log').open('a') as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                end = time.time() + 900
                while p.poll() is None:
                    time.sleep(5)
                    if not train_safe() or time.time() > min(end, deadline):
                        raise RuntimeError('Bounded semantic task yielded before training transition or deadline')
            finally:
                stop(p)
        runs.append((work, group))
        write(HERE / 'semantic/runs.json', [dict(path=str(r), job_ids=[j['job_id'] for j in g]) for r, g in runs])
        if p.returncode and len(group) > 1:
            groups = [[j] for j in group] + groups
        elif p.returncode:
            # A native data error may retain its original events. Infrastructure
            # errors must never be disguised as a completed annotation update.
            tail = (HERE / 'semantic' / f'run_{idx:03d}.log').read_text()[-8000:]
            if 'Context overflow; no truncation allowed' not in tail:
                raise RuntimeError(f'Native worker failed: {work}; inspect log')
    records, audits = [], []
    for job in jobs:
        matching = [r for r,g in runs if job['job_id'] in {j['job_id'] for j in g} and (r/'protocol.json').exists()]
        protocols = {str(r): read(r/'protocol.json') for r in matching}
        for r,g in runs:
            if r in matching and digest(g) != protocols[str(r)]['input_sha256']:
                raise ValueError('Worker input digest mismatch')
        _, events, audit, selected, reviews = recover_job(job, matching, protocols)
        original = job['record']
        audio = lambda es: sorted((e for e in es if e['modality']=='audio'), key=lambda e:e['event_id'])
        assert audio(events) == audio(original['events'])
        derived = dict(original, events=events, annotation_view='full50h_hard_check_with_original_fallback',
                       original_record_sha256=digest(original), recovery_policy='safe_components_else_verbatim_original; semantic_review_record_only')
        records.append(derived)
        audits.append(dict(audit, sample_id=original['sample_id'], selected_proposal=str(selected) if selected else None,
                           review_records=reviews, model_review='record_only', audio_unchanged=True,
                           original_record_sha256=digest(original)))
    expected = set(read(HERE/'sampling.json')['test_sample_ids'])
    assert {r['sample_id'] for r in records} == expected
    dest = HERE/'test24/integrated_labels.jsonl'
    dest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    write(HERE/'test24/annotation_audit.json', audits)
    write(HERE/'semantic/COMPLETE.json', dict(status='complete', samples=len(records), excluded=0,
        safe_visual_sources=sum(len(a['safe_source_ids']) for a in audits),
        original_visual_sources=sum(len(a['fallback_source_ids']) for a in audits),
        labels_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), model_review='record_only', audio_unchanged=True))
    write(HERE/'semantic/status.json', read(HERE/'semantic/COMPLETE.json'))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(HERE/'semantic/status.json', dict(status='failed', error=repr(exc), at=time.time()))
        raise
