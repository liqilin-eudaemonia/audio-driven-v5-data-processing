"""Reuse the running original test24 task, integrate the other24, publish full48."""
import json,time,sys,subprocess,os
from pathlib import Path
from eval_common import HERE,ROOT,write,read,sha
sys.path.insert(0,str(ROOT/'src'))

def safe_start():
    s=read(ROOT/'experiments/p2_full50h_duration384_20261028/training_queue/state.json')
    running=[j for j in s['jobs'].values() if j['status']=='running']
    if not any(j['stage']==1 and j['step']<800 for j in running):return False
    free=int(subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    return free>=80000

def main():
    import fcntl
    lock=(HERE/'semantic48.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    first=HERE/'semantic/COMPLETE.json';remaining=HERE/'semantic_remaining24'
    while not first.exists():
        write(HERE/'semantic48_status.json',dict(phase='waiting_running_original_test24',total_samples=48,at=time.time()))
        if read(HERE/'semantic/status.json')['status']=='failed':raise RuntimeError('Original test24 semantic task failed; inspect semantic/status.json')
        time.sleep(10)
    import run_semantic as native
    native.HERE=remaining
    # Preserve original native model/recovery code. Permit either current S1 arm;
    # release the small co-resident task before any training stage transition.
    def still_safe():
        s=read(ROOT/'experiments/p2_full50h_duration384_20261028/training_queue/state.json')
        return any(j['status']=='running' and j['stage']==1 and j['step']<1050 for j in s['jobs'].values())
    native.train_safe=still_safe
    attempts=0
    while not (remaining/'semantic/COMPLETE.json').exists():
        while not safe_start():
            write(HERE/'semantic48_status.json',dict(phase='waiting_safe_gpu_window_for_remaining24',completed_samples=24,total_samples=48,at=time.time()));time.sleep(10)
        write(HERE/'semantic48_status.json',dict(phase='integrating_remaining24',completed_samples=24,total_samples=48,at=time.time()))
        try:native.main()
        except RuntimeError as exc:
            if 'Yield GPU' in str(exc) or 'yielded before training' in str(exc):
                time.sleep(20);continue
            attempts+=1
            if attempts>=3:raise
            time.sleep(30)
    records=[];audits=[]
    for root in [HERE,remaining]:
        marker=read(root/'semantic/COMPLETE.json')
        if marker['labels_sha256']!=sha(root/'test24/integrated_labels.jsonl'):raise ValueError('Label hash differs')
        records.extend(json.loads(l) for l in (root/'test24/integrated_labels.jsonl').read_text().splitlines())
        audits.extend(read(root/'test24/annotation_audit.json'))
    ids=[o['sample_id'] for o in read(HERE/'test48/observations.json')];by={r['sample_id']:r for r in records}
    assert len(records)==len(by)==48 and set(by)==set(ids)
    path=HERE/'test48/integrated_labels.jsonl';path.write_text(''.join(json.dumps(by[s],ensure_ascii=False)+'\n' for s in ids))
    write(HERE/'test48/annotation_audit.json',audits)
    write(HERE/'test48/COMPLETE.json',dict(status='complete',samples=48,excluded=0,labels_sha256=sha(path),original_test24_subset=read(HERE/'sampling.json')['test_sample_ids'],review_policy='record_only'))
    write(HERE/'semantic48_status.json',dict(phase='complete',completed_samples=48,total_samples=48))

if __name__=='__main__':
    try:main()
    except Exception as exc:write(HERE/'semantic48_status.json',dict(phase='failed',error=repr(exc)));raise
