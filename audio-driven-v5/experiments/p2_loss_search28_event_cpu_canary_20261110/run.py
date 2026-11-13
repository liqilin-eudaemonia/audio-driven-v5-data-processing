"""Two actual historical pairs through native event preparation/candidates on CPU.

No inference/scoring or candidate-weight comparison. Training GPUs are untouched.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from p2_event_presence.contracts import rows,save_rows


def main():
    evaluation=ROOT/'experiments/p2_loss_search28_evaluation_20261110'
    fixture=ROOT/'experiments/p2_loss_search28_eval_cpu_bind_audit_20261110'
    selected=read(evaluation/'tuning64.json')
    ids={next(r['id']for r in selected if r['audio_group']==group)for group in ('silent','short')}
    arm=read(fixture/'protocol.json')['arms'][0]['name']
    manifest=[]
    for f in sorted((fixture/'metric_manifests'/arm/'seamless').glob('*.jsonl')):
        manifest.extend(r for r in rows(f)if r['id']in ids)
    assert len(manifest)==2 and all(Path(r['generated_audio']).is_file()for r in manifest)
    save_rows(WORK/'pairs.jsonl',manifest)
    cfg=read(ROOT/'configs/p2_event_presence_v1.json');events=WORK/'events'
    entry=ROOT/'scripts/p2_event_presence.py'
    def command(args,python,log):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'),
                 OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        lib=Path(python).parent.parent/'lib'
        if lib.is_dir():env['LD_LIBRARY_PATH']=str(lib)+':'+env.get('LD_LIBRARY_PATH','')
        with log.open('a')as stream:
            subprocess.run([python,str(entry),*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
    if not(events/'contract.json').exists():
        command(['bind','--manifest',str(WORK/'pairs.jsonl'),'--work',str(events),'--method','historical_cpu_canary'],sys.executable,WORK/'bind.log')
    command(['prepare','--work',str(events)],cfg['media_python'],WORK/'prepare.log')
    records=rows(events/'prepared_pairs.jsonl')
    assert len(records)==2
    for row in records:
        for side,m in row['prepared_media'].items():
            assert not m['errors'],(row['id'],side,m['errors'])
            assert m['audio']and m['visual25']and m['windows']
            assert sha(m['audio'])==m['audio_file_sha256']
    command(['candidates','--work',str(events)],sys.executable,WORK/'candidates.log')
    candidate_report=read(events/'CANDIDATES_COMPLETE.json')
    assert candidate_report['failed']==0,candidate_report
    details=[]
    for row in records:
        for side,m in row['prepared_media'].items():
            c=read(Path(m['folder'])/'candidates.json')
            details.append(dict(id=row['id'],side=side,frame_coverage=c['frame_coverage'],
                candidate_cues=len(c['cues']),full_silence=m['full_silence']))
    atomic_json(WORK/'ACCEPTED.json',dict(success=True,at=time.time(),pairs=2,media=candidate_report['media'],
        source_fixture=str(fixture),own_audio_prepared=True,no_gpu=True,no_detector_inference=True,
        no_search_candidate_score=True,details=details,prepared_sha256=sha(events/'prepared_pairs.jsonl'),
        candidate_report_sha256=sha(events/'CANDIDATES_COMPLETE.json')))
    print(json.dumps(read(WORK/'ACCEPTED.json'),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
