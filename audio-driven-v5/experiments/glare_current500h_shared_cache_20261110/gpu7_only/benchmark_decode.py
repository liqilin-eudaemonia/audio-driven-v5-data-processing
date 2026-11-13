"""Same real groups, frozen numerical pipeline, isolated execution comparison."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
ROOT=WORK.parents[1]
OUT=HERE/'decode_process_benchmark'
PY=ROOT/'compare/.envs/avatarforcing/bin/python'


def run():
    OUT.mkdir(exist_ok=True)
    records=[]
    for name,decode,detectors,inflight,process in [
        ('A_threads8',8,8,8,False),('B_process16',16,16,32,True)]:
        path=OUT/name;path.mkdir(exist_ok=False)
        shutil.copyfile(WORK/'protocol.json',path/'protocol.json')
        os.link(WORK/'groups.sqlite',path/'groups.sqlite')
        args=[str(PY),str(ROOT/'scripts/glare_visual_runtime.py'),'--gpu','7','--all-partitions',
              '--benchmark-offset','4500','--limit','48','--work',str(path),'--decode-threads',str(decode),
              '--detector-services',str(detectors),'--inflight-groups',str(inflight),'--profile-cpu-stages']
        if process:args+=['--process-decode']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=':'.join(str(x) for x in [
                ROOT/'experiments/glare_current500h_nonprosody_cache_20261110/runtime_deps',
                ROOT/'compare/.envs/glare-deps',ROOT/'src',ROOT/'scripts']))
        with (path/'run.log').open('a') as log:
            proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            (OUT/'RUNNING.json').write_text(json.dumps(dict(pid=proc.pid,round=name,at=time.time())))
            code=proc.wait(timeout=1800)
        assert code==0,(name,code)
        state=json.loads((path/'workers/gpu7.json').read_text())
        assert state['stage']=='finished' and state['failed']==0 and state['completed']==state['total_pairs'],state
        records.append(dict(name=name,state=state))
        (OUT/'ROUNDS.json').write_text(json.dumps(records,indent=2))
    # Compare decoded pixels and all saved feature/mask arrays, independently
    # of the execution knobs. State timings exclude initial GPU model loading.
    import numpy as np
    import torch
    left,right=[OUT/x for x in ['A_threads8','B_process16']]
    checks=[]
    db=sqlite3.connect(WORK/'groups.sqlite')
    ordinals=[o for (s,) in db.execute('SELECT ordinals FROM groups ORDER BY gid LIMIT 48 OFFSET 4500') for o in json.loads(s)]
    db.close()
    parent=ROOT/'experiments/glare_current500h_nonprosody_cache_20261110'
    db=sqlite3.connect(parent/'jobs.sqlite');db.row_factory=sqlite3.Row
    for ordinal in ordinals:
        job=dict(db.execute('SELECT * FROM jobs WHERE ordinal=?',(ordinal,)).fetchone());sid=job['sid']
        prefix=Path('cache')/hashlib.sha256(sid.encode()).hexdigest()[:2]/sid
        for kind in ['lia','audio','statistics']:
            receipts=[json.loads((base/str(prefix.with_name(sid+'.'+kind+'.json'))).read_text()) for base in [left,right]]
            payloads=[torch.load(r['payload'],map_location='cpu',weights_only=True) for r in receipts]
            assert set(payloads[0])==set(payloads[1])
            for k,v in payloads[0].items():assert torch.equal(v,payloads[1][k]),(sid,kind,k)
        arrays=[dict(np.load(base/str(prefix.with_name(sid+'.reaction.npz')))) for base in [left,right]]
        assert set(arrays[0])==set(arrays[1])
        for k,v in arrays[0].items():
            assert np.array_equal(v,arrays[1][k],equal_nan=True) if np.issubdtype(v.dtype,np.number) else np.array_equal(v,arrays[1][k]),(sid,k)
        metadata=[json.loads((base/str(prefix.with_name(sid+'.reaction.json'))).read_text()) for base in [left,right]]
        assert metadata[0]['measurement']['detector_video_sha256']==metadata[1]['measurement']['detector_video_sha256']
        lia_meta=[json.loads((base/str(prefix.with_name(sid+'.lia.json'))).read_text()) for base in [left,right]]
        assert lia_meta[0].get('pts')==lia_meta[1].get('pts')
        checks.append(sid)
    db.close()
    result=dict(passed=True,pairs=len(checks),groups=48,offset=4500,rounds=records,
        speedup=records[0]['state']['elapsed_s']/records[1]['state']['elapsed_s'],
        exact_tensors_and_reactions=True,production_video_sha_equal=True,lia_pts_equal=True,checks=checks,
        caveat='Two sequential trials on shared GPU7; production jobs remain active. Live throughput will also be checked.')
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rounds','checks']}),flush=True)


if __name__=='__main__':run()
