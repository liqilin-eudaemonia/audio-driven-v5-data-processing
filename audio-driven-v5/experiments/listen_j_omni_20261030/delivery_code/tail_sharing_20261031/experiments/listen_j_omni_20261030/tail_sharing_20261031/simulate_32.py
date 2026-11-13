"""32-worker CPU-only scheduler stress check, isolated from production results."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json,os,sys,tempfile,threading,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1]
sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write,JobQueue,result_path
from listen_benchmark.j_omni_stream import CrossSourceStream

with tempfile.TemporaryDirectory(prefix='omni-tail-32-') as d,ThreadPoolExecutor(8) as preparers,ThreadPoolExecutor(32) as consumers:
    w=Path(d);rows=[dict(sample_id=f'clip-{i}',source=dict(pair='source',duration_s=2)) for i in range(512)]
    write(w/'jobs/source.json',dict(job_id='source',source_result_sha256='fixed',rows=rows))
    write(w/'job_origins/source.json',dict(old_work='immutable-origin',source_result_sha256='fixed'))
    (w/'ready.jsonl').write_text('{"job_id":"source"}\n')
    write(w/'producer_complete.json',{});write(w/'tail_sharing/ENABLED.json',{'protocol':'clip-lock-tail-v1'})
    q=JobQueue(w,'caption');seen=[];origins=[];guard=threading.Lock()
    cfg=dict(pipeline_id='test',tail_sharing=dict(enabled=True,worker_count=32,packet_size=8,poll_seconds=1))
    streams=[]
    for i in range(32):
        streams.append(CrossSourceStream(w,q,f'worker-{i}',cfg,SimpleNamespace(pool=preparers),'r',
            dict(prefetch=2),None,dict(pid=os.getpid(),visual_excluded_clips=0,visual_review_clips=0),threading.Event()))
    def publish(s):
        s.tail.limit(4);s.tail.advertise(s.active)
        write(w/'workers'/f'{s.name}.json',dict(s.state,heartbeat=time.time()))
    def prepare(row,origin,*_):
        with guard:origins.append(origin)
        return row
    def consume(s):
        try:
            for row,f,_ in s:
                f.result();time.sleep(.015 if s.name=='worker-0' else .002)
                with guard:seen.append((s.name,row['sample_id']))
                write(result_path(w,'caption',row['sample_id']),dict(source=row['source'],pipeline_id='test',generation_checks_passed=True))
                s.persisted(row['sample_id']);publish(s)
        finally:s.close()
    with patch('listen_benchmark.j_omni.verified_job',side_effect=read),patch('listen_benchmark.j_omni_stream.prepare_call',side_effect=prepare):
        streams[0]._claim();publish(streams[0]);t=time.time()
        futures=[consumers.submit(consume,s) for s in streams]
        for f in futures:f.result(timeout=90)
    counts=Counter(sid for _,sid in seen);workers=Counter(name for name,_ in seen);state=read(q.path)
    assert len(seen)==len(counts)==512 and max(counts.values())==1
    assert state['completed']==1 and state['completed_clips']==512 and not state['inflight'] and not state['failed']
    assert set(origins)=={'immutable-origin'} and len(workers)>1
    out=dict(at=time.time(),passed=True,workers_launched=32,workers_produced=dict(workers),
             clips=512,source_receipts=1,duplicates=0,missing=0,source_identity_preserved=True,
             elapsed_s=time.time()-t,scope='CPU scheduler simulation; not GPU speed benchmark')
    write(D/'simulation_32.json',out);print(json.dumps(out))
