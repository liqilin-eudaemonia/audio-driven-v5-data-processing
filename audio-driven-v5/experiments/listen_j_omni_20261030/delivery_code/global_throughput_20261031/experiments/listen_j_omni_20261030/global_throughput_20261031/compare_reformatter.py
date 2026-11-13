import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,time,threading,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1];sys.path.insert(0,str(R/'src'))
from listen_benchmark.full_raw import open_source
from listen_benchmark.frame_crop import crop_original_frame,crop_original_frame_reused
records=[]
for index in [0,1]:
    row=json.loads((W/'input_speed_v1_20261031/shared_replay_guarded'/str(index)/'row.json').read_text())
    source=row['source'];outputs=[]
    for arm in [0,1,1,0]:
        stop=threading.Event();thread_counts=[]
        def watch():
            while not stop.wait(.005):
                info=dict(x.split(':',1) for x in Path('/proc/self/status').read_text().splitlines() if ':' in x)
                thread_counts.append(int(info['Threads']))
        monitor=threading.Thread(target=watch);monitor.start();start=time.monotonic();c0=time.process_time();h=hashlib.sha256();count=0
        try:
            with open_source(source['source']) as (container,reader):
                stream=container.streams.video[0];stream.codec_context.thread_count=1;origin=source['source_start_s']
                container.seek(int(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
                for f in container.decode(stream):
                    if float(f.pts*f.time_base)<origin:continue
                    im=(crop_original_frame_reused if arm else crop_original_frame)(f,source['video_transform'])
                    h.update(im.tobytes());count+=1
                    if count>=25:break
        finally:stop.set();monitor.join()
        result=dict(arm=arm,frames=count,seconds=time.monotonic()-start,cpu_seconds=time.process_time()-c0,
                    peak_threads=max(thread_counts,default=0),sha256=h.hexdigest())
        outputs.append(result);print(json.dumps(result),flush=True)
    assert len({x['sha256'] for x in outputs})==1
    records.append(dict(sample_id=row['sample_id'],phases=outputs))
(D/'reformatter_probe.json').write_text(json.dumps(records,indent=2))
