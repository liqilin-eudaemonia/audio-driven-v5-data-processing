#!/usr/bin/env python3
"""Prepare the same historical 50 clips for the existing 25 Hz GLARE recall tool."""
import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    parent=ROOT/'experiments/listen_native_caption_clean50_20261030/text8'
    rows=json.loads((parent/'manifest.json').read_text());assert len(rows)==50
    def one(row):
        import av
        from listen_labeling.media import crop_resize
        sid=row['sample_id'];duration=row['duration_s'];origin=row['source_reference']['source_start_s']
        prior=ROOT/'experiments/p2_features_full_crop384_20261025/media'/sid
        if (prior/'media.json').exists():
            meta=json.loads((prior/'media.json').read_text())
            assert meta['sample_id']==sid and meta['video_transform']==row['video_transform']
            assert abs(meta['source_start_s']-origin)<1e-6 and abs(meta['duration_s']-duration)<1e-6
            assert meta['fps']==25
            video=prior/'video.mp4';mode='reuse_existing_exact_clip_crop'
        else:
            folder=run/'media25'/sid;folder.mkdir(parents=True,exist_ok=True);video=folder/'video.mp4'
            assert sha(row['target_video'])==row['media_hashes']['target_video']
            images=[];times=[]
            with av.open(row['target_video']) as c:
                stream=c.streams.video[0];stream.codec_context.thread_count=1
                for frame in c.decode(stream):
                    assert frame.pts is not None
                    t=float(frame.pts*frame.time_base)
                    if t>=duration:break
                    images.append(crop_resize(frame.to_image(),row['video_transform']));times.append(t)
            assert images and abs(times[0])<1e-6 and duration-times[-1]<.1
            n=math.ceil(duration*25-1e-9)
            indices=[min(bisect.bisect_left(times,i/25),len(times)-1) for i in range(n)]
            with av.open(str(video),'w') as c:
                stream=c.add_stream('libx264',rate=25);stream.width=512;stream.height=512;stream.pix_fmt='yuv420p'
                stream.options={'crf':'18','preset':'fast','threads':'2'}
                for i,k in enumerate(indices):
                    frame=av.VideoFrame.from_image(images[k]);frame.pts=i;frame.time_base=Fraction(1,25)
                    for packet in stream.encode(frame):c.mux(packet)
                for packet in stream.encode():c.mux(packet)
            meta=dict(sample_id=sid,source_start_s=origin,duration_s=duration,fps=25,frames=n,
                      video_transform=row['video_transform'],source_video=row['target_video'],
                      source_video_sha256=row['media_hashes']['target_video'],source_frame_times_s=times,
                      output_to_source_indices=indices,video_sha256=sha(video),
                      frame_selection='first source frame at/after each 25Hz sample; last frame reused only at end')
            atomic_json(folder/'media.json',meta);mode='render_historical_clip_same_crop_25fps'
        return dict(id=sid,video=str(video),duration=duration,crop=row['video_transform'],source_start_s=origin,mode=mode)
    with ThreadPoolExecutor(max_workers=2) as pool: samples=list(pool.map(one,rows))
    atomic_json(run/'media_index.json',dict(samples=samples))
    atomic_json(run/'sample_ids.json',dict(ids=[r['sample_id'] for r in rows]))
    print(json.dumps(dict(samples=len(samples),reused=sum(s['mode'].startswith('reuse') for s in samples))))


if __name__=='__main__': main()
