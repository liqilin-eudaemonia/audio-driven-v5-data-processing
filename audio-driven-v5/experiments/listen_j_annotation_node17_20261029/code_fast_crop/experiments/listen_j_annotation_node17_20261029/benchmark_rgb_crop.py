#!/usr/bin/env python3
"""CPU-only experiment; never replaces production decoding or inference."""
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')


def main():
    os.nice(10)
    import numpy as np
    from PIL import Image
    from listen_benchmark.full_raw import open_source
    from listen_benchmark.j_annotation import read,write
    from listen_labeling.media import crop_resize,validate_transform
    from listen_visual_native8_nod import decode_slot

    def current(frame,t):return crop_resize(frame.to_image(),t)
    def sliced(frame,t):
        validate_transform(t)
        if [frame.width,frame.height]!=t['source_size']:raise ValueError('Source dimensions mismatch')
        rgb=frame.to_ndarray(format='rgb24');c=t['crop']
        return Image.fromarray(rgb[c['y']:c['y']+c['height'],c['x']:c['x']+c['width']]).resize(
            (512,512),resample=Image.Resampling.BICUBIC)

    pairs=['improvised_V03_S1355_I00000132','naturalistic_V03_S0444_I00000060',
           'improvised_V00_S0061_I00000581','naturalistic_V02_S5078_I00000281']
    cases=[];started=time.time()
    for pair in pairs:
        row=read(WORK/'jobs'/f'{pair}.json')['rows'][0];source=row['source'];t=source['video_transform']
        frames=[]
        with decode_slot(WORK/'resources',64):
            with open_source(source['source']) as (container,reader):
                stream=container.streams.video[0];stream.codec_context.thread_count=1
                origin=round(source['source_start_s']*48000)/48000
                container.seek(int(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
                for frame in container.decode(stream):
                    if frame.pts is None:raise ValueError('Missing PTS')
                    if float(frame.pts*frame.time_base)<origin:continue
                    frames.append(frame)
                    if len(frames)==8:break
        if len(frames)!=8:raise ValueError('Insufficient source frames')
        times={'current':[],'sliced':[]};hashes=[]
        for frame in frames:
            a,b=current(frame,t),sliced(frame,t)
            if not np.array_equal(np.asarray(a),np.asarray(b)):raise ValueError('RGB pixels changed')
            encoded=[]
            for im in [a,b]:
                buf=io.BytesIO();im.save(buf,format='JPEG',quality=85);encoded.append(buf.getvalue())
            if encoded[0]!=encoded[1]:raise ValueError('JPEG transport changed')
            hashes.append({'pts':frame.pts,'rgb_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                           'jpeg_sha256':hashlib.sha256(encoded[0]).hexdigest()})
            for repeat in range(3):
                order=[('current',current),('sliced',sliced)]
                if repeat%2:order.reverse()
                for name,fn in order:
                    tick=time.perf_counter();result=fn(frame,t);times[name].append(time.perf_counter()-tick)
        case={'sample_id':row['sample_id'],'transform':t,'reader':reader,'frames':hashes,
              'seconds':times,'median_seconds':{k:statistics.median(v) for k,v in times.items()},
              'total_seconds':{k:sum(v) for k,v in times.items()},'pixel_and_jpeg_exact':True}
        cases.append(case);print(json.dumps({k:case[k] for k in ['sample_id','median_seconds','total_seconds']}),flush=True)
        del frames
    write(WORK/'rgb_crop_benchmark.json',{'at':time.time(),'wall_s':time.time()-started,'cases':cases,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'4 source clips, 8 frames each, 3 alternating repeats; isolated crop CPU cost only',
        'production_changed':False})


if __name__=='__main__':main()
