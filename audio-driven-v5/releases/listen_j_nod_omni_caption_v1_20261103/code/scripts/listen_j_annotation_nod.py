#!/usr/bin/env python3
"""CPU GLARE recall supplement on exact J crops, with temporary 25 Hz media."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import bisect
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import JobQueue,acquire,read,write,result_path


def render_video(row,directory,fast_crop=False,decoded_dir=None):
    """Same sampling/PIL crop/x264 parameters as export_listen_debug50_media."""
    import av
    from listen_benchmark.full_raw import open_source
    from listen_labeling.media import crop_resize
    from listen_benchmark.frame_crop import crop_original_frame
    source=row['source'];rate=source['normalized_audio']['sample_rate']
    origin=round(source['source_start_s']*rate)/rate;duration=source['duration_s']
    images=[];times=[]
    if decoded_dir is not None:
        from listen_benchmark.shared_visual_decode import load_native
        from PIL import Image
        rgb,times,reader=load_native(decoded_dir,source)
        images=[Image.fromarray(frame) for frame in rgb]
    else:
        with open_source(source['source']) as (container,reader):
            stream=container.streams.video[0];stream.codec_context.thread_count=1
            assert [stream.width,stream.height]==source['video_transform']['source_size']
            container.seek(int(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
            for frame in container.decode(stream):
                assert frame.pts is not None
                t=float(frame.pts*frame.time_base)-origin
                if t<0:continue
                if t>=duration:break
                images.append(crop_original_frame(frame,source['video_transform']) if fast_crop
                              else crop_resize(frame.to_image(),source['video_transform']));times.append(t)
    if not images or times[0]>=.1 or duration-times[-1]>=.1:raise ValueError('Incomplete source frame support')
    n=math.ceil(duration*25-1e-9)
    indices=[min(bisect.bisect_left(times,i/25),len(times)-1) for i in range(n)]
    movie_path=directory/'video.mp4'
    with av.open(str(movie_path),'w') as movie:
        stream=movie.add_stream('libx264',rate=25);stream.width=512;stream.height=512;stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'fast','threads':'2'}
        for i,k in enumerate(indices):
            frame=av.VideoFrame.from_image(images[k]);frame.pts=i;frame.time_base=Fraction(1,25)
            for packet in stream.encode(frame):movie.mux(packet)
        for packet in stream.encode():movie.mux(packet)
    meta={'sample_id':row['sample_id'],'source_start_s':origin,'duration_s':duration,'fps':25,'frames':n,
        'video_transform':source['video_transform'],'source_video':source['source'],'reader':reader,
        'source_frame_times_s':times,'output_to_source_indices':indices,
        'video_sha256':hashlib.sha256(movie_path.read_bytes()).hexdigest(),
        'frame_selection':'first source frame at/after each 25Hz sample; last frame reused only at end'}
    return movie_path,meta


def annotate(row,face,nod,parameters,model,intervals,work,fast_crop=False,decoded_dir=None):
    from listen_visual_native8_nod import decode_slot
    with tempfile.TemporaryDirectory(prefix='listen-j-nod-',dir='/dev/shm') as tmp:
        directory=Path(tmp)
        # Same host-wide read budget as the VLM stream.
        tick=time.monotonic()
        with decode_slot(work/'resources',read(work/'selected.json')['decode_slots']):
            acquired=time.monotonic();video,meta=render_video(row,directory,fast_crop=fast_crop,decoded_dir=decoded_dir)
        rendered=time.monotonic()
        pkl=directory/'pose.pkl';extracted=face.process_video(video,pkl,str(model));faced=time.monotonic()
        if extracted['frames']!=meta['frames']:raise ValueError('Landmark frame count mismatch')
        detected,pitch,yaw=nod.load_pose_from_pkl(pkl)
        result={'sample_id':row['sample_id'],'duration_s':row['source']['duration_s'],'fps':25,
            'frames':meta['frames'],'source':row['source'],'media':meta,'profile':'recall','parameters':parameters,
            'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
            'pose':{'detected':detected.astype(int).tolist(),'pitch':pitch.tolist(),'yaw':yaw.tolist()},
            'input_execution':{'fast_frame_crop':fast_crop},
            'performance':{'decode_lock_wait_s':acquired-tick,'render_s':rendered-acquired,'face_s':faced-rendered},
            'completed_at':time.time(),'training_admitted':False}
        if not detected.any():
            # A detector with no face observations supplies no nod evidence.
            # Record missing evidence explicitly, never an empty negative label.
            return dict(result,status='unavailable',reason='no_detected_face',scores=None,events=None)
        scores=nod.compute_nodding_intensity(detected,pitch,yaw,argparse.Namespace(**parameters))
        events=intervals(scores,detected,25,row['source']['duration_s'])
        return dict(result,status='ok',
            scores=[float(x) for x in scores],
            events=[dict(e,event_id=f'nod-{i+1:04d}',type='nodding',modality='visual',description='Head nodding',
                           source='GLARE_nodding_v2_recall',status='candidate') for i,e in enumerate(events)],
            )


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--slot',type=int,required=True)
    p.add_argument('--once-job');args=p.parse_args();work=args.work.resolve();name=f'nod-{args.slot}'
    owner=acquire(work/'workers'/f'{name}.lock',False)
    if owner is None:return
    import cv2
    cv2.setNumThreads(1)
    from run_glare_nodding_pilot import module,FACE,NOD,MODEL,PROFILES,intervals
    face=module(f'j_face_{args.slot}',FACE);nod=module(f'j_nod_{args.slot}',NOD)
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set())
    state={'pid':os.getpid(),'slot':args.slot,'stage':'starting','completed_clips':0,'errors':0,
           'unavailable_clips':0,'visual_excluded_clips':0,'visual_review_clips':0,'revision':6,'started_at':time.time()}
    def pulse():
        while not stop.wait(10):write(work/'workers'/f'{name}.json',dict(state,heartbeat=time.time()))
    threading.Thread(target=pulse,daemon=True).start();queue=JobQueue(work,'nod')
    try:
        while not stop.is_set():
            claim=queue.claim(name)
            if claim is None:state['stage']='waiting_for_data';time.sleep(3);continue
            job_id,handle=claim;job=read(work/'jobs'/f'{job_id}.json');failed=[]
            state.update(stage='extracting',job=job_id)
            try:
                quality=None;quality_config=read(work/'config.json').get('visual_quality_gate')
                if quality_config:
                    from listen_benchmark.j_visual_quality_schedule import ensure_nod_quality
                    state['stage']='visual_quality_screening'
                    quality=ensure_nod_quality(quality_config,work/'jobs'/f'{job_id}.json')
                    state['stage']='extracting'
                for row in job['rows']:
                    if stop.is_set():break
                    target=result_path(work,'nod',row['sample_id'])
                    if target.exists():continue
                    try:
                        if quality is not None and quality[row['sample_id']]['decision']!='retain':
                            d=quality[row['sample_id']]
                            write(target,dict(sample_id=row['sample_id'],source=row['source'],status='visual_'+d['decision'],visual_quality=d,events=None,training_admitted=False))
                            state['visual_excluded_clips' if d['decision']=='exclude' else 'visual_review_clips']+=1
                            state['last_progress_at']=time.time()
                            continue
                        live_config=read(work/'config.json')
                        fast_crop=live_config.get('nod_fast_frame_crop',False)
                        state['fast_frame_crop_active']=fast_crop
                        input_row=row;crop_provenance=None
                        if live_config.get('corrected_crop_omni_work_v2'):
                            from listen_benchmark.j_omni_crop import effective_row
                            omni_config=read(Path(live_config['corrected_crop_omni_work_v2'])/'config.json')
                            applied,crop_provenance=effective_row(row,omni_config,decision=quality[row['sample_id']])
                            if crop_provenance['coordinates_changed']:input_row=applied
                            state['crop_revision']=crop_provenance['revision']
                        result=annotate(input_row,face,nod,PROFILES['recall'],MODEL,intervals,work,fast_crop=fast_crop)
                        if crop_provenance is not None:
                            result.update(original_source=row['source'],crop_provenance=crop_provenance)
                        write(target,result);state['completed_clips']+=1
                        state['unavailable_clips']+=result['status']=='unavailable'
                        state['last_progress_at']=time.time()
                    except Exception as e:
                        state['errors']+=1;failed.append(row['sample_id'])
                        write(work/'nod/errors'/f"{row['sample_id']}.json",{'error':repr(e),'at':time.time()})
                missing=[r['sample_id'] for r in job['rows'] if not result_path(work,'nod',r['sample_id']).exists()]
                # Graceful reconfiguration is not an input/inference failure.
                if missing and stop.is_set():pass
                elif missing:queue.finish(job_id,error={'missing':missing})
                else:queue.finish(job_id,receipt={'job_id':job_id,'clips':len(job['rows']),
                    'source_result_sha256':job['source_result_sha256'],'completed_at':time.time()})
            finally:handle.close()
            if args.once_job:break
    finally:
        stop.set();write(work/'workers'/f'{name}.json',dict(state,heartbeat=time.time()));owner.close()


if __name__=='__main__':main()
