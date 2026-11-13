"""CPU-only MediaPipe and native-pixel focus measurements; no experimental imports."""
import io
import math
import time
import importlib.util
from functools import lru_cache
from pathlib import Path
import numpy as np
from .rules import analyze_focus, propose_crop


@lru_cache(maxsize=1)
def backend(root):
    import cv2
    import av
    import mediapipe as mp
    cv2.setNumThreads(1)
    path = Path(root) / "related/visual_reaction_annotation/nod_headshake/face_detection/detect_face_multithread.py"
    spec = importlib.util.spec_from_file_location("visual_quality_vendor_face", path)
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    return vendor, cv2, av, mp


class SourceBytes(io.RawIOBase):
    """Seekable original TAR member bytes, including members larger than 2 GiB."""
    def __init__(self, source):
        self.file=open(source['archive'],'rb')
        self.offset=source['offset'];self.size=source['byte_size'];self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self, offset, whence=0):
        if whence not in (0,1,2):raise ValueError('Invalid whence')
        position=(0 if whence==0 else self.position if whence==1 else self.size)+offset
        if position<0:raise ValueError('Negative member seek')
        self.position=position;return position
    def read(self, size=-1):
        remaining=max(0,self.size-self.position)
        count=remaining if size<0 else min(size,remaining)
        self.file.seek(self.offset+self.position);data=self.file.read(count)
        self.position+=len(data);return data
    def close(self):
        self.file.close();super().close()


def raw_measure(result, width, height, c, t, source_t):
    r = dict(t=t, source_time=source_t, detected=bool(result.face_landmarks))
    if not r['detected']:
        return r
    points = np.array([[p.x * width, p.y * height] for p in result.face_landmarks[0]])
    x0, y0 = points.min(axis=0); x1, y1 = points.max(axis=0)
    area = (x1-x0)*(y1-y0)
    inter = max(0, min(x1,c['x']+c['width'])-max(x0,c['x'])) * max(0,min(y1,c['y']+c['height'])-max(y0,c['y']))
    outside = ((points[:,0]<c['x']) | (points[:,0]>c['x']+c['width']) |
               (points[:,1]<c['y']) | (points[:,1]>c['y']+c['height']))
    center_inside = c['x'] <= (x0+x1)/2 <= c['x']+c['width'] and c['y'] <= (y0+y1)/2 <= c['y']+c['height']
    r.update(bbox_native=[x0,y0,x1,y1], face_width_native=x1-x0, face_height_native=y1-y0,
             bbox_retained_fraction=inter/area if area>0 else None,
             landmarks_outside_fraction=float(outside.mean()), center_inside_crop=bool(center_inside))
    return r


def focus_metric(rgb,box):
    import cv2
    # Face-interior patch in native decoded pixels, before training resize/compression.
    x0,y0,x1,y1=box;w=x1-x0;h=y1-y0
    a,b=max(0,round(x0+.12*w)),max(0,round(y0+.10*h))
    c,d=min(rgb.shape[1],round(x1-.12*w)),min(rgb.shape[0],round(y1-.10*h))
    if c-a<48 or d-b<48:return None
    patch=rgb[b:d,a:c]
    gray=cv2.cvtColor(cv2.resize(patch,(128,128),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap=float(cv2.Laplacian(gray,cv2.CV_32F,ksize=1).var())
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    return dict(laplacian_variance=lap,gradient_energy=float((gx*gx+gy*gy).mean()),contrast_std=float(gray.std()),
                native_patch=[a,b,c,d],patch_short_side=min(c-a,d-b))


def measure_row(row, policy, root, model_path):
    vendor, cv2, av, mp = backend(root)
    sid=row["sample_id"]
    started=time.monotonic();s=row['source'];c=s['video_transform']['crop'];w,h=s['video_transform']['source_size']
    origin=round(s['source_start_s']*s['normalized_audio']['sample_rate'])/s['normalized_audio']['sample_rate'];duration=s['duration_s']
    context={'x':max(0,math.floor(c['x']-.25*c['width'])),'y':max(0,math.floor(c['y']-.25*c['height']))}
    context['width']=min(w,math.ceil(c['x']+1.25*c['width']))-context['x'];context['height']=min(h,math.ceil(c['y']+1.25*c['height']))-context['y']
    local=dict(c,x=c['x']-context['x'],y=c['y']-context['y']);samples=[];raw=[];n=math.ceil(duration*5-1e-9)
    try:
        with vendor.FaceLandmarker.create_from_options(vendor.build_options(str(model_path))) as full,\
             vendor.FaceLandmarker.create_from_options(vendor.build_options(str(model_path))) as expanded,\
             SourceBytes(s['source']) as source,av.open(source) as inp:
            stream=inp.streams.video[0];stream.codec_context.thread_count=1
            assert [stream.width,stream.height]==[w,h]
            inp.seek(int(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
            k=0;last=None;last_t=None
            def sample(rgb,t,k):
                timestamp=k*200
                fr=full.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)),timestamp)
                x,y,cw,ch=[context[q] for q in ('x','y','width','height')]
                cr=expanded.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb[y:y+ch,x:x+cw])),timestamp)
                fm=raw_measure(fr,w,h,c,k/5,origin+t);cm=raw_measure(cr,cw,ch,local,k/5,origin+t)
                if cm['detected']:
                    b=cm['bbox_native'];cm['bbox_native']=[b[0]+x,b[1]+y,b[2]+x,b[3]+y]
                use_context=not fm['detected'] or not fm.get('center_inside_crop',True)
                selected=dict(cm if use_context else fm);chosen=cr if use_context else fr
                selected.update(detector_input='expanded_context' if use_context else 'full_frame',full_frame_result=fm,context_result=cm)
                raw.append(selected)
                metric=focus_metric(rgb,selected['bbox_native']) if selected['detected'] else None
                q={'t':k/5,'source_time':origin+t,'measured':metric is not None,'raw_detected':selected['detected']}
                if metric:q.update(metric)
                if chosen.facial_transformation_matrixes:
                    pitch,roll,yaw=vendor.matrix_to_euler_degrees(chosen.facial_transformation_matrixes[0]);q.update(pitch=pitch,yaw=yaw,roll=roll)
                if selected['detected']:q['bbox_native']=selected['bbox_native']
                samples.append(q)
            for f in inp.decode(stream):
                t=float(f.pts*f.time_base)-origin
                if t<0:continue
                if t>=duration:break
                if k>=n:break
                if k/5>t+1e-10:
                    # Retain a potential last frame only if it could be the last sample.
                    last=f;last_t=t;continue
                rgb=f.to_ndarray(format='rgb24');last=f;last_t=t
                while k<n and k/5<=t+1e-10:sample(rgb,t,k);k+=1
            if last is None or duration-last_t>=.1 and k<n:raise ValueError('Incomplete original support')
            while k<n:sample(last.to_ndarray(format='rgb24'),last_t,k);k+=1
        assert len(samples)==n and samples[0]['source_time']-origin<.1
        result={'id':sid,'source':s,'status':'ok','focus_samples':samples,'raw_measurements':raw,
                'focus':analyze_focus(samples,duration,policy),'crop':propose_crop(raw,s,policy),'context_rect':context,
                'elapsed_s':time.monotonic()-started,'production_applied':False,'training_admitted':False}
    except Exception as e:
        result={'id':sid,'source':s,'status':'error','error':repr(e),'elapsed_s':time.monotonic()-started,'production_applied':False}
    return result
