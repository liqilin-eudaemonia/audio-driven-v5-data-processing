#!/usr/bin/env python3
"""Authorized small-sample crop/focus spike; never changes production admission."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/visual_quality_pilot_20261030'))
import run as base
import json, math, pickle, subprocess, tempfile
from fractions import Fraction
np,cv2,av,mp=base.np,base.cv2,base.av,base.mp
OUT=Path(__file__).resolve().parent
WEB=ROOT/'data_previews/visual_quality_pilot_20261030/crop_focus'
PRIOR=ROOT/'data_previews/visual_quality_pilot_20261030/data.json'


def proposal(r):
    s=r['source'];c=s['video_transform']['crop'];w,h=s['video_transform']['source_size']
    rows=[x for x in r['raw_measurements'] if x['detected']]
    b=np.array([x['bbox_native'] for x in rows]);fh=np.median(b[:,3]-b[:,1]);fw=np.median(b[:,2]-b[:,0])
    # Measured face trajectory plus explicit, approximate head margins. No head segmentation claim.
    env=[float(b[:,0].min()-.12*fw),float(b[:,1].min()-.35*fh),float(b[:,2].max()+.12*fw),float(b[:,3].max()+.12*fh)]
    side=min(w,h,max(c['width'],math.ceil(max(env[2]-env[0],env[3]-env[1])/.92)))
    x=int(np.clip(round((env[0]+env[2])/2-side/2),0,w-side));y=int(np.clip(round((env[1]+env[3])/2-side/2),0,h-side))
    new=dict(x=x,y=y,width=side,height=side)
    return dict(old=c,new=new,estimated_head_envelope=env,source_size=[w,h],
                scale_ratio=c['width']/side,shift=[x-c['x'],y-c['y']],
                method='fixed square contains observed 5 fps face trajectory, with estimated margins: sides/chin 0.12 face dimension; top 0.35 face height; 4% outer margin; never shrink old crop',
                head_top_estimated=True)


def retained(box,c):
    x0,y0,x1,y1=box
    return max(0,min(x1,c['x']+c['width'])-max(x0,c['x']))*max(0,min(y1,c['y']+c['height'])-max(y0,c['y']))/max((x1-x0)*(y1-y0),1e-9)


def focus_metric(rgb,box):
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


def crop_rgb(rgb,c):
    return np.asarray(base.Image.fromarray(rgb[c['y']:c['y']+c['height'],c['x']:c['x']+c['width']]).resize((512,512),base.Image.Resampling.BICUBIC))


def process(r,i):
    sid=r['sample_id'];s=r['source'];p=proposal(r);old,new=p['old'],p['new'];w,h=p['source_size']
    folder=WEB/'media'/sid;folder.mkdir(parents=True,exist_ok=True)
    original=base.OUT/'context_media_metadata'/f'{sid}.json';meta=json.loads(original.read_text())
    ts=meta['source_frame_times_s'];raw=r['raw_measurements'];origin=meta['source_start_s'];duration=s['duration_s']
    with (base.OUT/'landmarks'/f'{sid}.pkl').open('rb') as f:old_frames=pickle.load(f)
    samples=[];n=len(ts);raw_width=round(w*512/h/2)*2
    with tempfile.TemporaryDirectory(prefix='crop-focus-spike-') as tn:
        temp=Path(tn)
        with base.SourceBytes(s['source']) as source,av.open(source) as inp,\
             av.open(str(folder/'candidate.mp4'),'w') as candidate, av.open(str(temp/'silent.mp4'),'w') as comparison:
            stream=inp.streams.video[0];stream.codec_context.thread_count=1
            inp.seek(int(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
            enc=[]
            for out,width in [(candidate,512),(comparison,raw_width+1024)]:
                e=out.add_stream('libx264',rate=25);e.width=width;e.height=512;e.pix_fmt='yuv420p';e.options={'crf':'18','preset':'fast','threads':'1'};enc.append(e)
            k=0
            for f in inp.decode(stream):
                t=float(f.pts*f.time_base)
                if k>=n:break
                if t+1e-7<ts[k]:continue
                rgb=f.to_ndarray(format='rgb24')
                while k<n and ts[k]<=t+1e-7:
                    if abs(ts[k]-t)>.05:raise ValueError('Changed source frame selection')
                    before,after=crop_rgb(rgb,old),crop_rgb(rgb,new)
                    small=cv2.resize(rgb,(raw_width,512),interpolation=cv2.INTER_AREA)
                    for c,color in [(old,(70,220,110)),(new,(60,185,255))]:
                        cv2.rectangle(small,(round(c['x']*raw_width/w),round(c['y']*512/h)),(round((c['x']+c['width'])*raw_width/w),round((c['y']+c['height'])*512/h)),color,2)
                    combined=np.concatenate([small,before,after],axis=1)
                    base.encode(enc[0],candidate,after,k);base.encode(enc[1],comparison,combined,k)
                    if k%5==0:
                        raw_r=raw[k//5];assert abs(raw_r['t']-k/25)<1e-6
                        metric=focus_metric(rgb,raw_r['bbox_native']) if raw_r['detected'] else None
                        sample=dict(t=k/25,source_time=t,measured=metric is not None,raw_detected=raw_r['detected'])
                        if metric:sample.update(metric)
                        pose=old_frames[k];sample.update(crop_detected=pose['detected'],pitch=pose.get('pitch'),yaw=pose.get('yaw'))
                        if raw_r['detected']:sample.update(bbox_native=raw_r['bbox_native'],original_frame_retained=retained(raw_r['bbox_native'],dict(x=0,y=0,width=w,height=h)))
                        samples.append(sample)
                        # Keep sampled frames for audit; generated from original data, no blur synthesis.
                        cv2.imwrite(str(folder/f'frame_{k:05d}.jpg'),cv2.cvtColor(combined,cv2.COLOR_RGB2BGR))
                        if metric:
                            a,b,c,d=metric['native_patch'];patch=rgb[b:d,a:c]
                            cv2.imwrite(str(folder/f'face_{k:05d}.jpg'),cv2.cvtColor(cv2.resize(patch,(256,256),interpolation=cv2.INTER_CUBIC),cv2.COLOR_RGB2BGR))
                    k+=1
            if k!=n:raise ValueError('Missing output frames')
            for st,out in zip(enc,[candidate,comparison]):
                for packet in st.encode():out.mux(packet)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','1','-i',str(temp/'silent.mp4'),'-i',str(base.WEB/r['media']),'-map','0:v','-map','1:a','-c','copy','-t',str(duration),'-movflags','+faststart',str(folder/'comparison.mp4')],check=True,timeout=60)
    base.face.process_video(folder/'candidate.mp4',OUT/'landmarks'/f'{sid}.pkl',str(base.MODEL))
    with (OUT/'landmarks'/f'{sid}.pkl').open('rb') as f:frames=pickle.load(f)
    assert len(frames)==n
    def geometry(fs):
        detected=[q['detected'] for q in fs];edge=[]
        for q in fs:
            if not q['detected']:edge.append(False);continue
            xy=np.array(q['landmarks'])[:,:2];edge.append(float(min(xy.min(),1-xy.max()))<.03)
        return dict(detected=sum(detected),frames=len(fs),coverage=sum(detected)/len(fs),
                    missing=base.runs([not x for x in detected],25,duration),edge=base.runs(edge,25,duration))
    old_g,new_g=geometry(old_frames),geometry(frames)
    old_ret=[retained(q['bbox_native'],old) for q in raw if q['detected']]
    new_ret=[retained(q['bbox_native'],new) for q in raw if q['detected']]
    # Only measured face clipping can justify a stronger recommendation. Estimated head aesthetics cannot.
    clear_geometric_gain=min(old_ret)<.98 and min(new_ret)>=.999 and new_g['coverage']>=old_g['coverage'] and p['scale_ratio']>=.9 and len(new_g['edge'])<=len(old_g['edge'])
    result=dict(id=sid,ordinal=i,source=s,crop=p,before=old_g,after=new_g,
                min_face_retained_before=min(old_ret),min_face_retained_after=min(new_ret),
                recommendation='geometry_improved_review_composition' if clear_geometric_gain else 'retain_existing_pending_review',
                focus_samples=samples,old_review_findings=r['findings'],old_focus_s=r['focus_s'],
                media=f'media/{sid}/comparison.mp4',training_admitted=False,production_applied=False,
                policy='Mild lowering/brief hand occlusion is not exclusion evidence. Unmeasured frames are unknown. Crop-only clipping should be repaired before considering exclusion.',
                runner_sha256=base.sha(__file__),input_sha256=base.sha(PRIOR))
    base.save(OUT/'results'/f'{sid}.json',result)
    return result


def classify_focus(results):
    for r in results:
        samples=r['focus_samples'];valid=[q for q in samples if q['measured']]
        lap_ref=float(np.percentile([q['laplacian_variance'] for q in valid],80)) if valid else None
        grad_ref=float(np.percentile([q['gradient_energy'] for q in valid],80)) if valid else None
        pitches=[q['pitch'] for q in valid if q['pitch'] is not None];pitch_ref=float(np.median(pitches)) if pitches else None
        mask=[]
        for q in samples:
            q['confounds']=[]
            if not q['crop_detected']:q['confounds'].append('裁剪检测缺失，可能低头或遮挡')
            if q['yaw'] is not None and abs(q['yaw'])>40:q['confounds'].append('明显侧转')
            if pitch_ref is not None and q['pitch'] is not None and abs(q['pitch']-pitch_ref)>20:q['confounds'].append('俯仰变化较大')
            if q['measured']:
                q['lap_ratio']=q['laplacian_variance']/max(lap_ref,1e-9);q['gradient_ratio']=q['gradient_energy']/max(grad_ref,1e-9)
            else:q['lap_ratio']=q['gradient_ratio']=None
        # Two complementary edge measures; three-sample median. This is a ranking heuristic, not a focus diagnosis.
        for i,q in enumerate(samples):
            local=samples[max(0,i-1):min(len(samples),i+2)]
            vals=[v for v in local if v['measured']]
            low=q['measured'] and len(vals)>=2 and np.median([v['lap_ratio'] for v in vals])<.45 and np.median([v['gradient_ratio'] for v in vals])<.70
            mask.append(bool(low))
        intervals=[x for x in base.runs(mask,5,r['source']['duration_s']) if x['duration']>=.6-1e-8]
        for x in intervals:
            affected=[q for q in samples if x['start']<=q['t']<x['end']]
            x['confounds']=sorted({z for q in affected for z in q['confounds']})
            x['label']='低清晰度区间，失焦／运动／遮挡待区分'
            x['sampling_uncertainty_s']=.2
        source_edge=[q['t'] for q in valid if q.get('original_frame_retained',1)<.95]
        best=max(valid,key=lambda q:q['laplacian_variance']) if valid else None
        worst=min(valid,key=lambda q:q['laplacian_variance']) if valid else None
        r.update(focus_reference=dict(laplacian_p80=lap_ref,gradient_p80=grad_ref,scope='same clip only; no universal sharpness cutoff'),
                 focus_intervals=intervals,source_exit_suspect_samples=source_edge,
                 sharp_frame_t=best['t'] if best else None,soft_frame_t=worst['t'] if worst else None,
                 face_exit_interpretation='No detection is never evidence of leaving the original frame; source-edge landmarks are only review clues.',
                 focus_interpretation='No confirmed defocus labels. 5 fps face-interior patches in native pixels normalized to 128x128; persistent within-clip drop is a review cue. Uniformly blurry clips can be missed.')


def main():
    WEB.mkdir(parents=True,exist_ok=True)
    source=json.loads(PRIOR.read_text());results=[]
    for i,r in enumerate(source['results'],1):
        p=OUT/'results'/f"{r['sample_id']}.json"
        if p.exists():
            res=json.loads(p.read_text());assert res['runner_sha256']==base.sha(__file__)
        else:res=process(r,i)
        results.append(res)
        print(json.dumps(dict(done=i,id=res['id'],recommendation=res['recommendation'],old_coverage=res['before']['coverage'],new_coverage=res['after']['coverage'])),flush=True)
    classify_focus(results)
    summary=dict(clips=len(results),geometric_improvements=[r['ordinal'] for r in results if r['recommendation']=='geometry_improved_review_composition'],
                 focus_review_clips=[r['ordinal'] for r in results if r['focus_intervals']],
                 source_edge_suspect_clips=[r['ordinal'] for r in results if r['source_exit_suspect_samples']],
                 production_applied=0,automatically_rejected=0)
    data=dict(summary=summary,results=results,protocol=dict(runner_sha256=base.sha(__file__),input_sha256=base.sha(PRIOR),
              crop_detection_fps=25,focus_sampling_fps=5,thresholds_calibrated=False,focus_is_not_diagnosis=True))
    base.save(OUT/'data.json',data);base.save(WEB/'data.json',data)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
