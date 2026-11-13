#!/usr/bin/env python3
"""Add independent native-pixel face checks in an expanded, unclipped context ROI."""
import run as pilot
import json
import math
import pickle
import shutil
import tempfile
from pathlib import Path

np,cv2,av,mp=pilot.np,pilot.cv2,pilot.av,pilot.mp


def supplement(row):
    sid=row['sample_id'];s=row['source'];c=s['video_transform']['crop']
    width,height=s['video_transform']['source_size']
    # Expand each side by 25% of the training-crop side, clipped only at source edges.
    x0=max(0,math.floor(c['x']-.25*c['width']));y0=max(0,math.floor(c['y']-.25*c['height']))
    x1=min(width,math.ceil(c['x']+1.25*c['width']));y1=min(height,math.ceil(c['y']+1.25*c['height']))
    context_rect=dict(x=x0,y=y0,width=x1-x0,height=y1-y0)
    local_crop=dict(x=c['x']-x0,y=c['y']-y0,width=c['width'],height=c['height'])
    media= json.loads((pilot.OUT/'media_metadata'/f'{sid}.json').read_text())
    old=json.loads((pilot.OUT/'results'/f'{sid}.json').read_text())
    targets=media['raw_rows'];origin=media['source_start_s'];updated=[]
    with tempfile.TemporaryDirectory(prefix='visual-context-') as temp_name, \
         pilot.face.FaceLandmarker.create_from_options(pilot.face.build_options(str(pilot.MODEL))) as detector, \
         pilot.SourceBytes(s['source']) as source,av.open(source) as inp:
        temp=Path(temp_name);stream=inp.streams.video[0];stream.codec_context.thread_count=1
        inp.seek(int(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
        k=0
        for frame in inp.decode(stream):
            t=float(frame.pts*frame.time_base)
            if k>=len(targets):break
            if t+1e-7<targets[k]['source_time']:continue
            rgb=frame.to_ndarray(format='rgb24')
            while k<len(targets) and targets[k]['source_time']<=t+1e-7:
                original=targets[k]
                if abs(t-original['source_time'])>.05:raise ValueError('Context/source sampling mismatch')
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb[y0:y1,x0:x1])),round(original['t']*1000))
                measured=pilot.raw_measure(result,x1-x0,y1-y0,local_crop,original['t'],t)
                if measured['detected']:
                    b=measured['bbox_native'];measured['bbox_native']=[b[0]+x0,b[1]+y0,b[2]+x0,b[3]+y0]
                use_context=not original['detected'] or not original.get('center_inside_crop',True)
                r=dict(measured if use_context else original)
                r.update(detector_input='expanded_context' if use_context else 'full_frame',
                         full_frame_result=original,context_result=measured,context_rect=context_rect)
                updated.append(r)
                small_width=int(round(width*512/height/2))*2
                small=cv2.resize(rgb,(small_width,512),interpolation=cv2.INTER_AREA)
                sx=small_width/width;sy=512/height
                cv2.rectangle(small,(round(c['x']*sx),round(c['y']*sy)),(round((c['x']+c['width'])*sx),round((c['y']+c['height'])*sy)),(40,230,100),2)
                if r['detected']:
                    b=r['bbox_native'];cv2.rectangle(small,(round(b[0]*sx),round(b[1]*sy)),(round(b[2]*sx),round(b[3]*sy)),(255,190,30),2)
                crop=np.asarray(pilot.Image.fromarray(rgb[c['y']:c['y']+c['height'],c['x']:c['x']+c['width']]).resize((512,512),pilot.Image.Resampling.BICUBIC))
                cv2.imwrite(str(temp/f'{k}.jpg'),cv2.cvtColor(np.concatenate([small,crop],axis=1),cv2.COLOR_RGB2BGR));k+=1
        if len(updated)!=len(targets):raise ValueError('Missing context samples')
        media['raw_rows']=updated
        with (pilot.OUT/'landmarks'/f'{sid}.pkl').open('rb') as f:frames=pickle.load(f)
        result=pilot.summarize(row,media,frames)
        result['metrics'].update(full_frame_detected_samples=sum(r['full_frame_result']['detected'] for r in updated),
                                 expanded_context_detected_samples=sum(r['context_result']['detected'] for r in updated),
                                 context_fallback_detected_samples=sum(r['detector_input']=='expanded_context' and r['detected'] for r in updated))
        if not result['metrics']['raw_detected_samples']:
            result['findings'].append(dict(kind='unmeasured_geometry',label='原图及扩展区域均未测得面部几何，需人工检查',start=0,end=s['duration_s'],duration=s['duration_s']))
            result['status']='review'
        priority={'raw_outside_crop':5,'missing_face':4,'near_edge':3,'unmeasured_geometry':2,'small_face':1}
        candidates=sorted(result['findings'],key=lambda e:(priority[e['kind']],e['duration']),reverse=True)
        if candidates and candidates[0]['kind']=='raw_outside_crop':
            # Choose the sampled point with the largest measured crop loss.
            focused=min((f for f in candidates if f['kind']=='raw_outside_crop'),key=lambda f:f['retained_fraction'])
        else:focused=candidates[0] if candidates else None
        focus=(focused['start']+focused['end'])/2 if focused else min([r for r in updated if r['detected']],key=lambda r:r['face_height_native'],default={'t':s['duration_s']/2})['t']
        nearest=min(range(len(updated)),key=lambda k:abs(updated[k]['t']-focus))
        shutil.copyfile(temp/f'{nearest}.jpg',pilot.WEB/'media'/sid/'focus.jpg')
        result.update(focus_s=updated[nearest]['t'],focus_reason=focused['label'] if focused else '对照：面部高度较小的抽样帧',
                      media=old['media'],poster=old['poster'],runner_sha256=old['runner_sha256'],
                      source_result_sha256=old['source_result_sha256'],context_supplement_sha256=pilot.sha(__file__),
                      context_rect=context_rect)
        pilot.save(pilot.OUT/'context_media_metadata'/f'{sid}.json',media)
        pilot.save(pilot.OUT/'context_results'/f'{sid}.json',result)
    return result


def main():
    rows=pilot.select()['rows'];results=[]
    for i,row in enumerate(rows,1):
        p=pilot.OUT/'context_results'/f"{row['sample_id']}.json"
        if p.exists():
            r=json.loads(p.read_text())
            assert r['context_supplement_sha256']==pilot.sha(__file__)
        else:r=supplement(row)
        results.append(r)
        print(json.dumps(dict(done=i,id=r['sample_id'],raw_detected=r['metrics']['raw_detected_samples'],raw_total=r['metrics']['raw_samples'],outside=r['metrics']['raw_outside_crop_samples'],status=r['status'])),flush=True)
    summary=dict(clips=len(results),sources=len(pilot.PAIRS),review=sum(r['status']=='review' for r in results),
                 no_rule_alert=sum(r['status']=='no_rule_alert' for r in results),
                 any_missing=sum(r['metrics']['face_detection_fraction']<1 for r in results),
                 near_edge=sum(bool(r['metrics']['near_edge_intervals']) for r in results),
                 raw_outside_crop=sum(r['metrics']['raw_outside_crop_samples']>0 for r in results),
                 small_face=sum(any(f['kind']=='small_face' for f in r['findings']) for r in results),
                 unmeasured_geometry=sum(r['metrics']['raw_detected_samples']==0 for r in results),
                 cached_nod=sum(r['existing_nod_cache']['available'] for r in results),automatic_rejections=0,training_admitted=False)
    protocol=json.loads((pilot.OUT/'protocol.json').read_text())
    protocol.update(context_supplement_sha256=pilot.sha(__file__),context_method='Full original frame at 5 fps; where unavailable or face center outside training crop, use a 1.5x expanded original-pixel ROI, bounded only by source edges. Preserve both detector results and map all landmarks back to original pixels.',
                    review_only=True)
    pilot.save(pilot.OUT/'context_summary.json',summary)
    pilot.save(pilot.WEB/'data.json',dict(summary=summary,protocol=protocol,results=results))
    shutil.copyfile(pilot.OUT/'review.html',pilot.WEB/'index.html')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
