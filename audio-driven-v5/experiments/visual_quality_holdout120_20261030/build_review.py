#!/usr/bin/env python3
"""Render all candidates plus predeclared borderline/control review selections."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('holdout_run',Path(__file__).with_name('run.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
import json,math,random,subprocess,tempfile,concurrent.futures,shutil
from fractions import Fraction
np,cv2,av=h.np,h.cv2,h.av


def select(data):
    good=[r for r in data['results'] if r['status']=='ok'];chosen={};why={}
    def add(r,reason):chosen[r['id']]=r;why.setdefault(r['id'],[]).append(reason)
    for r in good:
        if r['focus']['candidate_exclusion']:add(r,'排除候选：全部展示')
    remaining=[r for r in good if not r['focus']['candidate_exclusion'] and r['focus']['low_s']>0]
    remaining.sort(key=lambda r:min(r['focus']['longest_low_s']/2,r['focus']['low_fraction']/.25),reverse=True)
    for r in remaining[:6]:add(r,'接近阈值，规则保留')
    incomplete=sorted([r for r in good if r['focus']['status']=='measurement_incomplete'],key=lambda r:r['focus']['measured_fraction'])
    for r in incomplete[:4]:add(r,'测量覆盖不足，不能当成质量通过')
    pool=[r for r in good if r['id'] not in chosen and r['focus']['status']=='not_triggered']
    random.Random(2026103004).shuffle(pool)
    for r in pool[:6]:add(r,'随机保留对照')
    shifted=sorted([r for r in good if r['crop']['changed']],key=lambda r:abs(r['crop']['new']['x']-r['crop']['old']['x'])+abs(r['crop']['new']['y']-r['crop']['old']['y']),reverse=True)
    for r in shifted[:2]:add(r,'裁框平移对照')
    expanded=sorted([r for r in good if r['crop']['status']=='small_expansion_proposal'],key=lambda r:r['crop']['scale_ratio'])
    for r in expanded[:2]:add(r,'受限扩大对照：边长最多 +5%')
    selected=sorted(chosen.values(),key=lambda r:(not r['focus']['candidate_exclusion'],r['ordinal']))
    h.base.save(h.OUT/'review_selection.json',{'sample_ids':[r['id'] for r in selected],'reasons':why,
                 'policy':'all exclusion candidates, up to 6 nearest retained cases, 4 lowest-support cases, 6 random retained controls, 2 largest translations and 2 largest bounded expansions; overlaps deduplicated'})
    return selected,why


def render(r):
    sid=r['id'];s=r['source'];old,new=r['crop']['old'],r['crop']['new'];folder=h.WEB/'media'/sid
    folder.mkdir(parents=True,exist_ok=True);receipt=folder/'receipt.json'
    if receipt.exists():return json.loads(receipt.read_text())
    duration=s['duration_s'];origin=round(s['source_start_s']*s['normalized_audio']['sample_rate'])/s['normalized_audio']['sample_rate']
    w,he=s['video_transform']['source_size'];raw_width=round(w*512/he/2)*2;n=math.ceil(duration*25-1e-9)
    valid=[q for q in r['focus_samples'] if q['measured']]
    sharp=max(valid,key=lambda q:q['laplacian_variance']) if valid else None
    soft=min(valid,key=lambda q:q['laplacian_variance']) if valid else None
    longest=max(r['focus']['intervals'],key=lambda q:q['duration'],default=None)
    at=(longest['start']+longest['end'])/2 if longest else (soft['t'] if soft else duration/2)
    points={'focus':min(n-1,round(at*25)), 'sharp':round(sharp['t']*25) if sharp else None,'soft':round(soft['t']*25) if soft else None}
    with tempfile.TemporaryDirectory(prefix='holdout-review-') as name:
        temp=Path(name);silent=temp/'silent.mp4';sample_times=[]
        with h.base.SourceBytes(s['source']) as source,av.open(source) as inp,av.open(str(silent),'w') as out:
            stream=inp.streams.video[0];stream.codec_context.thread_count=1;inp.seek(int(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
            e=out.add_stream('libx264',rate=25);e.width=raw_width+1024;e.height=512;e.pix_fmt='yuv420p';e.options={'crf':'18','preset':'fast','threads':'1'}
            k=0;last=None;last_t=None
            def write(rgb,t,k):
                before,after=h.focus.crop_rgb(rgb,old),h.focus.crop_rgb(rgb,new)
                raw=cv2.resize(rgb,(raw_width,512),interpolation=cv2.INTER_AREA)
                for c,color in [(old,(70,220,110)),(new,(60,185,255))]:
                    cv2.rectangle(raw,(round(c['x']*raw_width/w),round(c['y']*512/he)),(round((c['x']+c['width'])*raw_width/w),round((c['y']+c['height'])*512/he)),color,2)
                combined=np.concatenate([raw,before,after],axis=1);h.base.encode(e,out,combined,k);sample_times.append(origin+t)
                for label,index in points.items():
                    if index==k:
                        cv2.imwrite(str(folder/f'{label}.jpg'),cv2.cvtColor(combined,cv2.COLOR_RGB2BGR))
                        q=sharp if label=='sharp' else soft if label=='soft' else None
                        if q:
                            a,b,c,d=q['native_patch'];patch=rgb[b:d,a:c]
                            cv2.imwrite(str(folder/f'{label}_face.jpg'),cv2.cvtColor(cv2.resize(patch,(256,256),interpolation=cv2.INTER_CUBIC),cv2.COLOR_RGB2BGR))
            for f in inp.decode(stream):
                t=float(f.pts*f.time_base)-origin
                if t<0:continue
                if t>=duration:break
                if k>=n:break
                last=f;last_t=t
                if k/25>t+1e-10:continue
                rgb=f.to_ndarray(format='rgb24')
                while k<n and k/25<=t+1e-10:write(rgb,t,k);k+=1
            if last is None or (k<n and duration-last_t>=.1):raise ValueError('Incomplete preview support')
            while k<n:write(last.to_ndarray(format='rgb24'),last_t,k);k+=1
            for packet in e.encode():out.mux(packet)
        assert h.base.sha(s['source_result'])==s['source_result_sha256']
        full=json.loads(Path(s['source_result']).read_text())['clips'][s['source_clip_index']];assert full['sample_id']==sid
        speaker=full['normalized_audio']['speaker']['path']
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','1','-i',str(silent),'-ss',str(origin),'-i',s['normalized_audio']['path'],'-ss',str(origin),'-i',speaker,'-filter_complex_threads','1','-filter_complex','[1:a][2:a]amix=inputs=2:duration=shortest:normalize=1[a]','-map','0:v','-map','[a]','-t',str(duration),'-c:v','copy','-c:a','aac','-b:a','96k','-movflags','+faststart',str(folder/'comparison.mp4')],check=True,timeout=90)
    info={'id':sid,'media':f'media/{sid}/comparison.mp4','poster':f'media/{sid}/focus.jpg','focus_t':points['focus']/25,
          'sharp_t':sharp['t'] if sharp else None,'soft_t':soft['t'] if soft else None,
          'sharp_face':f'media/{sid}/sharp_face.jpg' if sharp else None,'soft_face':f'media/{sid}/soft_face.jpg' if soft else None,
          'frames':n,'source_frame_times_s':sample_times,'sha256':h.base.sha(folder/'comparison.mp4')}
    h.base.save(receipt,info);return info


def main():
    data=json.loads((h.OUT/'data.json').read_text());selected,reasons=select(data);media={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(render,r) for r in selected]
        for task in concurrent.futures.as_completed(tasks):
            r=task.result();media[r['id']]=r;print(json.dumps({'rendered':len(media),'total':len(selected),'id':r['id']}),flush=True)
    for r in data['results']:
        r['review_reasons']=reasons.get(r['id'],[])
        r['review_media']=media.get(r['id'])
    data['review_order']=[r['id'] for r in selected];data['summary']['rendered_reviews']=len(selected)
    h.base.save(h.WEB/'data.json',data);shutil.copyfile(h.OUT/'review.html',h.WEB/'index.html')
    print(json.dumps({'published':str(h.WEB),'rendered':len(selected)}),flush=True)


if __name__=='__main__':main()
