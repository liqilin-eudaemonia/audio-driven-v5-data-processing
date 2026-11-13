"""Run added reaction/audio metrics on an enriched bound pair manifest."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from .core import activity_comparison,aggregate_activity,mask
from .media import SpeechDetector,activity,audio_pair,grid25,write_video
from .bridge import Worker,ROOT
from ..video.paired_metrics.run import read_manifest,write_json
from ..video.paired_metrics.media import load_pair,sha256
from ..video.paired_metrics.backends import Networks


def windows_quality(gt,pred,ga,pa,gvideo,pvideo,folder,options,worker):
    import soundfile as sf
    duration=min(len(gt),len(pred))/16000;windows=[]
    for i,(start,end) in enumerate(ga['speech']):
        # One reference-defined continuous window; never join separate responses.
        a=max(0.,math.floor((start-options['quality_context_s'])*25)/25)
        b=min(duration,math.ceil((end+options['quality_context_s'])*25)/25)
        begin,finish=round(a*16000),round(b*16000)
        support=sum(max(0.,min(end,y)-max(start,x)) for x,y in pa['speech'])/(end-start)
        row=dict(reference_speech=[start,end],window=[a,b],generated_speech_fraction=support,reference_eligible={k:end-start>=options[k+'_min_s'] for k in ('utmos','sync')})
        for kind in ('utmos','sync'):
            if kind not in options.get('quality_kinds',['utmos','sync']):
                row[kind]=dict(status='not_applicable',reason='not_requested');continue
            minimum=options[kind+'_min_s']
            if end-start<minimum:
                row[kind]=dict(status='not_applicable',reason='reference_speech_too_short');continue
            if support<options['quality_min_generated_speech_fraction']:
                row[kind]=dict(status='not_applicable',reason='insufficient_generated_speech_not_quality_success');continue
            dest=folder/f'{i}_{kind}';dest.mkdir()
            for side,wave in [('gt',gt),('generated',pred)]:sf.write(dest/(side+'.wav'),wave[begin:finish],16000,subtype='FLOAT')
            request=dict(kind=kind,gt_audio=str(dest/'gt.wav'),generated_audio=str(dest/'generated.wav'),output=str(dest/'sync'))
            if kind=='sync':
                for side,frames in [('gt',gvideo),('generated',pvideo)]:
                    write_video(frames[round(a*25):math.ceil(b*25-1e-7)],dest/(side+'.mkv'))
                    request[side+'_video']=str(dest/(side+'.mkv'))
            try:row[kind]=worker.call(request)
            except Exception as exc:row[kind]=dict(status='error',reason=f'{type(exc).__name__}: {exc}')
        windows.append(row)
    return windows


def quality_summary(records,kinds=("utmos","sync")):
    result={}
    for kind in kinds:
        windows=[(r['id'],w) for r in records for w in r.get('audio',{}).get('quality_windows',[])]
        valid=[(i,w) for i,w in windows if w[kind]['status']=='success']
        block=dict(reference_windows=len(windows),eligible_reference_windows=sum(w['reference_eligible'][kind] for _,w in windows),
            scored_windows=len(valid),scored_pairs=len({i for i,_ in valid}),scored_pair_ids=sorted({i for i,_ in valid}),
            scored_window_seconds=sum(w['window'][1]-w['window'][0] for _,w in valid),
            unavailable_reasons=dict(Counter(w[kind].get('reason','unknown') for _,w in windows if w[kind]['status']!='success')),
            scope='conditional_common_windows; missed responses scored separately; compare common IDs/windows across systems')
        for side in ('gt','generated'):
            if kind=='utmos':block[side]=dict(mean=float(np.mean([w[kind]['scores'][side]['value'] for _,w in valid])) if valid else None)
            else:
                scores=[w[kind]['scores'][side] for _,w in valid]
                offsets=[t['offset_frames']/25 for s in scores for t in s.get('tracks',[]) if t['status']=='success']
                block[side]=dict(lse_d=float(np.mean([s['lse_d'] for s in scores])) if scores else None,
                    lse_c=float(np.mean([s['lse_c'] for s in scores])) if scores else None,
                    mean_absolute_offset_s=float(np.mean(np.abs(offsets))) if offsets else None,
                    offset_tracks=len(offsets))
        result[kind]=block
    return result


class Reactions:
    def __init__(self,assets,device,batch):
        sys.path.insert(0,str(ROOT/'src'))
        from p2_glare_eval.upstream import metrics,REACTIONS
        self.m=metrics();self.names=REACTIONS;self.annotations=[];self.ids=[]
        self.pool=[self.m['r_fid'].FeatureMoments(2048) for _ in range(2)]
        self.net=Networks(assets,device,batch);self.batch=batch
    def add(self,identifier,frames,detected):
        common=self.m['common'];annotations=[]
        # Stage features first so failed pair cannot partly pollute a distribution.
        features=[]
        for video,result in zip(frames,detected):
            ann=common.Annotation(len(video),tuple(common.Event(**e) for e in result['events']));annotations.append(ann)
            selected=np.zeros(len(video),bool)
            for e in ann.events:selected[e.start:e.end]=True
            blocks=[]
            indexes=np.flatnonzero(selected)
            for start in range(0,len(indexes),self.batch):blocks.append(self.net.fid(video[indexes[start:start+self.batch]]))
            features.append(np.concatenate(blocks) if blocks else np.empty((0,2048)))
        for pool,values in zip(self.pool,features):pool.update(values)
        self.annotations.append((identifier,*annotations));self.ids.append(identifier)
    def summary(self,total):
        pairs=self.annotations;m=self.m
        values={'R-F1':m['r_f1'].compute_r_f1(pairs) if pairs else None,
            'R-tIoU':m['r_tiou'].compute_r_tiou(pairs) if pairs else None,
            'R-ATD':m['r_atd'].compute_r_atd(pairs) if pairs else None,'R-FID':None}
        errors={}
        if min(p.count for p in self.pool)>=2:
            try:values['R-FID']=m['r_fid'].frechet_distance(*self.pool)
            except Exception as e:errors['R-FID']=str(e)
        else:errors['R-FID']='fewer_than_two_reaction_frames_in_one_pool'
        counts={n:dict(tp=0,fp=0,fn=0) for n in self.names}
        for _,g,p in pairs:
            matches=m['common'].match_events(g.events,p.events)
            for n,c in counts.items():
                tp=sum(x.ground_truth.reaction==n for x in matches);c['tp']+=tp
                c['fp']+=sum(x.reaction==n for x in p.events)-tp;c['fn']+=sum(x.reaction==n for x in g.events)-tp
        for c in counts.values():
            t,p,n=c['tp'],c['fp'],c['fn'];c['f1']=2*t/(2*t+p+n) if 2*t+p+n else 0.
        return dict(full_dataset_metrics=values if len(pairs)==total else {k:None for k in values},
            common_valid_subset_metrics=values,common_valid_pair_ids=self.ids,valid_pairs=len(pairs),requested_pairs=total,
            per_class=counts,reaction_frame_counts=[p.count for p in self.pool],errors=errors,
            missing_detection_policy='exclude_from_visual_metrics_only_and_report_not_negative',r_atd_scale=100)


def quality_video(row):
    g,p,meta=load_pair(row)
    duration=min(row.get('pair_duration_s',row.get('duration_s',meta['aligned_frames']/meta['fps'])),meta['aligned_frames']/meta['fps'])
    if abs(meta['gt_frames']-meta['generated_frames'])>1:raise ValueError('More than one frame missing')
    g25,grid=grid25(g,meta['fps'],duration);p25,_=grid25(p,meta['fps'],duration)
    return g25,p25,meta,grid,duration


def evaluate(args):
    import cv2,torch
    cv2.setNumThreads(2);torch.set_num_threads(2)
    selected=set(args.stages.split(','))
    if not selected or not selected<={'reaction','audio'}:raise ValueError('Expected reaction,audio stages')
    options=json.loads(args.config.read_text())
    options['quality_kinds']=getattr(args,'quality_kinds','utmos,sync').split(',')
    if not set(options['quality_kinds'])<={'utmos','sync'}:raise ValueError('Unknown quality metric')
    options['audio_models_device']=getattr(args,'audio_device','cpu')
    assets=json.loads(args.assets.read_text())
    if options['schema']!='listener-response-metrics-v1':raise ValueError('Unknown policy')
    for key in ('energy_min_s','energy_gap_s','event_collar_s','utmos_min_s','sync_min_s','quality_context_s'):
        if not math.isfinite(options[key]) or options[key]<0:raise ValueError('Invalid '+key)
    if options['utmos_min_s']<=3 or options['sync_min_s']<1.44:raise ValueError('Do not bypass model minimum support')
    if not -100<=options['energy_dbfs']<=0 or not 0<options['quality_min_generated_speech_fraction']<=1:raise ValueError('Invalid audio support thresholds')
    rows=read_manifest(args.manifest);out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for r in rows:
        for k in ('gt_audio','gt_asr','generated_audio'):
            if r.get(k):r[k]=str((args.manifest.resolve().parent/Path(r[k])).resolve())
    workers={};records=[];react=None;vad=None
    contract=dict(schema='listener-metrics-v1',manifest_sha256=sha256(args.manifest),options=options,
        self_check_only=all(r.get('self_check_only',False) for r in rows),stages=sorted(selected),datasets=sorted({r.get('dataset','unknown') for r in rows}),
        code={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},assets=assets,
        reaction_grid='25fps nearest source frame, ties earlier; same duration, no time stretch; original 8 video metrics unchanged',
        temporary_media='lossless per-pair only; deleted unless --keep-media',audio_references='GT-defined; never threshold normalized generated audio',
        thresholds_status='versioned engineering defaults, not human calibrated',models={})
    write_json(out/'contract.json',contract)
    try:
        if 'audio' in selected:
            vad=SpeechDetector();contract['models']['vad']=vad.identity;workers['audio']=Worker('audio',out,options['audio_models_device'])
        if 'reaction' in selected:
            react=Reactions(assets,args.device,args.batch_size);workers['glare']=Worker('glare',out)
        runtime=json.loads((ROOT/'configs/p2_omni_runtime.lock.json').read_text())
        contract['models']['legacy_audio_runtime']=runtime
        contract['models']['fid_inception_sha256']=sha256(assets['fid_inception']) if react else None
        write_json(out/'contract.json',contract)
        for i,row in enumerate(rows):
            started=time.time();dest=out/f'work_{i:06d}';dest.mkdir();result=dict(id=row['id'],dataset=row.get('dataset'),errors={})
            try:
                if selected=={'audio'} and options['quality_kinds']==['utmos']:
                    duration=float(row['pair_duration_s']);g25=p25=[]
                    result['media']=dict(audio_only=True,duration_s=duration,video_decoding_skipped=True)
                else:
                    g25,p25,meta,grid,duration=quality_video(row)
                    result.update(media=meta,evaluation_grid=grid)
            except Exception as exc:
                result['errors']['media']=f'{type(exc).__name__}: {exc}';g25=p25=None
            if g25 is not None and react:
                try:
                    detections=[]
                    for side,frames in [('gt',g25),('generated',p25)]:
                        video=dest/(side+'.mkv');write_video(frames,video)
                        detections.append(workers['glare'].call(dict(video=str(video),output=str(dest/side))))
                    result['reaction']=dict(gt=detections[0],generated=detections[1],valid=all(d['valid'] for d in detections))
                    if result['reaction']['valid']:react.add(row['id'],[g25,p25],detections)
                    else:result['errors']['reaction']='incomplete_independent_detection'
                except Exception as exc:result['errors']['reaction']=f'{type(exc).__name__}: {exc}'
            if g25 is not None and vad:
                try:
                    gaudio,paudio,audit=audio_pair(row,duration);duration_a=audit['evaluated_duration_s']
                    ga,pa=[activity(x,options,vad) for x in [gaudio,paudio]]
                    comparison={kind:activity_comparison(ga[kind],pa[kind],duration_a,options['event_collar_s']) for kind in ('acoustic','speech')}
                    silent=~mask(ga['acoustic'],duration_a);samplemask=np.repeat(silent,160)[:len(paudio)]
                    noise=paudio[:len(samplemask)][samplemask]
                    qfolder=dest/'quality';qfolder.mkdir()
                    quality=windows_quality(gaudio,paudio,ga,pa,g25,p25,qfolder,options,workers['audio'])
                    group='silent' if not ga['acoustic'] else 'long_speech' if any(b-a>=options['utmos_min_s'] for a,b in ga['speech']) else 'short_or_non_speech_sound'
                    result['audio']=dict(audit=audit,gt=ga,generated=pa,group=group,comparison=comparison,
                        generated_rms_on_gt_silent=float(np.sqrt(np.mean(noise.astype(float)**2))) if len(noise) else None,quality_windows=quality)
                    if any(w[k]['status']=='error' for w in quality for k in ('utmos','sync')):result['errors']['audio_quality']='window_worker_error'
                except Exception as exc:result['errors']['audio']=f'{type(exc).__name__}: {exc}'
            records.append(result)
            with (out/'pairs.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
            if not args.keep_media:shutil.rmtree(dest)
            print(f'{i+1}/{len(rows)} {row["id"]} {time.time()-started:.1f}s errors={list(result["errors"])}',flush=True)
        summary=dict(schema='listener-metrics-v1',self_check_only=contract['self_check_only'],requested_pairs=len(rows),datasets=contract['datasets'],
            pairs_with_errors=sum(bool(r['errors']) for r in records),errors=dict(Counter(k for r in records for k in r['errors'])))
        if react:summary['reaction']=react.summary(len(rows))
        if vad:
            valid=[r for r in records if 'audio' in r];summary['audio']=dict(valid_pairs=len(valid),requested_pairs=len(rows),
                groups=dict(Counter(r['audio']['group'] for r in valid)),
                activity={k:aggregate_activity([r['audio']['comparison'][k] for r in valid]) for k in ('acoustic','speech')},
                quality=quality_summary(records,options['quality_kinds']),failed_pair_ids=[r['id'] for r in records if 'audio' not in r])
        summary['status']='partial' if summary['pairs_with_errors'] else 'complete'
        write_json(out/'summary.json',summary)
        return 2 if summary['status']=='partial' else 0
    finally:
        for w in workers.values():w.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path(__file__).parents[1]/'video/paired_metrics/assets.local.json')
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));p.add_argument('--stages',default='reaction,audio')
    p.add_argument('--quality-kinds',default='utmos,sync');p.add_argument('--audio-device',default='cpu')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--keep-media',action='store_true')
    args=p.parse_args();raise SystemExit(evaluate(args))


if __name__=='__main__':main()
