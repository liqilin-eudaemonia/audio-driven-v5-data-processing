#!/usr/bin/env python3
"""Conservative candidate gate using physical-angle dominance and real reversals.

This is a pilot gate, not a validated head-shake classifier. Original outputs remain
immutable. Nod overlap is diagnostic only and never a rejection criterion.
"""
import argparse
import copy
import json
import hashlib
from pathlib import Path
import numpy as np
from run_glare_nodding_pilot import OUT as NOD, VENDOR, module, sha, save
from run_glare_reaction_pilot import OUT as BASE, TYPES

OUT=BASE/'headshake_direction_v1'
PARAMS={'smooth_frames':5,'min_yaw_range_deg':3.0,'min_reversal_leg_deg':2.0,
        'min_yaw_pitch_range_ratio':1.2,'min_yaw_pitch_rms_speed_ratio':1.5,'min_face_coverage':.95}

def legs(values, threshold):
    """Hysteresis extrema on smoothed raw yaw, without detrending or normalization."""
    if not len(values):return []
    anchor=extreme=float(values[0]);direction=0;result=[]
    for value in values[1:]:
        value=float(value)
        if direction==0:
            if abs(value-anchor)>=threshold:
                direction=1 if value>anchor else -1;extreme=value
        elif direction*(value-extreme)>=0:
            extreme=value
        elif direction*(extreme-value)>=threshold:
            result.append(extreme-anchor);anchor=extreme;extreme=value;direction=-direction
    if direction and abs(extreme-anchor)>=threshold:result.append(extreme-anchor)
    return result

def assess(yaw,pitch,detected,params=PARAMS):
    yr=float(np.ptp(yaw));pr=float(np.ptp(pitch))
    yv=float(np.sqrt(np.mean(np.diff(yaw)**2))) if len(yaw)>1 else 0.
    pv=float(np.sqrt(np.mean(np.diff(pitch)**2))) if len(pitch)>1 else 0.
    excursions=legs(yaw,params['min_reversal_leg_deg'])
    checks={'face_coverage':float(np.mean(detected))>=params['min_face_coverage'],
            'yaw_amplitude':yr>=params['min_yaw_range_deg'],
            'yaw_range_dominant':yr>=params['min_yaw_pitch_range_ratio']*pr,
            'yaw_speed_dominant':yv>=params['min_yaw_pitch_rms_speed_ratio']*pv,
            'opposite_yaw_excursions':any(x>=params['min_reversal_leg_deg'] for x in excursions) and any(x<=-params['min_reversal_leg_deg'] for x in excursions)}
    return {'retain':all(checks.values()),'failed_checks':[k for k,v in checks.items() if not v],
            'metrics':{'yaw_range_deg':yr,'pitch_range_deg':pr,'yaw_rms_speed_deg_per_frame':yv,
                       'pitch_rms_speed_deg_per_frame':pv,'yaw_excursions_deg':excursions,'face_coverage':float(np.mean(detected))}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=BASE,help='Reaction run containing original head-shake candidates')
    parser.add_argument('--nod-run',type=Path,default=NOD,help='Nod run containing per-frame pitch, yaw and face detection')
    parser.add_argument('--output',type=Path,help='Defaults to <base>/headshake_direction_v1')
    args=parser.parse_args()
    base,nod_run=args.base.resolve(),args.nod_run.resolve()
    out=(args.output or base/'headshake_direction_v1').resolve()
    mod=module('shake_direction_source',VENDOR/'reaction_detector/head_shaking/run_head_shaking_intensity_mt.py')
    ids=json.loads((base/'samples.json').read_text())['ids']
    protocol={'schema':'headshake-direction-gate-v1','parameters':PARAMS,'code_sha256':sha(Path(__file__)),
              'status':'selected_headshake_gate','training_admitted':False,
              'rule':'gate original candidates only; no new events; unchanged original boundaries; no nod-overlap veto',
              'base_protocol_sha256':sha(base/'protocol.json'),
              'base':str(base),'nod_run':str(nod_run)}
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    save(out/'protocol.json',protocol);results=[];diagnostics=[];overlap_count=0
    for sid in ids:
        path=base/'results'/f'{sid}.json';original=json.loads(path.read_text());r=copy.deepcopy(original)
        n=json.loads((nod_run/'results'/f'{sid}.json').read_text());mask=np.array(n['pose']['detected'])
        pose={k:mod.moving_average(mod.fill_pose_gaps(np.asarray(n['pose'][k]),mask),PARAMS['smooth_frames']) for k in ('pitch','yaw')}
        channel=r['channels']['head_shaking'];old=copy.deepcopy(channel)
        kept=[];scores=np.zeros(len(channel['scores']));review=[]
        for j,e in enumerate(channel['events']):
            a,b=e['start_frame'],e['end_frame_exclusive'];decision=assess(pose['yaw'][a:b],pose['pitch'][a:b],mask[a:b])
            overlap=sum(max(0,min(e['end'],v['end'])-max(e['start'],v['start'])) for v in r['channels']['nodding']['events'])
            overlap_count+=overlap>0
            decision.update(event=e,original_event_id=f'glare-head_shaking-{j+1:03d}',nod_overlap_seconds=overlap)
            review.append(decision)
            if decision['retain']:
                kept.append(e);scores[a:b]=np.asarray(channel['scores'][a:b])
        channel.update(events=kept,scores=scores.tolist())
        keep_ids={d['original_event_id'] for d in review if d['retain']}
        r['supplementary_events']=[e for e in r.get('supplementary_events',[]) if e['type']!='head_shaking' or e['event_id'] in keep_ids]
        for e in r['supplementary_events']:
            if e['type']=='head_shaking':e['configuration']='direction_gate_v1'
        r['headshake_direction_review']={'baseline_sha256':sha(path),'nod_sha256':sha(nod_run/'results'/f'{sid}.json'),
                                        'baseline_channel':old,'decisions':review,'protocol_fingerprint':fingerprint}
        r['fingerprint']=fingerprint
        # Strong invariants: neither nodding nor any expression is modified.
        for k in TYPES:
            if k!='head_shaking' and k in original.get('channels',{}):
                assert r['channels'][k]==original['channels'][k]
        assert all(e in old['events'] for e in channel['events'])
        save(out/'results'/f'{sid}.json',r);results.append(r)
        diagnostics.extend({'sample_id':sid,**d} for d in review)
    summary={'samples':len(ids),'ok':len(results),'errors':[],
             'original_headshake_intervals':len(diagnostics),'original_overlapping_nod_intervals':overlap_count,
             'retained_headshake_intervals':sum(d['retain'] for d in diagnostics),
             'channels':{k:{'clips':sum(bool(r.get('channels',{}).get(k,{}).get('events')) for r in results),
                            'intervals':sum(len(r.get('channels',{}).get(k,{}).get('events',[])) for r in results),
                            'seconds':sum(e['end']-e['start'] for r in results for e in r.get('channels',{}).get(k,{}).get('events',[]))} for k in TYPES}}
    save(out/'summary.json',summary);save(out/'diagnostics.json',diagnostics)
    with (out/'supplementary_events.jsonl').open('w') as f:
        for r in results:f.write(json.dumps({'sample_id':r['id'],'source_fingerprint':fingerprint,'training_admitted':False,'events':r['supplementary_events']})+'\n')
    print(json.dumps(summary))

if __name__=='__main__':main()
