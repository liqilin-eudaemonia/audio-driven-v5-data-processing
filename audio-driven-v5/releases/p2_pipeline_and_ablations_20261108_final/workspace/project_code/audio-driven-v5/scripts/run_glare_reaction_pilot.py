#!/usr/bin/env python3
"""Add all five other GLARE visual detectors to the reviewed nodding pilot.

Serial CPU FER inference; reuse immutable MediaPipe features and accepted nod scores.
No original VLM, audio, training cache or production annotation is changed.
"""
import os
import sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[k]='2'
_GPU='--gpu' in sys.argv
_VISIBLE=os.environ.get('CUDA_VISIBLE_DEVICES','')
if not _GPU:
    os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
import csv
import hashlib
import inspect
import json
import pickle
import time
from pathlib import Path
import importlib.metadata
from run_glare_nodding_pilot import ROOT, SOURCE, VENDOR, OUT as NOD_OUT, save, sha, module, intervals
if _GPU:
    os.environ['CUDA_VISIBLE_DEVICES']=_VISIBLE

OUT=ROOT/'experiments/listen_glare_reactions15_20261029'
TYPES=('nodding','head_shaking','smiling','laughing','frowning','surprised')
DESCRIPTIONS={'nodding':'Head nodding', 'head_shaking':'Head shaking', 'smiling':'Smiling',
              'laughing':'Visible laughter-like facial movement', 'frowning':'Frowning',
              'surprised':'Surprise-like facial expression'}

def defaults(mod):
    old=sys.argv
    try:
        sys.argv=[mod.__file__]
        args=mod.parse_args()
    finally:
        sys.argv=old
    return args

def overlap(a,b):
    x,y,w,h=a; x0,y0,x1,y1=b
    inter=max(0,min(x+w,x1)-max(x,x0))*max(0,min(y+h,y1)-max(y,y0))
    return inter/max(1,w*h+(x1-x0)*(y1-y0)-inter)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=15)
    parser.add_argument('--nod-run',type=Path,default=NOD_OUT)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--gpu',action='store_true',help='Put MTCNN on the visible CUDA device. Emotion scores stay on TFLite.')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    if not 0<=args.shard<args.shards:raise ValueError('shard must be inside [0, shards)')
    nod_run,out=args.nod_run.resolve(),args.output.resolve()
    import numpy as np
    import cv2
    import torch
    import tensorflow as tf
    from fer.fer import FER
    import fer
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if args.gpu:
        assert torch.cuda.is_available(), 'GPU requested but this process has no CUDA device'
        torch.cuda.set_per_process_memory_fraction(0.04)
    else:
        assert not tf.config.list_physical_devices('GPU') and not torch.cuda.is_available()
    detector=FER(mtcnn=True, use_tflite=True)
    mods={name:module('glare_'+name,VENDOR/f'reaction_detector/{name}/run_{name}_intensity_mt.py') for name in TYPES if name!='nodding'}
    parameters={name:vars(defaults(mod)) for name,mod in mods.items()}
    # Only mathematical parameters belong in the protocol; remove vendor host paths/runtime defaults.
    parameters={name:{k:v for k,v in p.items() if k.startswith(('min_', 'th_', 'smooth_', 'baseline_', 'pitch_penalty'))} for name,p in parameters.items()}
    fer_root=Path(fer.__file__).parent
    import facenet_pytorch
    mtcnn_root=Path(facenet_pytorch.__file__).parent/'data'
    protocol={'schema':'glare-reaction-supplement-v1', 'source_sha256':sha(SOURCE),
              'nod_run':str(nod_run),
              'nod_protocol_sha256':sha(nod_run/'protocol.json'),
              'runner_sha256':sha(Path(__file__)), 'parameters':parameters,
              'nod_profile':'recall', 'nod_user_feedback':'2026-10-29 user: 高召回候选基本正确，可以作为VLM信息补充',
              'vendor_sha256':{k:sha(Path(v.__file__)) for k,v in mods.items()},
              'packages':{k:importlib.metadata.version(k) for k in ('fer','tensorflow-cpu','facenet-pytorch','torch','torchvision','numpy')},
              'fer_code_sha256':sha(Path(inspect.getfile(FER))),
              'fer_model_sha256':sha(fer_root/'data/emotion_model_quantized.tflite'),
              'mtcnn_model_sha256':{p.name:sha(p) for p in sorted(mtcnn_root.glob('*.pt'))},
              'fer':{'mtcnn':True,'use_tflite':True,'mtcnn_device':'cuda' if args.gpu else 'cpu',
                     'gpu_memory_fraction':0.04 if args.gpu else None,
                     'face_selection':'maximum IoU with same-frame MediaPipe box, otherwise largest; all detections preserved',
                     'frames':'every decoded frame, no sampling or dropped no-face rows','inference_workers':1,'cpu_threads':2},
              'combine':'independent multi-label channels; no winner-takes-all; original VLM records unchanged',
              'human_ground_truth':False,'training_admitted':False}
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if args.shard==0:save(out/'protocol.json',protocol)
    ids=json.loads((nod_run/'samples.json').read_text())['ids'][:args.limit]
    if args.shard==0:save(out/'samples.json',{'ids':ids})
    if args.shards>1:ids=ids[args.shard::args.shards]
    results=[]; started=time.monotonic()
    for sid in ids:
        target=out/'results'/f'{sid}.json'
        nod_path=nod_run/'results'/f'{sid}.json'
        nod=json.loads(nod_path.read_text())
        video=Path(nod['video']);pkl=nod_run/'landmarks'/f'{sid}.pkl'
        hashes={'video':sha(video),'pose':sha(pkl),'nod_result':sha(nod_path)}
        assert hashes['video']==nod['video_sha256'] and nod['status']=='ok'
        if target.exists():
            old=json.loads(target.read_text())
            if old.get('fingerprint')==fingerprint and old.get('input_hashes')==hashes and old.get('status')=='ok':
                results.append(old);print(json.dumps({'id':sid,'resumed':True}),flush=True);continue
        r={'id':sid,'status':'error','fingerprint':fingerprint,'input_hashes':hashes,
           'frames':nod['frames'],'fps':nod['fps'],'duration':nod['duration'],'training_admitted':False}
        tick=time.monotonic()
        try:
            with pkl.open('rb') as f: poses=pickle.load(f)
            assert len(poses)==nod['frames']
            cap=cv2.VideoCapture(str(video));assert cap.isOpened()
            raw=[];rows=[];fer_detected=[];i=0
            try:
                while True:
                    good,frame=cap.read()
                    if not good:break
                    faces=detector.detect_emotions(frame)
                    faces=json.loads(json.dumps(faces,default=lambda x:x.tolist() if hasattr(x,'tolist') else float(x)))
                    box=poses[i]['bbox_px']
                    pick=max(range(len(faces)),key=lambda j:overlap(faces[j]['box'],box) if box else faces[j]['box'][2]*faces[j]['box'][3]) if faces else None
                    row={'frame_index':i,'detected':pick is not None}
                    row.update({k:faces[pick]['emotions'][k] if pick is not None else 0. for k in ('angry','disgust','fear','happy','sad','surprise','neutral')})
                    raw.append({'frame_index':i,'faces':faces,'selected':pick})
                    rows.append(row);fer_detected.append(int(pick is not None));i+=1
            finally:cap.release()
            assert i==nod['frames'], (i,nod['frames'])
            if not any(fer_detected):raise ValueError('FER detected no face: expression inference unavailable, not negative labels')
            save(out/'fer_raw'/f'{sid}.json',raw)
            fer_csv=out/'fer'/f'{sid}.csv';fer_csv.parent.mkdir(parents=True,exist_ok=True)
            with fer_csv.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            channels={'nodding':nod['profiles']['recall']}
            det=np.asarray(nod['pose']['detected'],dtype=np.float32)
            for name,mod in mods.items():
                cfg=parameters[name]
                if name=='head_shaking':
                    mask,pitch,yaw=mod.load_pose_from_pkl(pkl)
                    scores=mod.compute_head_shaking_intensity(mask,pitch,yaw,argparse.Namespace(**cfg))
                elif name=='smiling':
                    mask,smile=mod.load_smile_blendshape_scores(pkl)
                    scores=mod.compute_smiling_intensity(mod.load_fer_happy_scores(fer_csv),mask,smile,**cfg)
                elif name=='laughing':
                    mask,smile,jaw=mod.load_blendshape_signals(pkl)
                    scores=mod.compute_laughing_intensity(mod.load_fer_happy_scores(fer_csv),mask,smile,jaw,**cfg)
                elif name=='frowning':
                    sad,angry=mod.load_fer_emotion_scores(fer_csv)
                    mask,mouth,brow,au=mod.load_blendshape_signals(pkl)
                    scores=mod.compute_frowning_intensity(sad,angry,mask,mouth,brow,au,**cfg)
                else:
                    mask,jaw,eye,brow=mod.load_blendshape_signals(pkl)
                    scores=mod.compute_surprised_intensity(mod.load_fer_surprise_scores(fer_csv),mask,jaw,eye,brow,**cfg)
                assert len(scores)==nod['frames'] and np.isfinite(scores).all()
                dst=out/'scores'/name/f'{sid}.csv';dst.parent.mkdir(parents=True,exist_ok=True)
                mod.write_intensity_csv(dst,scores)
                events=intervals(scores,det,nod['fps'],nod['duration'])
                if name not in ('head_shaking',):
                    for e in events:
                        a,b=e['start_frame'],e['end_frame_exclusive']
                        e['fer_detection_fraction']=sum(fer_detected[a:b])/(b-a)
                channels[name]={'events':events,'scores':[float(x) for x in scores]}
            supplementary=[]
            for name in TYPES:
                for j,e in enumerate(channels[name]['events']):
                    supplementary.append({**e,'event_id':f'glare-{name}-{j+1:03d}', 'type':name,
                                          'description':DESCRIPTIONS[name],'modality':'visual',
                                          'source':'glare_dedicated_detector','status':'candidate','text_kind':'detector_label_not_vlm_quote',
                                          'configuration':'recall' if name=='nodding' else 'vendor_python_default',
                                          'training_admitted':False})
            r.update(status='ok',channels=channels,supplementary_events=sorted(supplementary,key=lambda e:(e['start'],e['end'],e['type'])),
                     fer_detected=fer_detected,fer_detection_fraction=sum(fer_detected)/len(fer_detected),
                     face_detection_fraction=nod['face_detection_fraction'])
        except Exception as exc:
            import traceback
            r['error']=repr(exc);r['traceback']=traceback.format_exc()
        r['elapsed_s']=time.monotonic()-tick;save(target,r);results.append(r)
        print(json.dumps({'completed':len(results),'total':len(ids),'id':sid,'status':r['status'],'error':r.get('error'),
                          'elapsed_s':r['elapsed_s'],'events':{k:len(v['events']) for k,v in r.get('channels',{}).items()}}),flush=True)
    ok=[r for r in results if r['status']=='ok']
    summary={'samples':len(ids),'ok':len(ok),'errors':[r['id'] for r in results if r['status']!='ok'],
             'wall_s':time.monotonic()-started,'channels':{k:{'clips':sum(bool(r['channels'][k]['events']) for r in ok),
                         'intervals':sum(len(r['channels'][k]['events']) for r in ok),
                         'seconds':sum(e['end']-e['start'] for r in ok for e in r['channels'][k]['events'])} for k in TYPES}}
    summary_path=out/'summary.json' if args.shards==1 else out/f'summary_shard{args.shard}.json'
    events_path=out/'supplementary_events.jsonl' if args.shards==1 else out/f'supplementary_events_shard{args.shard}.jsonl'
    save(summary_path,summary)
    with events_path.open('w') as f:
        for r in ok:f.write(json.dumps({'sample_id':r['id'],'source_fingerprint':fingerprint,'training_admitted':False,'events':r['supplementary_events']})+'\n')
    print(json.dumps(summary),flush=True)
    if summary['errors']:raise SystemExit(1)

if __name__=='__main__':main()
