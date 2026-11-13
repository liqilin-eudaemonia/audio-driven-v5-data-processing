"""Locally retrain L2L with an approved source-grouped ViCo split.

This is a documented local training protocol, not a reproduction of unpublished
DIM baseline settings. Test media enter only after validation selects both
checkpoints. Original generation/checkpoint files are never modified.
"""
import argparse
import concurrent.futures
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vico_common_deca import crop_frame,encode,extractor,read,save,sha,clip_metrics,GROUPS
from vico_l2l_train_pilot import prepare as prepare_split

WORK=ROOT/'experiments/common_method_comparison_20261106/l2l_retrained'
PILOT=WORK.parent/'l2l_training_pilot'
L2L=ROOT/'experiments/p2_compare_vico43_20261106/learning2listen'
REPO=ROOT/'compare/learning2listen/src'
DATA=Path('/public/0_DATA/2_Avatar/ViCo/dataset')
SEED=137


def prepare():
    split=prepare_split()
    assert len(split['train'])==351 and len(split['validation'])==46 and len(split['quarantined_training_rows'])==1
    protocol=dict(schema='l2l-vico-local-retrain-v1',seed=SEED,split_sha256=sha(PILOT/'split.json'),
        initialization='from scratch; official default architecture only; no person checkpoint selection',
        split=dict(train=351,validation=46,quarantined=1,test=43,ood=42),
        feature_space='DECA50 expression+3 global rotation+3 jaw rotation',
        extraction='all frames of train/validation pair; independent FAN68; same DECA coarse encoder and crop as shared evaluation',
        audio='official metadata.audio WAV; trim/zero-pad tail to video duration without stretching; mel power128 sr16000 win400 hop160 nfft512, Lanczos time resize4T; tail adjustment recorded per clip',
        windows=dict(frames=64,train_stride=32,validation_stride=64,tail='append last complete window; no padding; clips<64 skipped and reported'),
        smoothing='official bilateral_filter separately per complete clip before windowing',
        normalization='official mean_std_swap fit on train windows only; validation/test never fit statistics',
        architecture_files={str(p):sha(p) for p in [REPO/'configs/vq/delta_v6.json',REPO/'vqgan/configs/l2_32_smoothSS.json',
            REPO/'modules/fact_model.py',REPO/'vqgan/vqmodules/gan_models.py',REPO/'utils/load_utils.py',REPO/'test_vq_decoder.py']},
        train_recipe=dict(batch_size=64,max_steps_per_stage=20000,minimum_steps=2000,validation_interval=250,
            early_stop_patience_validations=15,lr=1e-4,betas=[.9,.98],eps=1e-9,
            vq_optimizer='AdamW weight_decay0.01',predictor_optimizer='Adam weight_decay0',
            schedule='constant LR; local ViCo recipe, differs from original100000-step warmup; not DIM original baseline settings',
            gradient_clip_norm=1.,validation_predictor_history='fully masked; validation target tokens used only for loss'),
        checkpoint_selection='minimum validation loss per stage; never compare candidate checkpoints on test',
        test=dict(generation='existing approved4t source-only AR adapter,32 zero history,seed137,one candidate',
            scoring='same raw DECA metric functions as common evaluation; native L2L coefficients',
            photorealistic_renderer=False,official_test_used_during_training=False),
        resource=dict(host='172.29.189.119',physical_gpu=7),
        author_protocol_reproduction=False,code_sha256=sha(__file__))
    if (WORK/'protocol.json').exists():assert read(WORK/'protocol.json')==protocol
    else:save(WORK/'protocol.json',protocol)
    save(WORK/'split.json',split)
    return protocol,split


def frames(path):
    import cv2
    cap=cv2.VideoCapture(str(path));assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-30)<1e-6
    out=[]
    try:
        while True:
            ok,im=cap.read()
            if not ok:break
            out.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    assert out
    return np.stack(out)


def feature_worker(shard,shards):
    import torch
    import face_alignment
    import librosa
    from PIL import Image
    protocol,split=read(WORK/'protocol.json'),read(WORK/'split.json')
    assert sha(__file__)==protocol['code_sha256']
    encoder=extractor()
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device='cuda',flip_input=False)
    jobs=[(name,r) for name in ['train','validation'] for r in split[name]]
    for name,row in jobs[shard::shards]:
        started=time.time();uid=row['uuid'];dest=WORK/'features'/name/(uid+'.npz');receipt=dest.with_suffix('.json')
        if receipt.exists():
            assert sha(dest)==read(receipt)['feature_sha256'];continue
        result,inputs,missing={}, {}, {}
        for role in ['speaker','listener']:
            path=DATA/'by_split/train/videos'/(uid+'.'+role+'.mp4')
            images=frames(path);inputs[role]=dict(path=str(path),sha256=sha(path))
            boxes=[]
            with torch.no_grad():
                for start in range(0,len(images),32):
                    x=torch.from_numpy(images[start:start+32].transpose(0,3,1,2).copy()).float().cuda()
                    boxes.extend(fa.face_detector.detect_from_batch(x))
                landmarks=[fa.get_landmarks_from_image(im,detected_faces=b[:1])[0] if len(b) else None for im,b in zip(images,boxes)]
            valid=[i for i,lm in enumerate(landmarks) if lm is not None]
            assert valid,f'No face: {uid}/{role}'
            missing[role]=len(images)-len(valid);previous=landmarks[valid[0]]
            for i,lm in enumerate(landmarks):
                if lm is not None:previous=lm
                else:landmarks[i]=previous
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                cropped=list(pool.map(crop_frame,zip(images,landmarks)))
            c=encode(encoder,cropped)
            result[role]=np.concatenate([c[:,150:200],c[:,200:206]],axis=1)
        assert result['speaker'].shape==result['listener'].shape
        path=DATA/'audios'/(row['audio']+'.wav');inputs['audio']=dict(path=str(path),sha256=sha(path))
        wave,sr=librosa.load(path,sr=16000)
        n=len(result['speaker']);audio_samples=len(wave);needed=round(n/30*sr)
        assert abs(audio_samples/sr-n/30)<.5,(uid,audio_samples/sr,n/30)
        if len(wave)<needed:wave=np.pad(wave,(0,needed-len(wave)))
        else:wave=wave[:needed]
        mel=librosa.feature.melspectrogram(y=wave,sr=sr,hop_length=160,win_length=400,n_fft=512,n_mels=128)
        result['audio']=np.array(Image.fromarray(mel).resize((4*n,128),Image.Resampling.LANCZOS)).T.astype(np.float32)
        assert all(np.isfinite(x).all() for x in result.values())
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**result)
        save(receipt,dict(id=uid,split=name,frames=n,inputs=inputs,missing_face_frames=missing,
            original_audio_samples=audio_samples,aligned_audio_samples=needed,audio_tail_adjustment_samples=needed-audio_samples,
            feature_sha256=sha(dest),protocol_sha256=sha(WORK/'protocol.json'),seconds=time.time()-started))
        print('feature',shard,name,uid,n,round(time.time()-started,2),flush=True)


def make_windows():
    sys.path.insert(0,str(REPO))
    from utils.load_utils import bilateral_filter,mean_std_swap
    split=read(WORK/'split.json');arrays={};coverage={}
    for name in ['train','validation']:
        windows={k:[] for k in ['speaker','listener','audio']};origins=[];skipped=[]
        for row in split[name]:
            uid=row['uuid'];dest=WORK/'features'/name/(uid+'.npz')
            assert sha(dest)==read(dest.with_suffix('.json'))['feature_sha256']
            f=np.load(dest);n=len(f['listener'])
            if n<64:skipped.append(dict(id=uid,frames=n));continue
            sx,sy=[bilateral_filter(f[k][None])[0] for k in ['speaker','listener']]
            starts=list(range(0,n-63,32 if name=='train' else 64))
            if starts[-1]!=n-64:starts.append(n-64)
            for start in starts:
                windows['speaker'].append(sx[start:start+64]);windows['listener'].append(sy[start:start+64]);windows['audio'].append(f['audio'][4*start:4*(start+64)])
                origins.append(dict(id=uid,start_frame=start))
        arrays[name]={k:np.stack(v).astype(np.float32) for k,v in windows.items()}
        coverage[name]=dict(pairs=len(split[name]),used_pairs=len(split[name])-len(skipped),windows=len(origins),skipped_short=skipped,window_origins=origins)
    stats={}
    for key in ['speaker','listener','audio']:
        mean,std=mean_std_swap(arrays['train'][key]);assert (std>0).all() and np.isfinite(std).all()
        stats[key+'_mean'],stats[key+'_std']=mean,std
        for name in arrays:arrays[name][key]=((arrays[name][key]-mean)/std).astype(np.float32)
    np.savez_compressed(WORK/'train_only_stats.npz',**stats)
    save(WORK/'WINDOWS.json',coverage)
    return arrays


def validate_gpu():
    # Shared extractor checks the newly authorized physical device UUID.
    model=extractor();del model


def fit():
    import torch
    from modules.fact_model import setup_model,calc_logit_loss
    from vqgan.vqmodules.gan_models import setup_vq_transformer,calc_vq_loss
    validate_gpu();torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(2)
    protocol=read(WORK/'protocol.json');recipe=protocol['train_recipe'];data=make_windows()
    config=read(REPO/'configs/vq/delta_v6.json');vqconfig=read(REPO/config['l_vqconfig'])
    vq,_,_=setup_vq_transformer(types.SimpleNamespace(),vqconfig,load_path=None)
    tensor=lambda x:torch.from_numpy(np.asarray(x,np.float32)).cuda()
    vq_data={name:np.concatenate([v['listener'][:,:32],v['listener'][:,32:]],axis=0) for name,v in data.items()}
    def run_stage(stage,model,training,validation,loss_fn):
        best_path=WORK/(stage+'_best.pth');history_path=WORK/(stage+'_history.json')
        if (WORK/(stage+'_COMPLETE.json')).exists():
            model.load_state_dict(torch.load(best_path,weights_only=False)['state_dict'],strict=True);return
        opt=(torch.optim.AdamW if stage=='vq' else torch.optim.Adam)(model.parameters(),lr=recipe['lr'],betas=tuple(recipe['betas']),eps=recipe['eps'])
        rng=np.random.RandomState(SEED);best=float('inf');stale=0;history=[];started=time.time()
        def evaluate():
            model.eval();total=0.;count=0
            with torch.no_grad():
                for start in range(0,len(validation),recipe['batch_size']):
                    indices=np.arange(start,min(start+recipe['batch_size'],len(validation)))
                    loss=loss_fn(indices,False);assert torch.isfinite(loss)
                    total+=loss.item()*len(indices);count+=len(indices)
            return total/count
        for step in range(1,recipe['max_steps_per_stage']+1):
            model.train();indices=rng.randint(len(training),size=recipe['batch_size'])
            opt.zero_grad(set_to_none=True);loss=loss_fn(indices,True);assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip_norm']);assert torch.isfinite(norm)
            opt.step()
            if step%recipe['validation_interval']==0 or step==1:
                value=evaluate();record=dict(step=step,train_loss=loss.item(),validation_loss=value,seconds=time.time()-started)
                history.append(record)
                if value<best:
                    best=value;stale=0
                    torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),step=step,validation_loss=value,
                        protocol_sha256=sha(WORK/'protocol.json'),stats_sha256=sha(WORK/'train_only_stats.npz')),best_path)
                else:stale+=1
                save(history_path,history);save(WORK/'status.json',dict(stage=stage,**record,best_validation_loss=best,stale_validations=stale))
                print(stage,record,'best',best,'stale',stale,flush=True)
                if step>=recipe['minimum_steps'] and stale>=recipe['early_stop_patience_validations']:break
        selected=torch.load(best_path,weights_only=False);model.load_state_dict(selected['state_dict'],strict=True)
        save(WORK/(stage+'_COMPLETE.json'),dict(selected_step=selected['step'],validation_loss=best,total_steps=step,
            early_stopped=step<recipe['max_steps_per_stage'],checkpoint_sha256=sha(best_path),seconds=time.time()-started))
    def vq_loss(indices,training):
        y=tensor(vq_data['train' if training else 'validation'][indices]);p,q=vq(y,None);return calc_vq_loss(p,y,q)
    run_stage('vq',vq,vq_data['train'],vq_data['validation'],vq_loss)
    vq.eval()
    for p in vq.parameters():p.requires_grad_(False)
    tokens={}
    with torch.no_grad():
        for name,v in data.items():
            past,future=[],[]
            for start in range(0,len(v['listener']),128):
                y=tensor(v['listener'][start:start+128]);_,pi=vq.module.get_quant(y[:,:32]);_,fi=vq.module.get_quant(y[:,32:]);past.append(pi.reshape(len(y),4));future.append(fi.reshape(len(y),4))
            tokens[name]=dict(past=torch.cat(past),future=torch.cat(future))
    save(WORK/'CODEBOOK_USAGE.json',{name:dict(unique_future_tokens=int(t['future'].unique().numel()),future_tokens=t['future'].numel()) for name,t in tokens.items()})
    torch.manual_seed(SEED)
    generator,_,_=setup_model(config,vqconfig,load_path=None)
    def predictor_loss(indices,training):
        name='train' if training else 'validation';v=data[name];t=tokens[name]
        x=dict(speaker_full=tensor(v['speaker'][indices,:40]),audio_full=tensor(v['audio'][indices,:160]),listener_past=t['past'][indices])
        p=generator(x,4,-1 if training else 0)
        return calc_logit_loss(p[:,:4],t['future'][indices])
    run_stage('predictor',generator,data['train']['listener'],data['validation']['listener'],predictor_loss)
    save(WORK/'TRAINING_COMPLETE.json',dict(selection='validation only',test_media_opened=False,
        vq=read(WORK/'vq_COMPLETE.json'),predictor=read(WORK/'predictor_COMPLETE.json'),
        codebook_usage=read(WORK/'CODEBOOK_USAGE.json'),protocol_sha256=sha(WORK/'protocol.json')))


def test():
    import torch
    from modules.fact_model import setup_model
    from vqgan.vqmodules.gan_models import setup_vq_transformer
    from test_vq_decoder import generate_prediction
    from utils.load_utils import bilateral_filter
    from vico_l2l import aligned_window
    assert (WORK/'TRAINING_COMPLETE.json').exists()
    validate_gpu();torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(2)
    config=read(REPO/'configs/vq/delta_v6.json');vqconfig=read(REPO/config['l_vqconfig'])
    vq,_,_=setup_vq_transformer(types.SimpleNamespace(),vqconfig,load_path=None)
    model,_,_=setup_model(config,vqconfig,load_path=None)
    for name,network in [('vq',vq),('predictor',model)]:
        checkpoint=WORK/(name+'_best.pth');assert sha(checkpoint)==read(WORK/(name+'_COMPLETE.json'))['checkpoint_sha256']
        network.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict'],strict=True);network.eval()
    stats=np.load(WORK/'train_only_stats.npz');items=read(L2L/'protocol.json')['samples']
    assert {s['id'] for s in items}==set(read(WORK/'split.json')['held_out_test_ids'])
    padded=(max(s['frames'] for s in items)+31)//32*32
    x=np.zeros((43,padded+32,56),np.float32);a=np.zeros((43,(padded+32)*4,128),np.float32)
    # No listener GT is opened anywhere in this inference block.
    for i,s in enumerate(items):
        n=s['frames'];speaker=np.load(L2L/'features/speaker'/(s['id']+'.npz'))['motion'];mel=np.load(L2L/'audio_features'/(s['id']+'.npy'))
        sx=(bilateral_filter(speaker[None])[0]-stats['speaker_mean'].reshape(56))/stats['speaker_std'].reshape(56)
        am=(mel-stats['audio_mean'].reshape(128))/stats['audio_std'].reshape(128)
        x[i,32:32+n]=sx;x[i,32+n:]=sx[-1];a[i,128:128+4*n]=am;a[i,128+4*n:]=(0-stats['audio_mean'].reshape(128))/stats['audio_std'].reshape(128)
    torch.manual_seed(SEED)
    history=np.zeros((43,32,56),np.float32);token=None;shape=None
    with torch.no_grad():
        for t in range(0,padded,8):
            past=history if token is None else token[:,-4:].cpu().numpy();xw,aw=aligned_window(x,a,t)
            pred,_,inputs,shape=generate_prediction(config,types.SimpleNamespace(sample_idx=None),vq,model,xw,past,aw,32,8,t//8,4,btc=shape)
            if token is None:token=inputs['listener_past']
            token=torch.cat((token,pred[:,0]),dim=1)
        code=token[:,4:];decoded=[]
        for start in range(0,code.shape[1],4):decoded.append(vq.module.decode_to_img(code[:,start:start+4],shape).cpu().numpy())
    output=np.concatenate(decoded,axis=1)*stats['listener_std']+stats['listener_mean'];assert np.isfinite(output).all()
    for i,s in enumerate(items):
        dest=WORK/'test_generated'/(s['id']+'.npz');dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,motion=output[i,:s['frames']].astype(np.float32))
    save(WORK/'TEST_GENERATION_COMPLETE.json',dict(clips=43,frames=16288,source_only=True,seed=SEED,candidates=1,selected_on_test=False))
    rows=[]
    # GT becomes accessible only in this subsequent scoring block.
    for s in items:
        uid=s['id'];source,gt=[np.load(L2L/'features'/role/(uid+'.npz'))['motion'].astype(float) for role in ['speaker','gt']]
        dest=WORK/'test_generated'/(uid+'.npz');prediction=np.load(dest)['motion'].astype(float)
        rows.append(dict(id=uid,frames=len(gt),metrics=clip_metrics(source,gt,prediction),prediction_sha256=sha(dest)))
    metrics={k:{g:float(np.mean([r['metrics'][k][g] for r in rows if r['metrics'][k][g] is not None])) for g in GROUPS} for k in rows[0]['metrics']}
    coverage={k:{g:sum(r['metrics'][k][g] is not None for r in rows) for g in GROUPS} for k in metrics}
    save(WORK/'RESULTS.json',dict(schema='l2l-vico-locally-retrained-v1',model='Learning2Listen locally retrained on conservative ViCo397',
        clips=43,frames=16288,metrics=metrics,coverage=coverage,per_clip=rows,space='DECA50+3+3',
        readout='native motion coefficients',training=read(WORK/'TRAINING_COMPLETE.json'),
        protocol_sha256=sha(WORK/'protocol.json'),strict_DIM_reproduction=False,photorealistic_metrics=None))


def audit():
    p=read(WORK/'protocol.json');split=read(WORK/'split.json');result=read(WORK/'RESULTS.json')
    assert sha(__file__)==p['code_sha256'] and sha(PILOT/'split.json')==p['split_sha256']
    assert result['clips']==43 and {r['id'] for r in result['per_clip']}==set(split['held_out_test_ids'])
    assert not set(r['uuid'] for r in split['train']+split['validation'])&set(split['held_out_test_ids']+split['held_out_ood_ids'])
    for name in ['vq','predictor']:
        selected=read(WORK/(name+'_COMPLETE.json'));history=read(WORK/(name+'_history.json'))
        assert selected['validation_loss']==min(r['validation_loss'] for r in history)
        assert sha(WORK/(name+'_best.pth'))==selected['checkpoint_sha256']
    for path,expected in p['architecture_files'].items():assert sha(path)==expected
    for metric,groups in result['coverage'].items():assert set(groups.values())=={42 if metric=='RTLCC' else 41 if metric=='RWTLCC' else 43}
    windows=read(WORK/'WINDOWS.json')
    save(WORK/'FINAL_AUDIT.json',dict(passed=True,manifest_train_pairs=351,manifest_validation_pairs=46,quarantined=1,
        train_pairs_with_complete_windows=windows['train']['used_pairs'],validation_pairs_with_complete_windows=windows['validation']['used_pairs'],
        test43_only_after_validation_selection=True,results_sha256=sha(WORK/'RESULTS.json'),
        training_windows=read(WORK/'WINDOWS.json')['train']['windows'],validation_windows=read(WORK/'WINDOWS.json')['validation']['windows']))


def run(shards):
    prepare();logs=[];processes=[];started=time.time()
    try:
        for i in range(shards):
            f=(WORK/f'features_{i:02d}.log').open('a');logs.append(f)
            processes.append(subprocess.Popen([sys.executable,__file__,'feature-worker','--shard',str(i),'--shards',str(shards)],stdout=f,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in processes];assert not any(codes),codes
    finally:
        for f in logs:f.close()
    save(WORK/'FEATURES_COMPLETE.json',dict(pairs=397,streams=794,seconds=time.time()-started))
    fit();test();audit()
    save(WORK/'COMPLETE.json',dict(status='complete',seconds=time.time()-started,results_sha256=sha(WORK/'RESULTS.json')))
    # Refresh presentation only after fully audited test results exist.
    for script in ['build_common_method_comparison.py','build_vico_three_method_review.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],check=True,cwd=ROOT)


if __name__=='__main__':
    sys.path.insert(0,str(REPO))
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','feature-worker','fit','test','audit','run']);parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=4);args=parser.parse_args()
    if args.command=='feature-worker':feature_worker(args.shard,args.shards)
    elif args.command=='run':run(args.shards)
    else:globals()[args.command]()
