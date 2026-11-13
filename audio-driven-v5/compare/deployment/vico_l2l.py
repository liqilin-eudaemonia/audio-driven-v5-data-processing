"""Four official person-specific L2L weights, source-only ViCo motion adaptation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'compare/learning2listen/src'
WORK=ROOT/'experiments/p2_compare_vico43_20261106/learning2listen'
BASE=ROOT/'experiments/p2_compare_vico43_20261106/dystream'
NAMES=['conan','fallon','stephen','trevor']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def aligned_window(speaker,audio,t):
    """One40-frame speaker window and its exact160-mel counterpart."""
    x=speaker[:,t:t+40];a=audio[:,4*t:4*(t+40)]
    if x.shape[1]!=40 or a.shape[1]!=160:raise ValueError('Incomplete aligned source window')
    return x,a


def audio_features(item):
    import librosa
    from PIL import Image
    path=WORK/'audio_features'/(item['id']+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return np.load(path)
    wave,sr=librosa.load(item['speaker'],sr=16000)
    # Author's corrected method in issue2: raw mel power, NOT MFCC/log-mel.
    mel=librosa.feature.melspectrogram(y=wave,sr=sr,hop_length=160,win_length=400,n_fft=512,n_mels=128)
    resized=np.array(Image.fromarray(mel).resize((item['frames']*4,128),Image.Resampling.LANCZOS)).T.astype(np.float32)
    assert resized.shape==(item['frames']*4,128) and np.isfinite(resized).all()
    np.save(path,resized);return resized


def generate(smoke=False):
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    assert (WORK/'DECA_COMPLETE.json').exists()
    sys.path.insert(0,str(REPO));os.chdir(REPO)
    from modules.fact_model import setup_model
    from vqgan.vqmodules.gan_models import setup_vq_transformer
    from test_vq_decoder import generate_prediction
    from utils.load_utils import bilateral_filter
    p=json.loads((BASE/'protocol.json').read_text());items=p['samples']
    if smoke:items=[min(items,key=lambda x:x['duration'])]
    torch.set_num_threads(2);args=types.SimpleNamespace(sample_idx=None)
    configs={n:REPO/'configs/vq'/('delta_v6.json' if n=='conan' else n+'_delta_v6.json') for n in NAMES}
    code=[Path(__file__),REPO/'test_vq_decoder.py',REPO/'utils/load_utils.py',REPO/'vqgan/vqmodules/gan_models.py',REPO/'modules/fact_model.py']
    protocol=dict(model='Learning2Listen four person-specific pretrained weights',dataset='ViCo paper test43',fps=30,
        feature_space='official DECA expression50 + global head rotation3 + jaw rotation3; not BFM64/translation',
        audio='author issue2 corrected melspectrogram: sr16000 win400 hop160 nfft512 n_mels128 power2; PIL Lanczos resize to4T; no log/MFCC',
        audio_reference='https://github.com/evonneng/learning2listen/issues/2',
        model_names=NAMES,seed=137,batch_size=len(items),
        adaptation='user-approved corrected audio slice4*t:4*(t+40); unchanged official predictor, VQ and nucleus sampling',
        history='32 standardized zero frames before clip origin for listener/speaker/audio; no target listener input at any time',
        tail='speaker edge pad; zero raw mel pad; target length padded to multiple32 for VQ decoding; only original frames scored',
        loop='8-frame/one-token AR; past4 tokens decoded/requantized by original create_data_vq; decode generated tokens in32-frame blocks',
        source_only=True,listener_future_input=False,photorealistic_renderer=False,
        deca_protocol_sha256=sha(WORK/'DECA_PROTOCOL.json'),code_sha256={str(q):sha(q) for q in code},samples=items,models={})
    mel=[audio_features(s) for s in items]
    # Read ONLY speaker features in this generation process. Listener GT arrays
    # are opened later by a separate CPU scoring command.
    speaker=[np.load(WORK/'features/speaker'/(s['id']+'.npz'))['motion'].astype(np.float32) for s in items]
    length=max(s['frames'] for s in items);padded=(length+31)//32*32;batch=len(items)
    start=time.time()
    for name in (['conan'] if smoke else NAMES):
        if not smoke and (WORK/'generated'/name/'COMPLETE.json').exists():continue
        torch.manual_seed(137);np.random.seed(137)
        config=json.loads(configs[name].read_text());vqconfig=json.loads((REPO/config['l_vqconfig']).read_text())
        checkpoint=REPO/config['model_path']/(config['tag']+config['pipeline']+'_best.pth')
        vqcheckpoint=REPO/'vqgan'/vqconfig['model_path']/(vqconfig['tag']+vqconfig['pipeline']+'_best.pth')
        xstats_path=REPO/config['model_path']/(config['tag']+config['pipeline']+'_preprocess_core.npz')
        ystats_path=REPO/'vqgan'/vqconfig['model_path']/(vqconfig['tag']+vqconfig['pipeline']+'_preprocess_core.npz')
        xs=np.load(xstats_path);ys=np.load(ystats_path)
        mean_x,std_x=xs['body_mean_X'].reshape(56),xs['body_std_X'].reshape(56)
        mean_a,std_a=xs['body_mean_audio'].reshape(128),xs['body_std_audio'].reshape(128)
        for stat in [xs['body_std_X'],xs['body_std_audio'],ys['body_std_Y']]:assert np.isfinite(stat).all() and (stat>0).all()
        protocol['models'][name]=dict(config=str(configs[name]),vq_config=config['l_vqconfig'],files={str(q):sha(q) for q in [checkpoint,vqcheckpoint,xstats_path,ystats_path]})
        save(WORK/('SMOKE_PROTOCOL.json' if smoke else 'protocol.json'),protocol)
        vq,vopt,_=setup_vq_transformer(args,vqconfig,load_path=str(vqcheckpoint),test=True);vq.eval()
        generator,gopt,_=setup_model(config,vqconfig,mask_index=0,test=True,load_path=str(checkpoint));generator.eval()
        x=np.zeros((batch,padded+32,56),np.float32);a=np.zeros((batch,(padded+32)*4,128),np.float32)
        for i,s in enumerate(items):
            n=s['frames'];sx=bilateral_filter(speaker[i][None])[0]
            xn=(sx-mean_x)/std_x;an=(mel[i]-mean_a)/std_a
            x[i,32:32+n]=xn;x[i,32+n:]=xn[-1]
            a[i,128:128+n*4]=an
            a[i,128+n*4:]=(0-mean_a)/std_a
        history=np.zeros((batch,32,56),np.float32);tokens=None;shape=None;probabilities=[];alignment=[]
        with torch.no_grad():
            for t in range(0,padded,8):
                past=history if tokens is None else tokens[:,-4:].cpu().numpy()
                xw,aw=aligned_window(x,a,t)
                prediction,prob,inputs,shape=generate_prediction(config,args,vq,generator,xw,past,aw,32,8,t//8,4,btc=shape)
                if tokens is None:tokens=inputs['listener_past']
                tokens=torch.cat((tokens,prediction[:,0]),dim=1)
                probabilities.append(prob.cpu().numpy());alignment.append(dict(speaker_start=t-32,audio_mel_start=4*(t-32),speaker_frames=40,audio_mel_frames=160))
                save(WORK/'status.json',dict(stage='generating',model=name,generated_frames=min(t+8,padded),padded_frames=padded,samples=batch,elapsed_s=time.time()-start))
            code=tokens[:,4:];assert code.shape==(batch,padded//8)
            decoded=[]
            for start_token in range(0,code.shape[1],4):decoded.append(vq.module.decode_to_img(code[:,start_token:start_token+4],shape).cpu().numpy())
        pred=np.concatenate(decoded,axis=1)*ys['body_std_Y']+ys['body_mean_Y']
        assert pred.shape==(batch,padded,56) and np.isfinite(pred).all()
        folder=WORK/('smoke' if smoke else 'generated')/name;folder.mkdir(parents=True,exist_ok=True)
        for i,s in enumerate(items):
            values=pred[i,:s['frames']].astype(np.float32);dest=folder/(s['id']+'.npz')
            np.savez_compressed(dest,motion=values,token_indices=code[i].cpu().numpy())
            save(dest.with_suffix('.json'),dict(id=s['id'],model=name,frames=len(values),fps=30,space='DECA50+3+3',
                source_only=True,listener_future_input=False,feature_sha256=sha(dest)))
        save(folder/'alignment.json',alignment)
        save(folder/'COMPLETE.json',dict(model=name,samples=batch,frames=sum(s['frames'] for s in items),padded_frames=padded,seed=137))
        del vq,generator,vopt,gopt,tokens,code;torch.cuda.empty_cache()
    if smoke:save(WORK/'SMOKE_COMPLETE.json',dict(model='conan',samples=1,frames=items[0]['frames'],finite=True))
    else:save(WORK/'GENERATION_COMPLETE.json',dict(models=NAMES,outputs=172,frames_per_model=sum(s['frames'] for s in items),seconds=time.time()-start))


def score():
    sys.path.insert(0,str(ROOT/'scripts'));from p2_vico_motion_metrics import frechet,tlcc
    p=json.loads((WORK/'protocol.json').read_text());all_results={}
    for name in NAMES:
        rows=[]
        for item in p['samples']:
            uid=item['id'];s=np.load(WORK/'features/speaker'/(uid+'.npz'))['motion'].astype(float)
            g=np.load(WORK/'features/gt'/(uid+'.npz'))['motion'].astype(float)
            pred=np.load(WORK/'generated'/name/(uid+'.npz'))['motion'].astype(float)
            assert pred.shape==g.shape==s.shape==(item['frames'],56)
            metrics={k:{} for k in ['L1','L2','Frechet','Delta_Frechet','SND','V_D','V_D_GT','RTLCC','RWTLCC']}
            for group,sl in [('exp',slice(0,50)),('head_rotation',slice(50,53)),('jaw_rotation',slice(53,56))]:
                a,b,c=s[:,sl],g[:,sl],pred[:,sl]
                metrics['L1'][group]=float(np.abs(b-c).mean());metrics['L2'][group]=float(np.linalg.norm(b-c,axis=1).mean())
                f=frechet(b,c);d=frechet(np.diff(b,axis=0),np.diff(c,axis=0));metrics['Frechet'][group]=f;metrics['Delta_Frechet'][group]=d;metrics['SND'][group]=f+d
                metrics['V_D'][group]=float(np.var(c,axis=0).mean());metrics['V_D_GT'][group]=float(np.var(b,axis=0).mean())
                metrics['RTLCC'][group]=float(np.abs(tlcc(a,b)[0]-tlcc(a,c)[0]).mean()) if len(a)>61 else None
                windows=[float(np.abs(tlcc(a[t:t+120],b[t:t+120])[0]-tlcc(a[t:t+120],c[t:t+120])[0]).mean()) for t in range(0,len(a)-119,30)]
                metrics['RWTLCC'][group]=float(np.mean(windows)) if windows else None
            rows.append(dict(id=uid,frames=len(g),metrics=metrics))
        avg={k:{group:float(np.mean([r['metrics'][k][group] for r in rows if r['metrics'][k][group] is not None])) for group in rows[0]['metrics'][k]} for k in rows[0]['metrics']}
        coverage={k:{group:sum(r['metrics'][k][group] is not None for r in rows) for group in avg[k]} for k in avg}
        all_results[name]=dict(clips=43,space='DECA50expression+3head_rotation+3jaw_rotation, distinct from BFM results',metrics=avg,coverage=coverage,per_clip=rows)
    save(WORK/'RESULTS.json',dict(schema='l2l-vico43-deca-motion-only-v1',models=all_results,photorealistic_metrics=None,source_only=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','generate','score','run']);a=p.parse_args()
    if a.command=='score':score()
    elif a.command=='run':generate();score()
    else:generate(a.command=='smoke')
