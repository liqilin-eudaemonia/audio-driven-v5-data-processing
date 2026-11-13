"""Official four L2L models, fixed 25-clip batches and existing GPU-resident input optimization."""
import time,argparse,math,types,os
import numpy as np
from runtime import *
NAMES=['conan','fallon','stephen','trevor']

def batches():
    for dataset in ['responsenet','seamless']:
        ss=[s for s in selected() if s['dataset']==dataset]
        for i in range(8):yield dataset,i,ss[25*i:25*(i+1)]

def complete(name,dataset,index,ss):
    marker=WORK/'l2l_batches'/name/dataset/f'{index:02d}.json'
    if not marker.exists():return False
    r=read(marker)
    return r['ids']==[s['id'] for s in ss] and all(Path(p).exists() and sha(p)==h for p,h in r['files'].items())

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--limit-batches',type=int);a=p.parse_args()
    loaded={};count=0
    while True:
        pending=False;processed=0
        for dataset,index,ss in batches():
            todo=[n for n in NAMES if not complete(n,dataset,index,ss)]
            if not todo:continue
            pending=True
            ff=[WORK/'features'/s['dataset']/s['id'] for s in ss]
            if not all((f/'complete.json').exists() for f in ff):continue
            lockpath=WORK/'l2l_batches'/dataset/f'{index:02d}.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
            with lockpath.open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if not loaded:
                    device()
                    import torch
                    from vico_l2l import REPO,aligned_window
                    from vico_l2l_fast import install_resident_inputs
                    sys.path.insert(0,str(REPO));os.chdir(REPO)
                    from modules.fact_model import setup_model
                    from vqgan.vqmodules.gan_models import setup_vq_transformer
                    from utils.load_utils import bilateral_filter
                    import test_vq_decoder
                    install_resident_inputs(test_vq_decoder)
                    args=types.SimpleNamespace(sample_idx=None)
                for f in ff:
                    for path,h in read(f/'complete.json')['files'].items():assert sha(path)==h
                speaker=[bilateral_filter(np.load(f/'speaker_deca.npz')['motion'][None].astype(np.float32))[0] for f in ff]
                mel=[np.load(f/'mel128.npy') for f in ff];lens=[len(x) for x in speaker];padded=(max(lens)+31)//32*32;batch=len(ss)
                for name in todo:
                    if complete(name,dataset,index,ss):continue
                    if name not in loaded:
                        cp=REPO/'configs/vq'/('delta_v6.json' if name=='conan' else name+'_delta_v6.json');config=read(cp);vc=read(REPO/config['l_vqconfig'])
                        wp=REPO/config['model_path']/(config['tag']+config['pipeline']+'_best.pth');vp=REPO/'vqgan'/vc['model_path']/(vc['tag']+vc['pipeline']+'_best.pth')
                        xp=REPO/config['model_path']/(config['tag']+config['pipeline']+'_preprocess_core.npz');yp=REPO/'vqgan'/vc['model_path']/(vc['tag']+vc['pipeline']+'_preprocess_core.npz')
                        vq,vo,_=setup_vq_transformer(args,vc,load_path=str(vp),test=True);vq.eval()
                        model,mo,_=setup_model(config,vc,mask_index=0,test=True,load_path=str(wp));model.eval()
                        loaded[name]=(config,vq,model,dict(np.load(xp)),dict(np.load(yp)),{str(p):sha(p) for p in [cp,wp,vp,xp,yp]})
                        del vo,mo
                    config,vq,model,xs,ys,assets=loaded[name]
                    torch.manual_seed(137);np.random.seed(137)
                    mx,sx=xs['body_mean_X'].reshape(56),xs['body_std_X'].reshape(56);ma,sa=xs['body_mean_audio'].reshape(128),xs['body_std_audio'].reshape(128)
                    x=np.zeros((batch,padded+32,56),np.float32);audio=np.zeros((batch,(padded+32)*4,128),np.float32)
                    for j,n in enumerate(lens):
                        normalized=(speaker[j]-mx)/sx;x[j,32:32+n]=normalized;x[j,32+n:]=normalized[-1]
                        audio[j,128:128+n*4]=(mel[j]-ma)/sa;audio[j,128+n*4:]=(0-ma)/sa
                    x=torch.from_numpy(x).cuda();audio=torch.from_numpy(audio).cuda();history=torch.zeros((batch,32,56),device='cuda');tokens=None;shape=None;started=time.time()
                    with torch.no_grad():
                        for t in range(0,padded,8):
                            past=history if tokens is None else tokens[:,-4:];xw,aw=aligned_window(x,audio,t)
                            prediction,prob,inputs,shape=test_vq_decoder.generate_prediction(config,args,vq,model,xw,past,aw,32,8,t//8,4,btc=shape)
                            if tokens is None:tokens=inputs['listener_past']
                            tokens=torch.cat((tokens,prediction[:,0]),1)
                        codes=tokens[:,4:];decoded=[vq.module.decode_to_img(codes[:,t:t+4],shape).cpu().numpy() for t in range(0,codes.shape[1],4)]
                    pred=np.concatenate(decoded,1)*ys['body_std_Y']+ys['body_mean_Y'];assert pred.shape==(batch,padded,56) and np.isfinite(pred).all()
                    files={}
                    for j,s in enumerate(ss):
                        d=folder('l2l_'+name,s);d.mkdir(parents=True,exist_ok=True);path=d/'motion.npz'
                        np.savez_compressed(path,motion=pred[j,:lens[j]].astype(np.float32),token_indices=codes[j].cpu().numpy(),native_fps=30)
                        files[str(path)]=sha(path)
                        save(d/'MOTION_COMPLETE.json',dict(id=s['id'],key=s['key'],files={str(path):files[str(path)]},source_only=True,native_fps=30,frames=lens[j],seed=137,batch_size=25,batch_ids=[v['id'] for v in ss],assets=assets,source_feature_receipt_sha256=sha(ff[j]/'complete.json'),code_sha256=sha(__file__)))
                    save(WORK/'l2l_batches'/name/dataset/f'{index:02d}.json',dict(passed=True,ids=[s['id'] for s in ss],files=files,assets=assets,seconds=time.time()-started))
                    print('L2L',name,dataset,index,round(time.time()-started,2),flush=True)
                    del x,audio,history,tokens,codes,decoded,pred
                count+=1;processed+=1
                if a.limit_batches and count>=a.limit_batches:return
        if not pending or not a.wait:return
        if not processed:time.sleep(10)
if __name__=='__main__':main()
