"""Frozen six-clip paired intervention: silence WAV CFG2 versus dropped branch."""
import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
RUN=ROOT/'experiments/p2_avatarforcing_table4_recheck_20261110'
DEST=WORK/'conditioning_ablation'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_avatarforcing_table4_generate import guard,read,save,sha


def prepare():
    p=read(RUN/'protocol.json');samples={s['id']:s for s in p['samples']}
    test=[r for r in csv.DictReader(Path('/public/0_DATA/2_Avatar/ViCo/dataset/metadata.csv').open()) if r['data_split']=='test']
    cases=[];used=set()
    for attitude in ['negative','neutral','positive']:
        pool=sorted([r for r in test if r['attitude']==attitude and r['uuid']!='YFhkBtIUgBM_000106_000107.0'],key=lambda r:(samples[r['uuid']]['duration'],r['uuid']))
        for fraction in [.25,.75]:
            index=int((len(pool)-1)*fraction)
            row=pool[index]
            if row['audio'] in used:
                row=next(r for r in pool if r['audio'] not in used)
            used.add(row['audio']);s=samples[row['uuid']]
            cases.append(dict(id=row['uuid'],attitude=attitude,duration=s['duration'],selection_quantile=fraction))
    plan=dict(cases=cases,selection='Two predeclared duration quartiles in each of3 attitudes; no duplicated audio; exclude already observed1second case; no generated-score selection',
        seed=137,user_cfg=1,nfe=10,avatar_cfg=[2,0],paired_same_process_same_inputs=True,
        original_protocol_sha256=sha(RUN/'protocol.json'),implementation_sha256=sha(__file__),
        scientific_scope='Diagnostic intervention, not asserted to be the authors Table4 listening configuration')
    dest=DEST/'protocol.json'
    if dest.exists():assert read(dest)==plan
    else:save(dest,plan)
    print(plan,flush=True)


def verify():
    p=read(DEST/'protocol.json');assert p['implementation_sha256']==sha(__file__)
    assert p['original_protocol_sha256']==sha(RUN/'protocol.json');return p


def generate():
    import cv2,soundfile as sf,torch,types
    from transformers import Wav2Vec2FeatureExtractor
    import vico_avatarforcing as adapter
    from p2_avatarforcing_table4_generate import verify as verify_original
    p=verify();original=verify_original();device=guard(.15,49152);cv2.setNumThreads(1)
    save(DEST/'GENERATION_DEVICE.json',device);adapter.WORK=DEST;model=adapter.build_model()
    model.encode_user_motion=types.MethodType(type(model).encode_user_motion,model)
    pp=Wav2Vec2FeatureExtractor.from_pretrained(model.opt.wav2vec_model_path,local_files_only=True)
    samples={s['id']:s for s in original['samples']};done=0
    for case in p['cases']:
        s=samples[case['id']];folder=RUN/'inputs'/s['id'];wave,sr=sf.read(s['speaker'],dtype='float32');assert sr==16000
        count=int(np.ceil(len(wave)/640));paths=sorted((folder/'speaker25').glob('*.jpg'));paths=(paths+[paths[-1]])[:count]
        def tensor(path):
            a=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            return torch.from_numpy(((a.astype('float32')-127.5)/127.5).transpose(2,0,1).copy()).unsqueeze(0)
        data=dict(avatar_ref=tensor(folder/'reference.png'),avatar_a=pp(np.zeros_like(wave),sampling_rate=sr,return_tensors='pt').input_values,
            user_a=pp(wave,sampling_rate=sr,return_tensors='pt').input_values,user_frame=[tensor(path) for path in paths])
        for cfg in p['avatar_cfg']:
            name=f'avatar_cfg{cfg}';output=DEST/name/s['id'];marker=output/'complete.json'
            if marker.exists():
                r=read(marker);assert sha(output/'generated.mp4')==r['sha256'];done+=1;continue
            start=time.monotonic();torch.manual_seed(p['seed']);np.random.seed(p['seed'])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                value=model.inference(dict(data),a_cfg_scale=cfg,u_cfg_scale=p['user_cfg'],nfe=p['nfe'],seed=p['seed'],use_kv_cache=True)['d_hat'].float().clamp(-1,1).cpu()
            rgb=((value-value.min())/(value.max()-value.min())*255).byte().permute(0,2,3,1).numpy()
            adapter.export_rgb(output/'generated.mp4',[rgb],512,25,18)
            save(marker,dict(id=s['id'],avatar_cfg=cfg,frames=count,sha256=sha(output/'generated.mp4'),protocol_sha256=sha(DEST/'protocol.json'),seconds=time.monotonic()-start))
            done+=1;save(DEST/'status.json',dict(stage='generation',completed=done,total=12,last=name+'/'+s['id']))
            print('generated',done,'/12',name,s['id'],flush=True)
            del value,rgb;torch.cuda.empty_cache()
    save(DEST/'GENERATION_COMPLETE.json',dict(passed=True,videos=12,protocol_sha256=sha(DEST/'protocol.json')))


def extract():
    import cv2,face_alignment
    from p2_spectre_coefficients import Encoder,decode25,crop_frames
    p=verify();assert read(DEST/'GENERATION_COMPLETE.json')['passed'];guard(.08,32768);cv2.setNumThreads(1)
    model=Encoder();detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device='cuda',flip_input=False);done=0
    for case in p['cases']:
        for cfg in p['avatar_cfg']:
            folder=DEST/f'avatar_cfg{cfg}'/case['id'];receipt=read(folder/'complete.json');path=folder/'spectre.npz'
            if path.exists():raise RuntimeError('Unexpected partial retry; preserve evidence and explicitly verify cached feature receipt')
            assert receipt['protocol_sha256']==sha(DEST/'protocol.json');assert sha(folder/'generated.mp4')==receipt['sha256']
            frames,_=decode25(folder/'generated.mp4',receipt['frames']);images,audit=crop_frames(frames,detector);v=model.chunks(images)
            np.savez_compressed(path,motion=v);save(folder/'spectre.json',dict(sha256=sha(path),frames=len(v),audit=audit,encoder=model.info))
            done+=1;save(DEST/'status.json',dict(stage='extraction',completed=done,total=12,last=f'cfg{cfg}/'+case['id']));print('extracted',done,'/12',flush=True)
    save(DEST/'EXTRACTION_COMPLETE.json',dict(passed=True,streams=12))


def report():
    from sklearn.cluster import KMeans
    from vico_literature_metrics import load_author
    p=verify();assert read(DEST/'EXTRACTION_COMPLETE.json')['passed'];author=load_author()
    full=read(RUN/'protocol.json');ids=[s['id'] for s in full['samples']]
    gt_all=[np.load(RUN/'features/gt'/(uid+'.npz'))['motion'] for uid in ids]
    ids=[c['id'] for c in p['cases']];g=[np.load(RUN/'features/gt'/(uid+'.npz'))['motion'] for uid in ids]
    x=np.concatenate([np.load(RUN/'features/speaker'/(uid+'.npz'))['motion'] for uid in ids]);results={};per_clip={}
    for group,sl,k in [('exp',slice(6,56),15),('pose',slice(0,6),9)]:
        cluster=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(np.concatenate(gt_all)[:,sl]);gc=np.concatenate(g)[:,sl]
        for cfg in p['avatar_cfg']:
            pred=[]
            for uid in ids:
                folder=DEST/f'avatar_cfg{cfg}'/uid;assert sha(folder/'spectre.npz')==read(folder/'spectre.json')['sha256'];pred.append(np.load(folder/'spectre.npz')['motion'])
            pc=np.concatenate(pred)[:,sl];fds=[];vars=[]
            for uid,a,b in zip(ids,g,pred):
                assert a.shape==b.shape
                fd=float(author.calculate_frechet_distance(*author.calculate_activation_statistics(a[:,sl]),*author.calculate_activation_statistics(b[:,sl])))
                var=float(np.var(b[:,sl],axis=0).mean());fds.append(fd);vars.append(var)
                per_clip.setdefault(uid,{}).setdefault(str(cfg),{})[group]=dict(FD=fd,temporal_variance_mean=var)
            h=np.bincount(cluster.predict(pc),minlength=k).astype(float);h/=h.sum()
            results.setdefault(str(cfg),{})[group]=dict(FD=float(np.mean(fds)),Var_equal_clip_temporal_mean=float(np.mean(vars)),
                Var_pooled_per_coefficient_mean=float(np.var(pc,axis=0).mean()),Var_pooled_flattened=float(np.var(pc)),
                SID_fixed_full43_GT_anchor=float(-np.sum(h*np.log2(h+1e-6))),
                rPCC=float(abs(np.corrcoef(gc.ravel(),x[:,sl].ravel())[0,1]-np.corrcoef(pc.ravel(),x[:,sl].ravel())[0,1])))
    save(DEST/'RESULTS.json',dict(paired_clips=6,frames=sum(map(len,g)),cases=p['cases'],metrics=results,per_clip=per_clip,
        diagnostic_only=True,author_listening_configuration_confirmed=False,protocol_sha256=sha(DEST/'protocol.json')))
    save(DEST/'status.json',dict(stage='complete',pairs=6,videos=12));print(results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','generate','extract','report']);args=parser.parse_args()
    globals()[args.action]()
