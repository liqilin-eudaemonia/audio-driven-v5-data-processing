"""Official EMOCA-v2 coefficient-only path; immutable videos and resumable receipts.

The source encoder and coefficient decomposition are executed without renderer,
texture, loss networks or Lightning. All encoder state tensors must load strictly.
"""
import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from vico_literature_protocol import ROOT,WORK,EMOCA,read,save,sha


def methods_from_source(path,class_name,names,base,namespace):
    tree=ast.parse(path.read_text())
    source=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    body=[copy.deepcopy(n) for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(body)==len(names)
    new=ast.ClassDef(name=class_name,bases=[ast.Name(id='SourceBase',ctx=ast.Load())],keywords=[],body=body,decorator_list=[])
    namespace=dict(namespace,SourceBase=base)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[new],type_ignores=[])),str(path)+':coefficient-only', 'exec'),namespace)
    return namespace[class_name]


class Encoder:
    def __init__(self):
        import torch
        import yaml
        sys.path.insert(0,str(EMOCA))
        from gdl.models import ResNet as resnet
        # Initialization is fully overwritten by strict encoder checkpoint load.
        # Avoid irrelevant ImageNet downloads; architecture and forward unchanged.
        resnet.load_ResNet50Model=lambda:resnet.ResNet(resnet.Bottleneck,[3,4,6,3])
        from gdl.models.DecaEncoder import ResnetEncoder
        models=WORK/'assets/EMOCA_v2_lr_mse_20'
        configs=list(models.rglob('cfg.yaml'));assert len(configs)==1,configs
        config=yaml.safe_load(configs[0].read_text())
        c=config['detail']['model']
        assert c['n_exp']==50 and c['n_pose']==6 and c['expression_backbone']=='deca_clone',c
        ckpts=list((models/'detail').rglob('*.ckpt'))
        if not ckpts:ckpts=list(models.rglob('*.ckpt'))
        assert ckpts,'Missing official checkpoint'
        def loss(p):
            try:return float(p.stem.rsplit('=',1)[1])
            except (IndexError,ValueError):return float('inf')
        ckpt=min(sorted(ckpts),key=loss);assert np.isfinite(loss(ckpt)) or len(ckpts)==1
        state=torch.load(ckpt,map_location='cpu',weights_only=False)['state_dict']
        official=EMOCA/'gdl/models/DECA.py'
        deca=methods_from_source(official,'DECA',['_encode_flame','decompose_code'],torch.nn.Module,dict(torch=torch))
        exp=methods_from_source(official,'ExpDECA',['_encode_flame','decompose_code'],deca,dict(torch=torch))
        self.model=exp();self.model.config=SimpleNamespace(**c)
        nparam=sum(c[k] for k in ['n_shape','n_tex','n_exp','n_pose','n_cam','n_light'])
        nexp=c['n_exp']+3*int(c['exp_deca_global_pose'])+3*int(c['exp_deca_jaw_pose'])
        self.model.E_flame=ResnetEncoder(nparam)
        self.model.E_expression=ResnetEncoder(nexp)
        loaded={}
        for name in ['E_flame','E_expression']:
            prefix='deca.'+name+'.'
            values={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
            assert values,(name,list(state)[:8])
            module=getattr(self.model,name);module.load_state_dict(values,strict=True)
            assert all(torch.equal(v,module.state_dict()[k]) for k,v in values.items())
            loaded[name]=len(values)
        self.model.cuda().eval();self.torch=torch
        self.info=dict(model='EMOCA_v2_lr_mse_20',checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),
            config=str(configs[0]),config_sha256=sha(configs[0]),encoder_tensors=loaded,
            dimensions=dict(nparam=nparam,nexp=nexp),global_pose_from_expression=c['exp_deca_global_pose'],
            jaw_pose_from_expression=c['exp_deca_jaw_pose'],source_methods_sha256=sha(official),
            source_encoder_sha256=sha(EMOCA/'gdl/models/DecaEncoder.py'),
            full_renderer_loaded=False,full_encoder_state_strict=True,
            precision='FP32; no autocast or TF32',initialization='Skipped ImageNet initialization, all tensors overwritten by checkpoint')

    def encode(self,images):
        with self.torch.inference_mode():
            c,_=self.model.decompose_code(self.model._encode_flame(images))
            out=self.torch.cat([c[3],c[2]],dim=1)
        assert out.shape==(len(images),56) and self.torch.isfinite(out).all()
        return out

    def independent(self,images):
        t=self.torch;c=self.model.config
        with t.inference_mode():
            raw=self.model.E_flame(images);e=self.model.E_expression(images)
            start=c.n_shape+c.n_tex+c.n_exp
            p=raw[:,start:start+6]
            if c.exp_deca_global_pose and c.exp_deca_jaw_pose:p=e[:,50:56]
            elif c.exp_deca_global_pose:p=t.cat([e[:,50:53],p[:,3:6]],1)
            elif c.exp_deca_jaw_pose:p=t.cat([p[:,:3],e[:,50:53]],1)
            return t.cat([p,e[:,:50]],1)


class Detector:
    def __init__(self):
        import torch
        import face_alignment
        from skimage.io import imread
        # Only an API-name compatibility alias; detector network unchanged.
        if not hasattr(face_alignment.LandmarksType,'_2D'):
            setattr(face_alignment.LandmarksType,'_2D',face_alignment.LandmarksType.TWO_D)
        sys.path.insert(0,str(EMOCA))
        from gdl.utils.FaceDetector import FAN
        from gdl.datasets.ImageDatasetHelpers import bbox2point,bbpoint_warp
        namespace=dict(np=np,Path=Path,imread=imread,bbox2point=bbox2point,bbpoint_warp=bbpoint_warp)
        cls=methods_from_source(EMOCA/'gdl/datasets/FaceDataModuleBase.py','FaceDataModuleBase',
            ['_detect_faces_in_image'],object,namespace)
        self.source=cls();self.source.face_detector=FAN(device='cuda',threshold=.9)
        self.source._instantiate_detector=lambda:None
        self.source.scale=1.25;self.source.image_size=224
        self.source.bb_center_shift_x=self.source.bb_center_shift_y=0.
        self.torch=torch

    def crop(self,frame):
        results=self.source._detect_faces_in_image(frame)
        self.torch.backends.cudnn.benchmark=False
        if not len(results[0]):return None,0
        # Official video processing stores these uint8 crops losslessly as PNG.
        image=results[0][0].transpose(2,0,1).astype(np.float32)/255.
        return image,len(results[0])


def frames(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            yield cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    finally:cap.release()


def run(smoke=False):
    from vico_more_common import gpu_guard
    import torch,cv2
    import fcntl
    WORK.mkdir(exist_ok=True)
    lock=(WORK/'extract.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    device=gpu_guard(16);cv2.setNumThreads(1)
    protocol=read(WORK/'protocol.json')
    for p,h in protocol['source_files'].items():assert sha(p)==h,p
    assert read(WORK/'LICENSE_CONFIRMATION.json')['research_use']
    encoder=Encoder();detector=Detector()
    save(WORK/'ENCODER.json',encoder.info);save(WORK/'DEVICE.json',device)
    specification=dict(model_sha256=encoder.info['checkpoint_sha256'],source_protocol_sha256=sha(WORK/'protocol.json'),
        crop='Official FaceDataModuleBase._detect_faces_in_image; FAN2D, threshold.9, scale1.25,224, bicubic, uint8 lossless PNG-equivalent, first face',
        video_decode='OpenCV RGB full timeline; reference decoder parity checked on smoke',
        missing_face='NaN at original frame index; never interpolation; scorer discloses shared valid mask',
        batch=16,coefficients='pose6+exp50, unnormalized',implementation_sha256=sha(__file__))
    dest=WORK/'EXTRACTION_PROTOCOL.json'
    if dest.exists():assert read(dest)==specification,'Extraction changed; use a new version'
    else:save(dest,specification)
    tasks=protocol['tasks'];records=[];begin=time.time()
    for task in tasks[:1] if smoke else tasks:
        folder=WORK/'features'/task['role'];folder.mkdir(parents=True,exist_ok=True)
        out=folder/(task['id']+'.npz');receipt=out.with_suffix('.json')
        if not smoke and receipt.exists():
            r=read(receipt)
            assert sha(out)==r['sha256'] and r['video_sha256']==task['video_sha256']
            assert r['extraction_protocol_sha256']==sha(dest)
            records.append(r);continue
        assert sha(task['video'])==task['video_sha256']
        matrix=np.full((task['expected_frames'],56),np.nan,dtype=np.float32)
        face_counts=[];batch=[];indices=[];seen=0
        def flush():
            if not batch:return
            a=torch.from_numpy(np.stack(batch)).cuda()
            values=encoder.encode(a)
            matrix[indices]=values.cpu().numpy();batch.clear();indices.clear()
        for i,frame in enumerate(frames(task['video'])):
            assert i<task['expected_frames'];seen=i+1
            crop,count=detector.crop(frame);face_counts.append(count)
            if crop is not None:
                if smoke:
                    import subprocess
                    reference=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-threads','1',
                        '-i',task['video'],'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
                    reference=np.frombuffer(reference,np.uint8).reshape(frame.shape)
                    decoder_error=int(np.abs(frame.astype(np.int16)-reference.astype(np.int16)).max())
                    assert decoder_error==0,decoder_error
                    a=torch.from_numpy(crop[None]).cuda()
                    source=encoder.encode(a);independent=encoder.independent(a)
                    error=float((source-independent).abs().max());assert error<1e-7
                    grouped=encoder.encode(a.repeat(16,1,1,1))
                    batching_error=float((grouped-source).abs().max());assert batching_error<1e-4
                    save(WORK/'ENCODER_PARITY.json',dict(passed=True,max_abs=error,
                        source='Unmodified official encoder and decompose methods vs independent coefficient selection',
                        frame_id=task['id'],checkpoint_sha256=encoder.info['checkpoint_sha256'],
                        strict_encoder_tensors=encoder.info['encoder_tensors'],renderer_parity_claim=False))
                    save(WORK/'MEDIA_PARITY.json',dict(passed=True,decoder_max_abs=decoder_error,
                        encoder_batch1_vs16_max_abs=batching_error,model_sha256=encoder.info['checkpoint_sha256'],
                        source_video_sha256=task['video_sha256']))
                    image=(crop.transpose(1,2,0)*255).round().astype(np.uint8)
                    cv2.imwrite(str(WORK/'smoke_crop.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                    flush();return
                batch.append(crop);indices.append(i)
                if len(batch)==16:flush()
        flush();assert seen==task['expected_frames'],(task,seen)
        valid=np.isfinite(matrix).all(axis=1)
        assert valid.any(),task
        np.savez_compressed(out,motion=matrix,frame_indices=np.arange(seen),valid=valid,face_counts=np.asarray(face_counts))
        r=dict(id=task['id'],role=task['role'],path=str(out),sha256=sha(out),video_sha256=task['video_sha256'],
            frames=seen,valid_frames=int(valid.sum()),missing_frames=np.where(~valid)[0].tolist(),
            multi_face_frames=int(np.sum(np.asarray(face_counts)>1)),extraction_protocol_sha256=sha(dest))
        save(receipt,r);records.append(r)
        save(WORK/'status.json',dict(stage='extracting',completed=len(records),total=len(tasks),
            latest=task['role']+'/'+task['id'],elapsed_s=time.time()-begin))
        print('extracted',len(records),'/',len(tasks),task['role'],task['id'],'missing',len(r['missing_frames']),flush=True)
    save(WORK/'EXTRACTION_COMPLETE.json',dict(success=True,records=records,tasks=len(tasks),pairs=43,
        at=time.time(),protocol_sha256=sha(dest)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.smoke)
