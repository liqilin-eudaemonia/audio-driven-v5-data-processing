"""Official SPECTRE coefficient encoder with temporal context and strict weights."""
import argparse
import importlib
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from p2_avatarforcing_table4_generate import ROOT,WORK,SPECTRE,guard,read,save,sha


def temporal_indices(length,batch):
    if length<1 or batch<1:raise ValueError('Empty temporal input')
    for start in range(0,length,batch):
        end=min(length,start+batch)
        yield start,end,np.clip(np.arange(start-2,end+2),0,length-1)


class Encoder:
    def __init__(self,device='cuda'):
        import torch,torchvision
        from vico_literature_emoca import methods_from_source
        package=types.ModuleType('af_spectre_models');package.__path__=[str(SPECTRE/'src/models')]
        sys.modules['af_spectre_models']=package
        resnet=importlib.import_module('af_spectre_models.resnet')
        encoders=importlib.import_module('af_spectre_models.encoders')
        resnet.load_ResNet50Model=lambda:resnet.ResNet(resnet.Bottleneck,[3,4,6,3])
        # Only initializers are skipped; every encoder state tensor is overwritten strictly.
        with patch.object(torch.hub,'load',lambda *args,**kwargs:torchvision.models.mobilenet_v2(weights=None)):
            flame=encoders.ResnetEncoder(236)
            expression=encoders.PerceptualEncoder(50,SimpleNamespace(backbone='mobilenetv2'))
        state=torch.load(WORK/'assets/spectre_model.tar',map_location='cpu',weights_only=False)
        loaded={}
        for name,model in [('E_flame',flame),('E_expression',expression)]:
            model.load_state_dict(state[name],strict=True)
            assert all(torch.equal(model.state_dict()[k],v) for k,v in state[name].items())
            loaded[name]=len(state[name])
        cls=methods_from_source(SPECTRE/'src/spectre.py','SPECTRE',['decompose_code','encode'],torch.nn.Module,dict(torch=torch))
        self.model=cls();self.model.E_flame=flame;self.model.E_expression=expression
        self.model.param_dict=dict(shape=100,tex=50,exp=50,pose=6,cam=3,light=27)
        self.model.to(device).eval().requires_grad_(False);self.device=device;self.torch=torch
        self.info=dict(checkpoint_sha256=sha(WORK/'assets/spectre_model.tar'),strict_encoder_tensors=loaded,
            source={str(p):sha(p) for p in [SPECTRE/'src/spectre.py',SPECTRE/'src/models/encoders.py',SPECTRE/'src/models/resnet.py',SPECTRE/'demo.py']},
            precision='float32; no autocast/TF32',residual='official demo adds initial DECA exp and jaw once',
            mobilenet='torchvision MobileNetV2 architecture; all state keys/shapes loaded strictly',
            flame_mesh_renderer_loaded=False,temporal_context=2)

    def encode(self,images):
        t=self.torch
        with t.inference_mode():
            code,deca_exp,deca_jaw=self.model.encode(images)
            exp=code['exp']+deca_exp
            pose=code['pose'].clone();pose[...,3:]+=deca_jaw
            values=t.cat([pose,exp],dim=-1)
        assert values.shape==(len(images),56) and t.isfinite(values).all()
        return values

    def chunks(self,images,batch=32):
        result=[]
        for start,end,indices in temporal_indices(len(images),batch):
            a=self.torch.from_numpy(images[indices].copy()).to(self.device)
            result.append(self.encode(a)[2:-2].cpu().numpy())
        return np.concatenate(result)

    def parity(self,images):
        t=self.torch
        indices=np.clip(np.arange(-2,len(images)+2),0,len(images)-1)
        a=t.from_numpy(images[indices].copy()).to(self.device)
        source=self.encode(a)[2:-2].cpu().numpy()
        chunked=self.chunks(images,batch=3)
        error=float(np.abs(source-chunked).max());assert error<1e-4,error
        with t.inference_mode():
            raw=self.model.E_flame(a);delta,jaw=self.model.E_expression(a)
            independent=t.cat([raw[:,200:203],raw[:,203:206]+jaw,raw[:,150:200]+delta],dim=-1)[2:-2]
        selection_error=float(np.abs(source-independent.cpu().numpy()).max());assert selection_error==0
        return dict(passed=True,full_vs_context_chunk_max_abs=error,source_vs_independent_coefficient_max_abs=selection_error,
            frames=len(images),device=self.device,strict_encoder_state=True)


def decode25(path,expected_frames):
    import cv2
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
    while True:
        okay,frame=cap.read()
        if not okay:break
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames or fps<=0:raise ValueError('Video decode failed')
    if abs(len(frames)/fps-expected_frames/25)>.081:raise ValueError('Full timeline differs')
    indices=np.floor(np.arange(expected_frames)*fps/25+.5).astype(int).clip(0,len(frames)-1)
    values=np.stack([cv2.resize(frames[i],(512,512),interpolation=cv2.INTER_CUBIC) if frames[i].shape[:2]!=(512,512) else frames[i] for i in indices])
    return values,dict(source_fps=fps,source_frames=len(frames),source_indices=indices.tolist(),target_fps=25)


def source_crop_function():
    import ast,copy
    from skimage.transform import estimate_transform
    tree=ast.parse((SPECTRE/'demo.py').read_text())
    fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='crop_face'))
    namespace=dict(np=np,estimate_transform=estimate_transform)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SPECTRE/'demo.py')+':crop_face','exec'),namespace)
    return namespace['crop_face']


def crop_frames(frames,detector):
    from skimage.transform import warp
    spec=importlib.util.spec_from_file_location('af_spectre_data_utils',SPECTRE/'datasets/data_utils.py')
    utils=importlib.util.module_from_spec(spec);spec.loader.exec_module(utils)
    landmarks=[];counts=[]
    for frame in frames:
        points=detector.get_landmarks_from_image(frame)
        counts.append(len(points) if points is not None else 0)
        landmarks.append(points[0] if points is not None and len(points) else None)
    filled=utils.landmarks_interpolate(landmarks)
    if filled is None:raise ValueError('No face in full sequence')
    crop=source_crop_function()
    images=np.stack([warp(frame,crop(frame,points,scale=1.6).inverse,output_shape=(224,224)).transpose(2,0,1).astype(np.float32) for frame,points in zip(frames,filled)])
    return images,dict(missing_landmark_frames=[i for i,c in enumerate(counts) if c==0],multi_face_frames=sum(c>1 for c in counts),
        crop='official SPECTRE demo crop_face scale1.6; 224 warp',
        landmark_backend='installed face_alignment SFD/2DFAN; first face; not vendor custom tracking fork',
        failed_landmarks='official SPECTRE landmarks_interpolate; leading/trailing nearest valid',
        num_frames=len(frames))


def extract(smoke=False,cpu=False):
    import cv2,torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if not cpu:save(WORK/'SPECTRE_DEVICE.json',guard(.08,32768))
    model=Encoder('cpu' if cpu else 'cuda')
    save(WORK/'SPECTRE_ENCODER.json',model.info)
    p=read(WORK/'protocol.json')
    if smoke:
        frames,_=decode25(p['samples'][0]['gt_video'],max(8,math_frames(p['samples'][0])))
        images=np.stack([cv2.resize(frame,(224,224)).transpose(2,0,1).astype(np.float32)/255 for frame in frames[:8]])
        save(WORK/'SPECTRE_PARITY.json',model.parity(images));return
    if cpu:raise ValueError('Full extraction requires explicit GPU guard')
    if not (WORK/'GENERATION_COMPLETE.json').exists():raise ValueError('Generate all43 outputs first')
    import face_alignment
    enum=getattr(face_alignment.LandmarksType,'TWO_D',None) or face_alignment.LandmarksType._2D
    detector=face_alignment.FaceAlignment(enum,device='cuda',flip_input=False)
    torch.backends.cudnn.benchmark=False
    tasks=[]
    for s in p['samples']:
        for role,path in [('speaker',s['speaker_video']),('gt',s['gt_video']),('avatarforcing',str(WORK/'generated'/s['id']/'generated.mp4'))]:
            tasks.append(dict(id=s['id'],role=role,video=path,frames=read(WORK/'generated'/s['id']/'complete.json')['frames']))
    contract=dict(encoder=model.info,implementation_sha256=sha(__file__),generation_protocol_sha256=sha(WORK/'protocol.json'),
        temporal='2frame edge replication; exact context around each32 valid frames',fps=25,batch=32,
        tasks=[dict(t,video_sha256=sha(t['video'])) for t in tasks])
    dest=WORK/'SPECTRE_PROTOCOL.json'
    if dest.exists():assert read(dest)==contract
    else:save(dest,contract)
    receipts=[]
    for task in contract['tasks']:
        folder=WORK/'features'/task['role'];folder.mkdir(parents=True,exist_ok=True)
        output=folder/(task['id']+'.npz');receipt=output.with_suffix('.json')
        if receipt.exists():
            r=read(receipt);assert sha(output)==r['sha256'] and r['protocol_sha256']==sha(dest)
            receipts.append(r);continue
        frames,grid=decode25(task['video'],task['frames']);images,audit=crop_frames(frames,detector)
        if not (WORK/'SPECTRE_GPU_PARITY.json').exists():save(WORK/'SPECTRE_GPU_PARITY.json',model.parity(images[:8]))
        values=model.chunks(images)
        assert values.shape==(task['frames'],56)
        np.savez_compressed(output,motion=values,frame_indices=np.arange(len(values)),fps=25)
        r=dict(id=task['id'],role=task['role'],path=str(output),sha256=sha(output),protocol_sha256=sha(dest),
            video_sha256=task['video_sha256'],frames=len(values),grid=grid,face_audit=audit)
        save(receipt,r);receipts.append(r)
        save(WORK/'status.json',dict(stage='spectre_extraction',completed=len(receipts),total=129,last=task['role']+'/'+task['id']))
        print('SPECTRE',len(receipts),'/129',task['role'],task['id'],flush=True)
        del frames,images,values
    save(WORK/'EXTRACTION_COMPLETE.json',dict(passed=True,records=receipts,streams=129,protocol_sha256=sha(dest)))


def math_frames(s):
    import math
    return math.ceil(s['duration']*25)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['smoke','extract']);parser.add_argument('--cpu',action='store_true')
    args=parser.parse_args();extract(args.action=='smoke',args.cpu)
