"""PIRender EMA with predicted semantics and a fixed listener first frame."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from vico_more_common import ROOT, WORK, read, save, sha, gpu_guard, test_samples

VENDOR = WORK / 'vendor/rlhg/PIRender'
WEIGHT = WORK / 'assets/renderer/face/epoch_00190_iteration_000400000_checkpoint.pt'


def model_for():
    import yaml
    sys.path.insert(0, str(VENDOR))
    from generators.face_model import FaceGenerator
    config = yaml.safe_load((VENDOR/'config/face_demo.yaml').read_text())
    model = FaceGenerator(**config['gen']['param'])
    checkpoint = torch.load(WEIGHT, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['net_G_ema'], strict=True)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in checkpoint['net_G_ema'].items())
    save(WORK/'renderer/WEIGHT_AUDIT.json', dict(strict=True, tensor_exact=True, state='net_G_ema',
        keys=len(checkpoint['net_G_ema']), checkpoint_sha256=sha(WEIGHT),
        epoch=checkpoint['current_epoch'], iteration=checkpoint['current_iteration'],
        config_sha256=sha(VENDOR/'config/face_demo.yaml')))
    return model.eval()


def semantics(dynamic, centers):
    # Driver angle-exp-trans-crop -> renderer exp-angle-trans-crop.
    native = dynamic[:,np.r_[3:67,0:3,67:73]]
    idx=np.clip(np.asarray(centers)[:,None]+np.arange(-13,14)[None],0,len(native)-1)
    return torch.from_numpy(native[idx].transpose(0,2,1).copy()).float()


def semantic_parity(dynamic):
    # Invoke original dataset methods without importing unrelated lmdb loaders.
    import ast
    source=VENDOR/'data/vox_video_dataset.py'
    parent=VENDOR/'data/vox_dataset.py'
    methods=[]
    for path,name in [(source,'transform_semantic'),(parent,'obtain_seq_index')]:
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
        methods.append(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
    env=dict(np=np,torch=torch)
    for node in methods:
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    from types import SimpleNamespace,MethodType
    obj=SimpleNamespace(semantic_radius=13,cross_id=False,norm_crop_param=False)
    obj.obtain_seq_index=MethodType(env['obtain_seq_index'],obj)
    fake=np.zeros((len(dynamic),260),np.float32)
    fake[:,80:144]=dynamic[:,3:67];fake[:,224:227]=dynamic[:,:3]
    fake[:,254:257]=dynamic[:,67:70];fake[:,257:260]=dynamic[:,70:73]
    indices=[0,min(13,len(dynamic)-1),len(dynamic)-1]
    expected=torch.stack([env['transform_semantic'](obj,fake,i,None) for i in indices])
    actual=semantics(dynamic,indices)
    assert torch.equal(expected,actual)
    return dict(passed=True, exact=True, centers=indices, order='exp64,angle3,trans3,crop3',window=27)


def first_frame(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        ok,frame=cap.read();assert ok
    finally:cap.release()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    assert frame.shape==(256,256,3)
    return torch.from_numpy(frame.transpose(2,0,1).copy()).float().div(255).sub(.5).div(.5)[None]


def smoke():
    """Rendering-only sanity check on a training clip, not a generated baseline."""
    gpu_guard(12)
    from vico_more_features import OUT as FEATURES
    task=next(t for t in read(FEATURES/'protocol.json')['tasks'] if t['split']=='train')
    dyn=np.load(FEATURES/'train'/task['id']/'listener.npz')['dynamic']
    parity=semantic_parity(dyn)
    model=model_for().cuda();image=first_frame(task['inputs']['listener']['path']).cuda()
    with torch.inference_mode():
        x=semantics(dyn,[0,1,2,3]).cuda()
        batch=model(image.expand(4,-1,-1,-1),x)['fake_image']
        single=torch.cat([model(image,a[None])['fake_image'] for a in x])
    diff=float((single-batch).abs().max())
    assert diff<1e-3 and torch.isfinite(batch).all()
    from PIL import Image
    pixels=((batch[0].clamp(-1,1)+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    Image.fromarray(pixels).save(WORK/'renderer/training_reconstruction_sanity.png')
    save(WORK/'renderer/SMOKE.json',dict(passed=True,semantic=parity,batch_max_abs=diff,
        training_clip=task['id'],diagnostic='GT driver renderer sanity only; not a baseline prediction',
        finite=True,pixel_mean=float(pixels.mean()),pixel_std=float(pixels.std())))


def render(method):
    gpu_guard(18)
    out=WORK/method
    gen=read(out/'GENERATION_COMPLETE.json');assert gen['success']
    model=model_for().cuda()
    samples={s['id']:s for s in test_samples()}
    records=[]
    checked=False
    for rec in gen['records']:
        uid=rec['id'];dest=out/'scoring'/method/(uid+'.listener.mp4');dest.parent.mkdir(parents=True,exist_ok=True)
        receipt=dest.with_suffix('.json')
        if receipt.exists():
            previous=read(receipt);assert sha(dest)==previous['sha256'];records.append(previous);continue
        t=time.time()
        assert sha(rec['path'])==rec['sha256']
        dyn=np.load(rec['path'])['dynamic'];n=len(dyn)
        source=samples[uid]['streams']['gt']
        assert sha(source['path'])==source['sha256']
        image=first_frame(source['path']).cuda()
        if not checked:
            parity=semantic_parity(dyn)
            with torch.inference_mode():
                x=semantics(dyn,[0,1,2,3]).cuda()
                batch=model(image.expand(4,-1,-1,-1),x)['fake_image']
                single=torch.cat([model(image,s[None])['fake_image'] for s in x])
            difference=float((batch-single).abs().max())
            assert difference<1e-3,difference
            save(out/'RENDER_PARITY.json',dict(semantic=parity,batch_float_max_abs=difference,
                source_frames=1,batch=4,checkpoint='PIRender official pretrained EMA'))
            checked=True
        temp=dest.with_name(dest.stem+'.tmp.mp4')
        command=['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','256x256',
            '-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','medium',
            '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            with torch.inference_mode():
                for start in range(0,n,4):
                    centers=np.arange(start,min(start+4,n))
                    x=semantics(dyn,centers).cuda()
                    result=model(image.expand(len(centers),-1,-1,-1),x)['fake_image']
                    pixels=((result.clamp(-1,1)+1)/2*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
                    process.stdin.write(pixels.tobytes())
            process.stdin.close();assert process.wait()==0
        finally:
            if process.poll() is None:process.kill()
        # Full decode verifies exact native frame grid, rather than container headers alone.
        import cv2
        cap=cv2.VideoCapture(str(temp));count=0
        while True:
            ok,f=cap.read()
            if not ok:break
            assert f.shape==(256,256,3);count+=1
        cap.release();assert count==n,(count,n)
        temp.replace(dest)
        r=dict(id=uid,path=str(dest),sha256=sha(dest),frames=n,fps=30,width=256,height=256,
            complete_decoded=True,source_frames=1,future_listener_identity=False,future_listener_crop=False,
            predicted_crop=True,seconds=time.time()-t)
        save(receipt,r);records.append(r)
        print(method,'render',len(records),uid,round(time.time()-t,2),flush=True)
    # Reuse exact sealed official metric scripts with an explicit method directory.
    previous=read(ROOT/'experiments/p2_vico256_combined8000_20261106/protocol.json')
    save(out/'protocol.json',dict(method=method,datasets=previous['datasets'],
        training_protocol_sha256=sha(out/'TRAIN_PROTOCOL.json'),
        generation_complete_sha256=sha(out/'GENERATION_COMPLETE.json'),
        renderer_audit_sha256=sha(WORK/'renderer/WEIGHT_AUDIT.json')))
    import shutil
    shutil.copyfile(ROOT/'experiments/p2_vico256_combined8000_20261106/INPUT_PREFLIGHT.json',out/'INPUT_PREFLIGHT.json')
    save(out/'RENDER_COMPLETE.json',dict(success=True,outputs=43,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','render','smoke']);p.add_argument('--method',choices=['rlhg','listenformer'])
    a=p.parse_args()
    if a.stage=='audit':model_for()
    elif a.stage=='smoke':smoke()
    else:render(a.method)
