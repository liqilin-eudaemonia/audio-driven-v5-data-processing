"""Public L2L motion -> public GPAvatar renderer, with first-frame-only identity."""
import argparse
import copy
import fcntl
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'experiments/p2_l2l_public_renderer_20261110'
VENDOR = WORK / 'vendor/GPAvatar'
L2L = ROOT / 'experiments/p2_compare_vico43_20261106/learning2listen'
UUID = 'GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
NAMES = ['conan','fallon','stephen','trevor']
sys.path[:0] = [str(WORK/'runtime_extra'),str(WORK/'runtime_py'),str(VENDOR),str(ROOT/'compare/deployment')]


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024**2):result.update(data)
    return result.hexdigest()


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def guard():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7', 'node11 physicalGPU7 only'
    row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert row[0].strip()==UUID, row
    assert int(row[1])>=32768, 'Insufficient free capacity; no other job stopped'
    import torch
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.10,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(137)
    return dict(uuid=UUID,free_mib=int(row[1]),allocator_fraction=.10,pid=os.getpid(),host=os.uname().nodename)


def prepare():
    import urllib.request
    import tarfile
    WORK.mkdir(exist_ok=True)
    downloads=read(WORK/'DOWNLOADS.json')
    for record in downloads['receipts']:
        assert record['passed'] and sha(record['path'])==record['sha256']
    assert read(ROOT/'experiments/p2_vico_literature_protocol_20261108/LICENSE_CONFIRMATION.json')['research_use']
    resources=WORK/'assets/extracted'
    if not (resources/'EXTRACTED.json').exists():
        resources.mkdir(exist_ok=True)
        with tarfile.open(WORK/'assets/resources.tar') as archive:
            members=[m for m in archive.getmembers() if any(m.name.startswith('resources/'+n+'/') for n in ['main_params','track_params','matting','emoca','mica_base'])]
            assert members
            archive.extractall(resources,members=members,filter='data')
        save(resources/'EXTRACTED.json',dict(archive_sha256=sha(WORK/'assets/resources.tar'),members=len(members)))
    track=VENDOR/'core/libs/lightning_track/engines'
    links={VENDOR/'core/libs/FLAME/assets':'main_params',
           VENDOR/'inference_tools/libs/flame_lite/assets':'main_params',
           track/'FLAME/assets':'track_params',track/'human_matting/assets':'matting',
           track/'emoca/assets':'emoca',track/'mica/assets':'mica_base'}
    for dest,name in links.items():
        source=resources/'resources'/name
        assert source.is_dir(), source
        dest.mkdir(parents=True,exist_ok=True)
        for f in source.iterdir():
            p=dest/f.name
            if p.exists():assert p.resolve()==f.resolve(), p
            else:p.symlink_to(f,target_is_directory=f.is_dir())
    target=VENDOR/'inference_tools/checkpoints/one_model.ckpt'
    if not target.exists():target.symlink_to(WORK/'assets/one_model.ckpt')
    config=WORK/'assets/mask2former_config.json'
    if not config.exists():
        url='https://huggingface.co/facebook/mask2former-swin-tiny-coco-instance/resolve/main/config.json'
        config.write_bytes(urllib.request.urlopen(url,timeout=30).read())
    original=read(L2L/'protocol.json');samples=[]
    for s in original['samples']:
        assert sha(s['image'])==s['image_sha256']
        preds={}
        for name in NAMES:
            f=L2L/'generated'/name/(s['id']+'.npz');r=read(f.with_suffix('.json'))
            assert r['source_only'] and not r['listener_future_input'] and sha(f)==r['feature_sha256']
            preds[name]=dict(path=str(f),sha256=sha(f))
        samples.append(dict(id=s['id'],frames=s['frames'],fps=30,image=s['image'],image_sha256=s['image_sha256'],predictions=preds))
    assert len(samples)==43
    sources=[Path(__file__),*sorted((VENDOR/'inference_tools').rglob('*.py')),
        *sorted((VENDOR/'core/libs/lightning_track').rglob('*.py')),VENDOR/'core/data/tools.py',VENDOR/'core/utils/visualize.py']
    sources=[p for p in sources if '__pycache__' not in p.parts]
    assets={str(p):sha(p) for p in sorted(resources.rglob('*')) if p.is_file()}
    assets.update({str(WORK/'assets/one_model.ckpt'):sha(WORK/'assets/one_model.ckpt'),str(config):sha(config)})
    protocol=dict(schema='l2l-public-gpavatar-v1',methods=NAMES,samples=samples,pairs=43,outputs=172,
        gpavatar_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip(),
        source_files={str(p):sha(p) for p in sources},assets=assets,
        l2l_protocol_sha256=sha(L2L/'protocol.json'),seed=137,
        appearance='Only common listener first frame; official GPAvatar tracker and perspective_input',
        motion='Existing official public L2L exp50 + head axis-angle3 + jaw axis-angle3, raw, at30fps',
        pose='Canonical FLAME geometry with predicted exp/jaw. Global pose converted to equivalent camera motion about FLAME root. First-frame tracker calibrates camera; no global pose applied twice.',
        unpredicted='Identity shape and camera translation calibration fixed from first frame; neck/eyes zero as official GPAvatar',
        style_noise='Official supported randomize_noise=False; frozen noise buffers for temporal consistency',
        reference_padding='Fixed64px black border around512px first frame, then resize512; original tightly cropped first reference failed MICA. All frames/models share this preprocessing.',
        output='Raw padded512/30 retained. Inverse static padding by bicubic grid_sample scale0.8 align_cornersFalse gives common512 canvas, then bicubic-antialias256/30 scoring copy; no time resampling; no generated audio',
        future_listener_input=False,gt_based_person_selection=False,
        label='L2L public person-specific weights + GPAvatar public renderer; not original L2L photorealistic pipeline',
        research_license_confirmation=str(ROOT/'experiments/p2_vico_literature_protocol_20261108/LICENSE_CONFIRMATION.json'),
        resource=dict(node='node11',physical_gpu=7,minimum_free_mib=32768,allocator_fraction=.10))
    dest=WORK/'protocol.json'
    if dest.exists():assert read(dest)==protocol,'Frozen protocol changed; create a new revision'
    else:save(dest,protocol)
    save(WORK/'status.json',dict(stage='prepared',pairs=43,outputs=172))
    print('Prepared43 identities and172 immutable L2L motion streams',flush=True)


def verify():
    p=read(WORK/'protocol.json')
    for f,h in {**p['source_files'],**p['assets']}.items():assert sha(f)==h,f
    assert sha(L2L/'protocol.json')==p['l2l_protocol_sha256']
    return p


def numpy_compat():
    import numpy as np
    import inspect
    for k,v in dict(bool=bool,int=int,float=float,complex=complex,object=object,str=str,unicode=str).items():
        if k not in np.__dict__:setattr(np,k,v)
    if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec


def tracker():
    numpy_compat()
    import torch
    from transformers import Mask2FormerConfig,Mask2FormerForUniversalSegmentation
    original=Mask2FormerForUniversalSegmentation.from_pretrained
    # The official StyleMatte checkpoint strictly replaces the entire pixel
    # decoder. Avoid downloading an irrelevant, discarded initialization.
    cfg=Mask2FormerConfig.from_dict(read(WORK/'assets/mask2former_config.json'))
    Mask2FormerForUniversalSegmentation.from_pretrained=classmethod(lambda cls,*a,**kw:cls(cfg))
    try:
        # Bypass unrelated dataset/training package initializers. All tracking
        # module bodies and model forwards below remain the official code.
        import types
        for package,relative in [('core','core'),('core.libs','core/libs'),('core.utils','core/utils')]:
            if package not in sys.modules:
                module=types.ModuleType(package);module.__path__=[str(VENDOR/relative)]
                sys.modules[package]=module
        import onnxruntime as ort
        original_session=ort.InferenceSession
        class BoundedSession(original_session):
            def __init__(self,path,sess_options=None,**kwargs):
                if sess_options is None:
                    sess_options=ort.SessionOptions()
                    sess_options.intra_op_num_threads=2;sess_options.inter_op_num_threads=1
                super().__init__(path,sess_options=sess_options,**kwargs)
        ort.InferenceSession=BoundedSession
        from core.libs.lightning_track import TrackEngine
        engine=TrackEngine(focal_length=12.,device='cuda')
    finally:Mask2FormerForUniversalSegmentation.from_pretrained=original
    return engine


def track_references(smoke=False):
    import torch,torchvision,numpy as np
    p=verify();device=guard();save(WORK/'TRACK_DEVICE.json',device)
    engine=tracker()
    selected=p['samples']
    if smoke:selected=[min(selected,key=lambda s:s['frames']),max(selected,key=lambda s:s['frames'])]
    # Import the official helper as a standalone module; no dataset/trainer initialization.
    spec=importlib.util.spec_from_file_location('gpavatar_official_data_tools',VENDOR/'core/data/tools.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for s in selected:
        folder=WORK/'references'/s['id'];receipt=folder/'complete.json'
        if receipt.exists():
            for f,h in read(receipt)['files'].items():assert sha(f)==h
            continue
        started=time.time();assert sha(s['image'])==s['image_sha256']
        image=torchvision.io.read_image(s['image'],mode=torchvision.io.ImageReadMode.RGB).float()/255.
        image=torchvision.transforms.functional.resize(image,[512,512],antialias=True)
        image=torch.nn.functional.pad(image,[64]*4)
        image=torchvision.transforms.functional.resize(image,[512,512],antialias=True)
        captured={};native=engine.track_base
        def capture(frame):
            result=native(frame)
            if result is not None:captured['reference_pose']=np.asarray(result['emoca_pose']).copy()
            return result
        engine.track_base=capture
        folder.mkdir(parents=True,exist_ok=True)
        try:images,results=engine.track_images([image],vis_path=None)
        finally:engine.track_base=native
        assert '0' in results and 'reference_pose' in captured,s['id']
        record=results['0'];transform=torch.as_tensor(record['transform_matrix'])[None].float().cuda()
        warped=module.perspective_input(images[0][None].cuda(),transform,dict(focal_length=12.,principal_point=[0.,0.]),fill=0.)[0]
        avatar=dict(feature_image=warped[None].cpu(),feature_shape=torch.as_tensor(record['mica_shape']).float().cpu(),
            frame_trans=transform.cpu(),reference_pose=torch.as_tensor(captured['reference_pose']).float().cpu())
        assert avatar['feature_image'].shape==(1,3,512,512)
        assert avatar['feature_shape'].shape==(100,) and avatar['reference_pose'].shape==(6,)
        assert all(torch.isfinite(a).all() for a in avatar.values())
        torch.save(avatar,folder/'avatar.pt');torch.save(record,folder/'tracked.pt')
        torchvision.utils.save_image(image,folder/'tracking_input.png')
        torchvision.utils.save_image(warped,folder/'feature_plane.png')
        torchvision.utils.save_image(images[0],folder/'matted_reference.png')
        save(receipt,dict(id=s['id'],source_image_sha256=s['image_sha256'],source_frames=1,
            future_listener_frames_used=0,seconds=time.time()-started,
            fixed_padding=64,tracking_size=512,common_canvas_inverse_scale=.8,
            files={str(folder/n):sha(folder/n) for n in ['avatar.pt','tracked.pt','feature_plane.png','matted_reference.png','tracking_input.png']}))
        done=len(list((WORK/'references').glob('*/complete.json')))
        save(WORK/'status.json',dict(stage='tracking_first_frames',completed=done,total=43,last=s['id']))
        print('Tracked',done,'/43',s['id'],round(time.time()-started,2),flush=True)
    if smoke:save(WORK/'TRACK_SMOKE_COMPLETE.json',dict(passed=True,identities=2,source_frames_each=1,ids=[s['id'] for s in selected]))
    else:save(WORK/'TRACK_COMPLETE.json',dict(passed=True,identities=43,source_frames_each=1))


def renderer():
    numpy_compat()
    import torch
    from inference_tools import gpavatar_r2g
    model=gpavatar_r2g().cuda().eval()
    state=torch.load(WORK/'assets/one_model.ckpt',map_location='cpu',weights_only=False)['state_dict']
    actual=model.state_dict();matched={};missing=[]
    for name,param in model.named_parameters():
        if name in state:
            assert torch.equal(param.detach().cpu(),state[name]),name
            matched[name]=list(param.shape)
        elif name not in ['flame.eye_pose','flame.neck_pose']:missing.append(name)
    assert not missing,missing
    core_keys=[k for k in state if not k.startswith(('percep_loss.','flame_model.','flame.'))]
    assert all(k in actual and torch.equal(actual[k].cpu(),state[k]) for k in core_keys)
    save(WORK/'WEIGHTS_LOADED.json',dict(passed=True,all_learned_parameters_exact=True,
        matched_parameters=len(matched),missing=missing,checkpoint_sha256=sha(WORK/'assets/one_model.ckpt'),
        fixed_geometry_from_official_release_assets=True))
    model.style_tplane.forward=functools.partial(model.style_tplane.forward,randomize_noise=False)
    model.up_renderer.forward=functools.partial(model.up_renderer.forward,randomize_noise=False)
    return model


def calibrated_camera(model, avatar, motion):
    """Move head-global FLAME rotation into the view, keeping query space canonical.

    PyTorch3D uses row-vector x @ R + T. FLAME global rotation gives
    x' = x @ Q.T + (J - J @ Q.T), where J is the shape/expression root.
    This conversion has an exact independently checked projection identity.
    """
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix
    f=model.flame;device=motion.device
    shape=avatar['feature_shape'][None].to(device).expand(len(motion),-1)
    ref=axis_angle_to_matrix(avatar['reference_pose'][None,:3].to(device))
    track=avatar['frame_trans'].to(device)
    root_basis=torch.einsum('v,vcl->cl',f.J_regressor[0],f.shapedirs)*f.scale
    root0=torch.einsum('v,vc->c',f.J_regressor[0],f.v_template)*f.scale
    def roots(exp):return torch.cat([shape,exp],1)@root_basis.T+root0
    # Static source calibration uses its tracked source expression, retained separately.
    reference_exp=avatar.get('reference_expression')
    if reference_exp is None:raise ValueError('Missing first-frame reference expression')
    refroot=(torch.cat([avatar['feature_shape'],reference_exp]).to(device)@root_basis.T+root0)[None]
    refshift=refroot-torch.bmm(refroot[:,None],ref.transpose(1,2))[:,0]
    base_rotation=ref@track[:,:,:3]
    base_translation=track[:,:,3]-torch.bmm(refshift[:,None],base_rotation)[:,0]
    q=axis_angle_to_matrix(motion[:,50:53])
    root=roots(motion[:,:50]);shift=root-torch.bmm(root[:,None],q.transpose(1,2))[:,0]
    rotation=q.transpose(1,2)@base_rotation
    translation=torch.bmm(shift[:,None],base_rotation.expand(len(motion),-1,-1))[:,0]+base_translation
    return torch.cat([rotation,translation[:,:,None]],dim=-1)


def render_motion(model,avatar,motion):
    import torch
    camera=calibrated_camera(model,avatar,motion)
    pose=torch.cat([torch.zeros_like(motion[:,50:53]),motion[:,53:56]],dim=-1)
    return model(motion[:,:50],pose,camera)


def restore_canvas(rgb):
    import torch
    matrix=rgb.new_tensor([[[.8,0.,0.],[0.,.8,0.]]]).expand(len(rgb),-1,-1)
    grid=torch.nn.functional.affine_grid(matrix,rgb.shape,align_corners=False)
    return torch.nn.functional.grid_sample(rgb,grid,mode='bicubic',padding_mode='border',align_corners=False).clamp(0,1)


def camera_parity(model,avatar,motion):
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix
    a=copy.deepcopy(avatar);ref=a['reference_pose'].to(motion.device)
    source=torch.cat([a['reference_expression'].to(motion.device),ref])[None]
    camera=calibrated_camera(model,a,source)
    error=float((camera-a['frame_trans'].to(motion.device)).abs().max())
    assert error<1e-4,error
    # Compare posed-world projection against canonical-world equivalent view.
    f=model.flame;shape=a['feature_shape'][None].to(motion.device)
    refq=axis_angle_to_matrix(ref[None,:3]);track=a['frame_trans'].to(motion.device)
    neutralpose=torch.cat([torch.zeros_like(ref[None,:3]),ref[None,3:]],1)
    neutral=f(shape,source[:,:50],neutralpose)
    posed=f(shape,source[:,:50],ref[None])
    shift=(posed-neutral@refq.transpose(1,2)).mean(1)
    rigid_error=float((posed-neutral@refq.transpose(1,2)-shift[:,None]).abs().max())
    assert rigid_error<1e-4,rigid_error
    base_rotation=refq@track[:,:,:3]
    base_translation=track[:,:,3]-(shift[:,None]@base_rotation)[:,0]
    errors=[]
    for m in motion[:3]:
        raw=m[None];jaw=torch.cat([torch.zeros_like(raw[:,50:53]),raw[:,53:56]],1)
        vertices=f(shape,raw[:,:50],jaw)
        posed=f(shape,raw[:,:50],raw[:,50:56])
        cam=calibrated_camera(model,a,raw)
        expected=posed@base_rotation+base_translation[:,None]
        actual=vertices@cam[:,:,:3]+cam[:,:,3][:,None]
        errors.append(float((actual-expected).abs().max()))
    assert max(errors)<1e-4,errors
    return dict(passed=True,reference_camera_max_abs=error,flame_rigid_max_abs=rigid_error,
                independently_posed_world_vs_view_max_abs=max(errors),no_double_global_rotation=True)


def output_video(path,frames,size=512):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.stem+'.partial.mp4')
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{size}x{size}','-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','1',
        '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE)
    try:
        for pixels in frames:proc.stdin.write(pixels.tobytes())
        proc.stdin.close()
        assert proc.wait()==0
    finally:
        if proc.poll() is None:proc.kill()
    temporary.replace(path)


def generate(smoke=False):
    import torch,torchvision,numpy as np,cv2
    p=verify();device=guard();save(WORK/'RENDER_DEVICE.json',device)
    model=renderer();samples=p['samples']
    if smoke:samples=[min(samples,key=lambda s:s['frames']),max(samples,key=lambda s:s['frames'])]
    begin=time.time();frame_seconds=[];proofs=[]
    for sample in samples:
        uid=sample['id'];folder=WORK/'references'/uid
        avatar=torch.load(folder/'avatar.pt',map_location='cpu',weights_only=True)
        tracked=torch.load(folder/'tracked.pt',map_location='cpu',weights_only=False)
        avatar['reference_expression']=torch.as_tensor(tracked['emoca_expression']).float()
        for f,h in read(folder/'complete.json')['files'].items():assert sha(f)==h
        torch.manual_seed(137)
        with torch.inference_mode():model.build_avatar(avatar)
        for name in NAMES:
            motionfile=sample['predictions'][name]['path']
            assert sha(motionfile)==sample['predictions'][name]['sha256']
            motion=np.load(motionfile)['motion'].astype(np.float32)
            assert motion.shape==(sample['frames'],56) and np.isfinite(motion).all()
            dest=WORK/('smoke' if smoke else 'generated')/name/uid
            receipt=dest/'complete.json'
            if receipt.exists():
                r=read(receipt)
                for f,h in r['files'].items():assert sha(f)==h
                proofs.append(r)
                continue
            selected=np.arange(len(motion))
            if smoke:selected=np.unique([0,len(motion)//2,len(motion)-1])
            tensor=torch.from_numpy(motion).cuda()
            proof=camera_parity(model,avatar,tensor)
            if smoke and name=='conan' and not (WORK/'RENDER_CONTROL_PARITY.json').exists():
                differences={}
                with torch.inference_mode():
                    base=render_motion(model,avatar,tensor[:1]).clamp(0,1)
                    for group,column,delta in [('expression',0,.5),('head_rotation',51,.15),('jaw_rotation',53,.15)]:
                        changed=tensor[:1].clone();changed[:,column]+=delta
                        result=render_motion(model,avatar,changed).clamp(0,1)
                        differences[group]=float((result-base).abs().mean())
                assert all(v>1e-5 for v in differences.values()),differences
                save(WORK/'RENDER_CONTROL_PARITY.json',dict(passed=True,source_frames=1,
                    mean_rgb_change=differences,camera=proof,motion_reaches_actual_rgb=True))
            started=time.time();pixels=[];raw_pixels=[]
            # One frame per query bounds memory independently of clip length.
            with torch.inference_mode():
                for i,index in enumerate(selected):
                    tick=time.time();rgb=render_motion(model,avatar,tensor[index:index+1]).clamp(0,1)
                    assert rgb.shape==(1,3,512,512) and torch.isfinite(rgb).all()
                    raw_pixels.append((rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
                    rgb=restore_canvas(rgb)
                    pixels.append((rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
                    frame_seconds.append(time.time()-tick)
                    if i%30==0:save(WORK/'status.json',dict(stage='rendering',model=name,id=uid,frames_done=i+1,clip_frames=len(selected),completed_outputs=len(list((WORK/'generated').glob('*/*/complete.json'))),total_outputs=172))
            output_video(dest/'generated.mp4',pixels)
            output_video(dest/'raw_padded.mp4',raw_pixels)
            scoring=WORK/('smoke_scoring' if smoke else 'scoring')/('l2l_gpavatar_'+name)/(uid+'.listener.mp4')
            small=[]
            for x in pixels:
                a=torch.from_numpy(x.copy()).permute(2,0,1)[None].float()/255.
                a=torch.nn.functional.interpolate(a,size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
                small.append((a[0].permute(1,2,0).numpy()*255).round().astype(np.uint8))
            output_video(scoring,small,256)
            cap=cv2.VideoCapture(str(dest/'generated.mp4'));count=0
            while True:
                ok,image=cap.read()
                if not ok:break
                assert image.shape==(512,512,3);count+=1
            cap.release();assert count==len(selected)
            if smoke:
                for index,x in zip(selected,pixels):
                    cv2.imwrite(str(dest/(f'frame_{index:05d}.png')),cv2.cvtColor(x,cv2.COLOR_RGB2BGR))
            r=dict(id=uid,model=name,label='L2L public '+name+' + GPAvatar',source_frames=1,
                future_listener_input=False,motion_sha256=sample['predictions'][name]['sha256'],
                protocol_sha256=sha(WORK/'protocol.json'),frames=count,fps=30,score_path=str(scoring),
                native_size=512,score_size=256,selected_indices=selected.tolist() if smoke else None,
                camera_parity=proof,seconds=time.time()-started,files={str(f):sha(f) for f in [dest/'generated.mp4',dest/'raw_padded.mp4',scoring]})
            save(receipt,r);proofs.append(r)
            print('Rendered',name,uid,count,round(r['seconds'],2),flush=True)
    if smoke:
        assert len(proofs)==8
        save(WORK/'SMOKE_COMPLETE.json',dict(passed=True,records=proofs,total_seconds=time.time()-begin,
            mean_frame_seconds=float(np.mean(frame_seconds)),median_frame_seconds=float(np.median(frame_seconds)),
            estimated172_render_hours=float(np.mean(frame_seconds))*sum(s['frames'] for s in p['samples'])*4/3600))
    else:
        receipts=list((WORK/'generated').glob('*/*/complete.json'));assert len(receipts)==172
        save(WORK/'RENDER_COMPLETE.json',dict(passed=True,outputs=172,pairs=43,models=NAMES,
            records=[read(f) for f in receipts],protocol_sha256=sha(WORK/'protocol.json')))
        save(WORK/'status.json',dict(stage='render_complete',outputs=172,pairs=43))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','track_smoke','track','smoke','generate'])
    args=parser.parse_args()
    WORK.mkdir(exist_ok=True)
    stage_lock=(WORK/'gpu_stage.lock').open('a')
    fcntl.flock(stage_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage=='prepare':prepare()
    elif args.stage in ['track','track_smoke']:track_references(args.stage=='track_smoke')
    else:generate(args.stage=='smoke')
