"""Reuse official GPAvatar first-frame-only tracking; cache once for four L2L models."""
import argparse,time,importlib.util
from runtime import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();device()
    import torch,torchvision,numpy as np
    import p2_l2l_gpavatar as gp
    gp.save=lambda p,v:save(WORK/'model_load/gpavatar'/Path(p).name,v)
    engine=gp.tracker()
    spec=importlib.util.spec_from_file_location('gpavatar_official_tools',gp.VENDOR/'core/data/tools.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    count=0
    for s in selected():
        resident_checkpoint()
        d=WORK/'gpavatar_references'/s['dataset']/s['id'];d.mkdir(parents=True,exist_ok=True)
        with (d/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (d/'complete.json').exists() or (d/'TRACK_UNAVAILABLE.json').exists():continue
            start=time.time();image=torchvision.io.read_image(s['reference'],mode=torchvision.io.ImageReadMode.RGB).float()/255.
            image=torchvision.transforms.functional.resize(image,[512,512],antialias=True);image=torch.nn.functional.pad(image,[64]*4);image=torchvision.transforms.functional.resize(image,[512,512],antialias=True)
            captured={};original=engine.track_base
            def capture(frame):
                value=original(frame)
                if value is not None:captured['pose']=np.asarray(value['emoca_pose']).copy()
                return value
            engine.track_base=capture
            try:images,results=engine.track_images([image],vis_path=None)
            finally:engine.track_base=original
            if '0' not in results or 'pose' not in captured:
                save(d/'TRACK_UNAVAILABLE.json',dict(reason='Official tracker produced no valid first-frame identity',id=s['id'],source_image_sha256=s['reference_sha256']))
                continue
            r=results['0'];transform=torch.as_tensor(r['transform_matrix'])[None].float().cuda()
            warped=module.perspective_input(images[0][None].cuda(),transform,dict(focal_length=12.,principal_point=[0.,0.]),fill=0.)[0]
            avatar=dict(feature_image=warped[None].cpu(),feature_shape=torch.as_tensor(r['mica_shape']).float().cpu(),frame_trans=transform.cpu(),reference_pose=torch.as_tensor(captured['pose']).float().cpu())
            assert all(torch.isfinite(x).all() for x in avatar.values())
            torch.save(avatar,d/'avatar.pt');torch.save(r,d/'tracked.pt')
            save(d/'complete.json',dict(source_frames=1,future_listener_frames_used=0,source_image_sha256=s['reference_sha256'],files={str(d/n):sha(d/n) for n in ['avatar.pt','tracked.pt']},seconds=time.time()-start,code_sha256=sha(__file__)))
            count+=1;print('GPAvatar tracked',s['key'],round(time.time()-start,2),flush=True)
            if a.limit and count>=a.limit:return
if __name__=='__main__':main()
