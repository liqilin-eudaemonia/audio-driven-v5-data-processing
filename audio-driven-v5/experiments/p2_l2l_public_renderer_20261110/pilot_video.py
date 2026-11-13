"""One complete shortest clip, retained for the formal172-output run."""
from pathlib import Path
import sys
import time
import fcntl
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import p2_l2l_gpavatar as gp
import torch,numpy as np,cv2

stage_lock=(gp.WORK/'gpu_stage.lock').open('a')
fcntl.flock(stage_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=gp.verify()
assert gp.read(gp.WORK/'SMOKE_COMPLETE.json')['passed']
assert gp.read(gp.WORK/'VISUAL_ACCEPTED.json')['passed']
device=gp.guard();model=gp.renderer()
sample=min(p['samples'],key=lambda s:s['frames']);uid=sample['id'];name='conan'
motionfile=sample['predictions'][name]['path']
assert gp.sha(motionfile)==sample['predictions'][name]['sha256']
reference=gp.WORK/'references'/uid
for f,h in gp.read(reference/'complete.json')['files'].items():assert gp.sha(f)==h
avatar=torch.load(reference/'avatar.pt',map_location='cpu',weights_only=True)
record=torch.load(reference/'tracked.pt',map_location='cpu',weights_only=False)
avatar['reference_expression']=torch.as_tensor(record['emoca_expression']).float()
motion=torch.from_numpy(np.load(motionfile)['motion']).float().cuda()
assert motion.shape==(sample['frames'],56)
torch.manual_seed(137)
with torch.inference_mode():model.build_avatar(avatar)
raws=[];pixels=[];small=[];errors=[];begin=time.time()
indices={0,len(motion)//2,len(motion)-1}
with torch.inference_mode():
    for i in range(len(motion)):
        raw=gp.render_motion(model,avatar,motion[i:i+1]).clamp(0,1)
        rgb=gp.restore_canvas(raw)
        assert torch.isfinite(rgb).all()
        native=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
        raws.append((raw[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8));pixels.append(native)
        quantized=torch.from_numpy(native.copy()).permute(2,0,1)[None].float()/255.
        low=torch.nn.functional.interpolate(quantized,size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        small.append((low[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
        if i in indices:
            expected=cv2.imread(str(gp.WORK/'smoke'/name/uid/f'frame_{i:05d}.png'))[:,:,::-1]
            errors.append(int(np.abs(expected.astype(np.int16)-native.astype(np.int16)).max()))
assert max(errors)<=2,errors
dest=gp.WORK/'generated'/name/uid
score=gp.WORK/'scoring'/('l2l_gpavatar_'+name)/(uid+'.listener.mp4')
gp.output_video(dest/'generated.mp4',pixels)
gp.output_video(dest/'raw_padded.mp4',raws)
gp.output_video(score,small,256)
for f,size in [(dest/'generated.mp4',512),(dest/'raw_padded.mp4',512),(score,256)]:
    cap=cv2.VideoCapture(str(f));seen=0
    while True:
        ok,image=cap.read()
        if not ok:break
        assert image.shape==(size,size,3);seen+=1
    cap.release();assert seen==len(motion)
receipt=dict(id=uid,model=name,label='L2L public conan + GPAvatar',source_frames=1,
    future_listener_input=False,motion_sha256=sample['predictions'][name]['sha256'],
    protocol_sha256=gp.sha(gp.WORK/'protocol.json'),frames=len(motion),fps=30,score_path=str(score),
    native_size=512,score_size=256,selected_indices=None,
    camera_parity=gp.camera_parity(model,avatar,motion),seconds=time.time()-begin,
    pilot_retained_for_formal=True,pilot_source_sha256=gp.sha(__file__),
    files={str(f):gp.sha(f) for f in [dest/'generated.mp4',dest/'raw_padded.mp4',score]})
gp.save(dest/'complete.json',receipt)
gp.save(gp.WORK/'PILOT_VIDEO_COMPLETE.json',dict(passed=True,id=uid,frames=len(motion),fps=30,
    smoke_vs_complete_clip_pixels_max_abs=max(errors),retained_for_formal=True,device=device,
    seconds=time.time()-begin,receipt=str(dest/'complete.json'),peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
print('Complete real clip passed',uid,len(motion),'frames; max smoke difference',max(errors),flush=True)
