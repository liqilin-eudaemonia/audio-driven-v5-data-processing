"""New full-user-visual Avatar Forcing run; original source-only run is read-only."""
import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'compare/deployment')]
WORK=ROOT/'experiments/p2_avatarforcing_table4_recheck_20261110'
REPO=ROOT/'compare/AvatarForcing'
OLD=ROOT/'experiments/p2_vico43_unified_20261107'
SPECTRE=ROOT/'compare/deployment/vendor/spectre'
UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def guard(fraction,minimum_mib):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='7':raise ValueError('Only node11 physicalGPU7 is authorized')
    row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if row[0].strip()!=UUID:raise ValueError('Unexpected GPU')
    free,total=int(row[1]),int(row[2]);required=max(minimum_mib,math.ceil(total*fraction)+16384)
    if free<required:raise RuntimeError(f'Insufficient capacity: {free}MiB free, {required}MiB required; no other job stopped')
    import torch
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(fraction,0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return dict(uuid=UUID,free_mib=free,required_mib=required,fraction=fraction,pid=os.getpid(),host=os.uname().nodename)


def extract_methods(path,cls_name,methods,namespace):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==cls_name)
    body=[copy.deepcopy(n) for n in node.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    assert len(body)==len(methods)
    cls=ast.ClassDef(name=cls_name,bases=[],keywords=[],body=body,decorator_list=[])
    namespace=dict(namespace);exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path)+':source-methods','exec'),namespace)
    return namespace[cls_name]


def prepare():
    WORK.mkdir(parents=True,exist_ok=True)
    old=read(OLD/'avatarforcing/protocol.json')
    motion=read(ROOT/'experiments/p2_vico_motion8000_20261106/protocol.json')
    byid={s['id']:s for s in motion['samples']};samples=[]
    for row in old['samples']:
        s=dict(row,speaker_video=byid[row['id']]['streams']['speaker']['path'],
            gt_video=byid[row['id']]['streams']['gt']['path'])
        s['source_sha256']={str(Path(p).resolve()):sha(p) for p in [s['speaker_video'],s['gt_video'],s['speaker'],s['image']]}
        samples.append(s)
    sources=[Path(__file__),ROOT/'compare/deployment/vico_avatarforcing.py',REPO/'inference.py',REPO/'preprocess_user_video.py',
        REPO/'configs/inference.yaml',*sorted((REPO/'models').rglob('*.py'))]
    p=dict(schema='avatarforcing-table4-recheck-v1',samples=samples,
        source_files={str(p):sha(p) for p in sources},assets=old['assets'],
        paper_sha256=sha(ROOT/'writing/43_Avatar_Forcing_CVPR2026.pdf'),
        source_run_sha256=sha(OLD/'COMPLETE.json'),official_test43=True,
        seed=137,nfe=10,avatar_cfg=2.,user_cfg=1.,fps=25,
        user_visual='real speaker frames; official preprocessing and unmodified encode_user_motion',
        avatar_audio='zeros; explicit listening-only inference assumption, paperTable4 does not specify it',
        source_videos='official croppedViCo256/30, full speaker waveform; convert25fps explicitly',
        reference='official single-image DataProcessor.preprocess_face from common first-frame source',
        sampling='one seed137 sample, not DIM best-of10',
        spectre_source='https://github.com/filby89/spectre',
        target=dict(FD=dict(exp=16.64,pose=.05),rPCC=dict(exp=.01,pose=.01),SID=dict(exp=3.12,pose=3.00),Var=dict(exp=2.80,pose=.03)),
        exact_checkpoint_table4_binding='not explicitly disclosed; official public final checkpoints',
        old_results_unchanged=True)
    dest=WORK/'protocol.json'
    if dest.exists():assert read(dest)==p,'Protocol changed; use a new version'
    else:save(dest,p)
    return p


def verify():
    p=read(WORK/'protocol.json')
    for path,h in {**p['source_files'],**p['assets']}.items():assert sha(path)==h,path
    for sample in p['samples']:
        for path,h in sample['source_sha256'].items():assert sha(path)==h,path
    return p


def preprocess(sample,detector):
    import cv2,numpy as np
    folder=WORK/'inputs'/sample['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        for path,h in read(folder/'complete.json')['files'].items():assert sha(path)==h,path
        return folder
    raw=folder/'raw_speaker25';raw.mkdir(exist_ok=True)
    cropped=folder/'speaker25';cropped.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-i',sample['speaker_video'],
        '-vf','fps=25','-qscale:v','0','-threads','1',str(raw/'%05d.jpg')],check=True)
    from types import SimpleNamespace
    cls=extract_methods(REPO/'preprocess_user_video.py','UserVideoProcessor',['img_loader','dectect_and_crop_face_from_frames'],dict(cv2=cv2,np=np,Path=Path,os=os))
    proc=cls();proc.fa=detector;proc.opt=SimpleNamespace(input_size=512)
    proc.dectect_and_crop_face_from_frames(str(raw),str(cropped),pad_ratio=1.0)
    cls=extract_methods(REPO/'inference.py','DataProcessor',['default_img_loader','preprocess_face'],dict(cv2=cv2,np=np))
    ref=cls();ref.fa=detector;ref.opt=SimpleNamespace(input_size=512)
    image=ref.preprocess_face(sample['image'],pad_ratio=1.0)
    cv2.imwrite(str(folder/'reference.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    files=[folder/'reference.png',*sorted(cropped.glob('*.jpg'))]
    save(folder/'complete.json',dict(files={str(p):sha(p) for p in files},frame_count=len(files)-1,
        speaker_video=sample['speaker_video'],reference_source=sample['image'],official_methods=True))
    return folder


def generate(smoke=False):
    import cv2,numpy as np,soundfile as sf,torch,face_alignment,types
    from transformers import Wav2Vec2FeatureExtractor
    import vico_avatarforcing as old
    p=verify();device=guard(.15,49152);cv2.setNumThreads(1)
    save(WORK/'GENERATION_DEVICE.json',device)
    old.WORK=WORK
    model=old.build_model()
    model.encode_user_motion=types.MethodType(type(model).encode_user_motion,model)
    assert model.encode_user_motion.__func__ is type(model).encode_user_motion
    enum=getattr(face_alignment.LandmarksType,'TWO_D',None) or face_alignment.LandmarksType._2D
    detector=face_alignment.FaceAlignment(enum,flip_input=False,device='cuda')
    pp=Wav2Vec2FeatureExtractor.from_pretrained(str(REPO/'pretrained_dir/wav2vec2-base-960h'),local_files_only=True)
    samples=p['samples']
    if smoke:samples=[min(samples,key=lambda s:s['duration']),max(samples,key=lambda s:s['duration'])]
    begin=time.time()
    for s in samples:
        output=WORK/'generated'/s['id'];output.mkdir(parents=True,exist_ok=True)
        if (output/'complete.json').exists():
            r=read(output/'complete.json');assert sha(output/'generated.mp4')==r['sha256'];continue
        started=time.time();folder=preprocess(s,detector)
        wave,sr=sf.read(s['speaker'],dtype='float32');assert sr==16000
        frames=math.ceil(len(wave)/640)
        paths=sorted((folder/'speaker25').glob('*.jpg'))
        if abs(len(paths)-frames)>1:raise ValueError('User video/audio length mismatch')
        paths=(paths+[paths[-1]])[:frames]
        user=[]
        for path in paths:
            rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            user.append(torch.from_numpy(((rgb.astype(np.float32)-127.5)/127.5).transpose(2,0,1).copy()).unsqueeze(0))
        ref=cv2.cvtColor(cv2.imread(str(folder/'reference.png')),cv2.COLOR_BGR2RGB)
        data=dict(avatar_ref=torch.from_numpy(((ref.astype(np.float32)-127.5)/127.5).transpose(2,0,1).copy()).unsqueeze(0),
            avatar_a=pp(np.zeros_like(wave),sampling_rate=sr,return_tensors='pt').input_values,
            user_a=pp(wave,sampling_rate=sr,return_tensors='pt').input_values,user_frame=user)
        torch.manual_seed(p['seed']);np.random.seed(p['seed'])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=model.inference(data,a_cfg_scale=p['avatar_cfg'],u_cfg_scale=p['user_cfg'],nfe=p['nfe'],seed=p['seed'],use_kv_cache=True)['d_hat']
        assert prediction.shape==(frames,3,512,512) and torch.isfinite(prediction).all()
        prediction=prediction.float().clamp(-1,1).cpu();lo,hi=prediction.min(),prediction.max();assert hi>lo
        rgb=((prediction-lo)/(hi-lo)*255).byte().permute(0,2,3,1).numpy()
        old.export_rgb(output/'generated.mp4',[rgb],512,25,18)
        save(output/'complete.json',dict(id=s['id'],sha256=sha(output/'generated.mp4'),frames=frames,fps=25,
            protocol_sha256=sha(WORK/'protocol.json'),seconds=time.time()-started,real_user_visual=True,
            avatar_audio_zero=True,listener_future_conditioned=False,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))
        done=len(list((WORK/'generated').glob('*/complete.json')))
        save(WORK/'status.json',dict(stage='generation',completed=done,total=43,last=s['id'],elapsed_s=time.time()-begin))
        print('generated',done,'/43',s['id'],'seconds',round(time.time()-started,2),flush=True)
        del prediction,rgb,data,user;torch.cuda.empty_cache()
    if smoke:save(WORK/'GENERATION_SMOKE.json',dict(passed=True,ids=[s['id'] for s in samples],retained_for_formal=True))
    if len(list((WORK/'generated').glob('*/complete.json')))==43:
        save(WORK/'GENERATION_COMPLETE.json',dict(passed=True,outputs=43,protocol_sha256=sha(WORK/'protocol.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','smoke','generate'])
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:generate(args.action=='smoke')
