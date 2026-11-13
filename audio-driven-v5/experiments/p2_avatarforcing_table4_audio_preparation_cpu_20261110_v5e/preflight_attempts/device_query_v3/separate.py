"""Official IIANet target speech extraction on CPU; same ViCo source and weights."""
import argparse
import ast
from collections import deque
import contextlib
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import time
import types

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
PARENT=WORK.parent/'p2_avatarforcing_table4_reference_mean_20261110_v3'
VENDOR=WORK/'vendor/IIANet'
sys.path[:0]=[str(WORK/'python_deps'),str(VENDOR),str(ROOT/'scripts')]
from p2_avatarforcing_table4_generate import read,save,sha,guard


class VideoReader:
    """The cv2 read sequence used by mmcv.VideoReader, without unrelated deps."""
    def __init__(self,path):
        import cv2
        self.path=path
        cap=cv2.VideoCapture(path)
        self.width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));self.height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution=(self.width,self.height);self.fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    def __iter__(self):
        import cv2
        cap=cv2.VideoCapture(self.path)
        try:
            while True:
                okay,frame=cap.read()
                if not okay:break
                yield frame
        finally:cap.release()


def source_functions():
    import cv2,numpy as np,torch,face_alignment
    from PIL import Image,ImageDraw
    from facenet_pytorch import MTCNN
    from skimage import transform as tf
    path=VENDOR/'inference.py'
    detector_calls=[]
    class ObservedMTCNN(MTCNN):
        def __init__(self,*args,**kwargs):
            kwargs['device']='cpu'
            super().__init__(*args,**kwargs)
        def detect(self,*args,**kwargs):
            result=super().detect(*args,**kwargs)
            boxes,probabilities=result
            detector_calls.append(dict(boxes=None if boxes is None else boxes.tolist(),probabilities=None if probabilities is None else probabilities.tolist()))
            return result
    def fixed_face_alignment(*args,**kwargs):
        kwargs['device']='cpu'
        tracker=face_alignment.FaceAlignment(*args,**kwargs)
        # FaceAlignment changes this process-global switch in its constructor.
        # Fix deterministic FP32 policy before any MTCNN/FAN inference.
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        return tracker
    face_api=types.SimpleNamespace(FaceAlignment=fixed_face_alignment,LandmarksType=face_alignment.LandmarksType)
    tree=ast.parse(path.read_text())
    ns=dict(os=os,cv2=cv2,np=np,torch=torch,face_alignment=face_api,MTCNN=ObservedMTCNN,
            Image=Image,ImageDraw=ImageDraw,tf=tf,deque=deque,mmcv=types.SimpleNamespace(VideoReader=VideoReader))
    module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
    exec(compile(module,str(path)+':official-functions','exec'),ns)
    original_save=ns['save2npz']
    def save_compatible(filename,data=None):
        assert data is not None
        # Convert before opening the ZIP. An exception inside np.savez can
        # leave an old ZipFile destructor overwriting a subsequent retry.
        try:values=np.asarray(data)
        except ValueError:values=np.asarray(data,dtype=object)
        return original_save(filename,values)
    ns['save2npz']=save_compatible
    ns['detector_calls']=detector_calls
    return ns


def prepare():
    path=WORK/'PROTOCOL.json'
    generation=read(PARENT/'NATIVE_GENERATION_PROTOCOL.json')
    files={str(Path(__file__)):sha(__file__)}
    for row in read(WORK/'IIANET_SOURCE_MANIFEST.json')['records']:
        p=ROOT/row['path'];assert sha(p)==row['sha256'];files[str(p)]=row['sha256']
    weights=[WORK/'released_checkpoints/vox2/best_model.pth',WORK/'released_checkpoints/vox2/conf.yml',WORK/'issue6_bundle/IIANetv2/weights/lrw_resnet18_mstcn_adamw_s3.pth.tar']
    for p in weights:files[str(p)]=sha(p)
    for pattern in ['facenet_pytorch/data/*.pt','facenet_pytorch/models/mtcnn.py','facenet_pytorch/models/utils/detect_face.py']:
        for p in (WORK/'python_deps').glob(pattern):files[str(p)]=sha(p)
    if path.exists():
        p=read(path);assert p['source_and_weights']==files
        return p
    assert read(WORK/'IIANET_MODEL_ASSEMBLY_AUDIT.json')['passed']
    samples=[]
    for sample in generation['samples']:
        uid=sample['id'];record=dict(id=uid,mix=sample['speaker'],mix_sha256=sha(sample['speaker']),samples=round(sample['duration']*16000),roles=[])
        for role in ['speaker','listener']:
            v=PARENT/'inputs'/uid/role/'cropped25.mkv';record['roles'].append(dict(role=role,video=str(v),video_sha256=sha(v)))
        samples.append(record)
    p=dict(schema='official-IIANet-Vox2-all-CPU-separator-v5',created_unix=time.time(),source_and_weights=files,samples=samples,
        basis='AF paper5.1 explicit visual-grounded speech separation and official README IIANet recommendation; frozen input currently mixed conversation',
        IIANet_commit=read(WORK/'IIANET_SOURCE_MANIFEST.json')['commit'],audio_checkpoint='Official Vox2 best_model; identical bytes in author issue6 bundle',
        visual_checkpoint='Exact lrw_resnet18_mstcn_adamw_s3 from official author issue6 bundle; every instantiated video state checked',
        video_input='Existing per-person original-view512/25 fixed crop; explicit single target person per input',
        mouth='Unmodified official detectface/crop_mouth functions; MTCNN every8frames,1.5 head scale,224FMP4 face video; FAN4;12frame smoothing;96gray ROI;88center crop and LRW normalization',
        tracking_device='CPU for MTCNN/SFD/FAN4,8threads; same model weights and functions; audio/video separation networks also run on CPU',
        I_O_adaptations=['cv2 sequential VideoReader replaces mmcv wrapper only','Explicit object dtype if NumPy ragged landmark save fails; source interpolation unchanged','Absolute paths; no hardcoded GPU9 or source main executed'],
        wave='Frozen16k mono mix, no gain normalization, full clip; official full-sequence model forward, no window resampling or seed search',
        output='Raw float estimate.npy plus official torchaudio.save default WAV (FLOAT in the verified native backend); saved versus raw metrics and all sample counts disclosed',
        precision='CPU FP32,8torch threads,MKLDNN deterministicTrue; CUBLAS_WORKSPACE_CONFIG=:4096:8 retained unused; torch.use_deterministic_algorithms(True); FP32 eval/inference_mode; immediately after FaceAlignment construction and before separation benchmarkFalse deterministicTrue TF32False; source library cannot leave autotune enabled',
        own_audio_for_future_AF_generation='Remain zero in current speaker-only comparison; listener stem measured separately, never silently supplied as GT condition',
        canary_ids=['ClNUxPsUByg_000210_000221.0','Ew2z_sYABE0_000063_000083.0'],no_Table4_score_used_for_selection=True,
        source_private_AF_separator_checkpoint_confirmed=False,ClearVoice_chain_not_specified_by_paper=True)
    save(path,p);return p


def metrics(mix,estimate):
    import numpy as np
    x=mix.astype(np.float64);y=estimate.astype(np.float64)
    xr=float(np.sqrt(np.mean(x*x)));yr=float(np.sqrt(np.mean(y*y)))
    x-=x.mean();y-=y.mean();den=float(np.linalg.norm(x)*np.linalg.norm(y))
    corr=float(x@y/den) if den>1e-15 else None
    alpha=float(x@y/(y@y)) if float(y@y)>1e-15 else None
    return dict(mix_rms=xr,estimate_rms=yr,gain_db=20*float(np.log10(max(yr,1e-15)/max(xr,1e-15))),
        centered_correlation=corr,scale_invariant_residual_rms_ratio=float(np.linalg.norm(x-alpha*y)/np.linalg.norm(x)) if alpha is not None and float(np.linalg.norm(x))>0 else None,
        raw_abs_peak=float(np.max(np.abs(estimate))),samples_over_pcm16_range=int(np.count_nonzero(np.abs(estimate)>1.)),samples=len(mix))


def cpu_guard():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(8)
    torch.backends.mkldnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return dict(device='cpu',GPU_used=False,threads=8,MKLDNN_deterministic=True,pid=os.getpid(),host=os.uname().nodename)


def run(action):
    import cv2,numpy as np,torch,torchaudio,yaml,soundfile as sf
    from look2hear.models import IIANet
    from look2hear.videomodels import ResNetVideoModel
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    p=prepare();device=cpu_guard();torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1);torch.set_num_threads(8)
    torch.manual_seed(20);np.random.seed(20)
    conf=yaml.safe_load((WORK/'released_checkpoints/vox2/conf.yml').read_text())
    audio=IIANet.from_pretrain(str(WORK/'released_checkpoints/vox2/best_model.pth'),sample_rate=16000,**conf['audionet']['audionet_config']).cpu().eval()
    video=ResNetVideoModel(pretrain=str(WORK/'issue6_bundle/IIANetv2/weights/lrw_resnet18_mstcn_adamw_s3.pth.tar')).cpu().eval()
    spec=importlib.util.spec_from_file_location('official_iianet_transforms',VENDOR/'look2hear/datas/transform.py');transform=importlib.util.module_from_spec(spec);spec.loader.exec_module(transform)
    funcs=source_functions();samples=p['samples'] if action=='all' else [s for s in p['samples'] if s['id'] in p['canary_ids']]
    save(WORK/'RUNTIME.json',dict(pid=os.getpid(),device=device,torch=torch.__version__,global_deterministic=torch.are_deterministic_algorithms_enabled(),tracking_device='cpu',tracking_threads=8,cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),opencv=cv2.__version__,face_alignment=__import__('face_alignment').__version__,protocol_sha256=sha(WORK/'PROTOCOL.json'),started_unix=time.time()))
    runtime_sha=sha(WORK/'RUNTIME.json');history=WORK/'runtime_history';history.mkdir(exist_ok=True)
    (history/(runtime_sha+'.json')).write_bytes((WORK/'RUNTIME.json').read_bytes())
    rows=[]
    with (WORK/'separation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sample in samples:
            assert sha(sample['mix'])==sample['mix_sha256']
            mix,sr=sf.read(sample['mix'],dtype='float32');assert sr==16000 and mix.shape==(sample['samples'],)
            per_role=[]
            for role in sample['roles']:
                folder=WORK/'separated'/sample['id']/role['role'];folder.mkdir(parents=True,exist_ok=True);receipt=folder/'complete.json'
                if receipt.exists():
                    r=read(receipt);assert r['protocol_sha256']==sha(WORK/'PROTOCOL.json')
                    for f,h in r['files'].items():assert sha(f)==h
                    per_role.append(r);continue
                assert sha(role['video'])==role['video_sha256'];started=time.time()
                oldcwd=Path.cwd()
                torch.set_num_threads(8)
                funcs['detector_calls'].clear()
                try:
                    os.chdir(VENDOR)
                    with (folder/'preprocess.log').open('a') as log,contextlib.redirect_stdout(log):
                        index=funcs['detectface'](role['video'],str(folder),8,1.5,1)
                        funcs['crop_mouth'](str(folder/'faces'),str(folder/'landmark'),index,str(folder/'mouthroi'),convert_gray=True,testset_only=False)
                finally:
                    os.chdir(oldcwd);torch.set_num_threads(8)
                save(folder/'MTCNN_AUDIT.json',dict(calls=funcs['detector_calls'],selection='Unmodified source first target; MTCNN default select_largest=True',source_tracking_interval=8))
                roi_path=folder/'mouthroi/speaker1.npz';assert roi_path.exists(),'No complete mouth track; do not skip clip'
                roi=np.load(roi_path)['data'];assert roi.ndim==3 and roi.shape[1:]==(96,96)
                expected=round(sample['samples']/16000*25);assert len(roi)==expected
                normalized=transform.get_preprocessing_pipelines()['val'](roi)
                inputs=torch.from_numpy(normalized[None,None]).float().cpu();wave=torch.from_numpy(mix[None,None]).cpu()
                torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                with torch.inference_mode():
                    embedding=video(inputs);value=audio(wave,embedding)
                    assert value.shape==(1,1,len(mix)) and torch.isfinite(value).all()
                    if action=='canary':
                        repeat=audio(wave,embedding);error=float((value-repeat).abs().max());assert error<1e-6,error
                    else:error=None
                estimate=value[0,0].cpu().numpy().copy();np.save(folder/'estimate.npy',estimate)
                torchaudio.save(str(folder/'target.wav'),value[0].cpu(),16000)
                saved,rs=sf.read(folder/'target.wav',dtype='float32');assert rs==16000 and saved.shape==mix.shape and np.isfinite(saved).all()
                info=sf.info(folder/'target.wav')
                files={str(f):sha(f) for f in [folder/'estimate.npy',folder/'target.wav',roi_path,folder/'landmark/speaker1.npz',folder/'MTCNN_AUDIT.json']}
                r=dict(id=sample['id'],role=role['role'],video=role['video'],video_sha256=role['video_sha256'],mix_sha256=sample['mix_sha256'],protocol_sha256=sha(WORK/'PROTOCOL.json'),runtime_sha256=runtime_sha,files=files,
                    roi_frames=len(roi),source_video_frames=expected,raw_audio_metrics=metrics(mix,estimate),saved_audio_metrics=metrics(mix,saved),subtype=info.subtype,
                    repeated_audio_forward_max_abs=error,seconds=time.time()-started,peak_allocated_mib=0.,execution_device="cpu",cuda_initialized=torch.cuda.is_initialized())
                save(receipt,r);per_role.append(r)
                save(WORK/'STATUS.json',dict(stage='separating',action=action,pid=os.getpid(),completed_streams=len(list((WORK/'separated').glob('*/*/complete.json'))),last_id=sample['id'],last_role=role['role'],updated_unix=time.time()))
                print('IIANet',sample['id'],role['role'],r['raw_audio_metrics'],flush=True)
                del inputs,wave,embedding,value,estimate,roi,normalized
            rows.append(dict(id=sample['id'],roles=per_role))
        dest=WORK/('CANARY_COMPLETE.json' if action=='canary' else 'SEPARATION_COMPLETE.json')
        save(dest,dict(passed=True,pairs=len(rows),streams=sum(len(row['roles']) for row in rows),rows=rows,protocol_sha256=sha(WORK/'PROTOCOL.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','canary','all']);a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.action)
