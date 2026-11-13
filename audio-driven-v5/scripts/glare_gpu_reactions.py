"""Execution-only GPU batched MTCNN; native reaction assembly remains unchanged."""
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def service_type(pipeline):
    class GPUService(pipeline.CPUService):
        def __init__(self,gpu,slot):
            folder=pipeline.WORK/'workers';folder.mkdir(exist_ok=True)
            self.log=(folder/('cpu_gpu%d_%d.log'%(gpu,slot))).open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',GLARE_FER_GPU=str(gpu),OMP_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'))
            self.process=subprocess.Popen([str(pipeline.MP_PY),str(Path(__file__).resolve()),'server'],
                cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,start_new_session=True)
    return GPUService


def fer_server():
    import contextlib,json
    import numpy as np,torch,cv2,tensorflow as tf
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
    from fer.fer import FER
    from listen_j_corrected_reaction import _overlap
    from glare_shared_reactions import EMOTIONS
    # The production helper import hides CUDA; restore only this FER child.
    os.environ['CUDA_VISIBLE_DEVICES']=os.environ['GLARE_FER_GPU']
    with contextlib.redirect_stdout(sys.stderr):model=FER(mtcnn=True,use_tflite=True)
    assert model._mtcnn.device.type=='cuda','Requested GPU detector unavailable'
    for line in sys.stdin:
        try:
            req=json.loads(line);pixels=np.load(req['pixels'],mmap_mode='r');boxes=req['bboxes'];n=len(pixels)
            probs=np.zeros((n,7),np.float32);detected=np.zeros(n,bool)
            association=np.zeros(n,np.float32);counts=np.zeros(n,np.int16)
            for start in range(0,n,32):
                frames=np.array(pixels[start:start+32])
                with contextlib.redirect_stdout(sys.stderr):rectangles,_=model._mtcnn.detect(frames)
                for j,(frame,bb) in enumerate(zip(frames,rectangles)):
                    i=start+j
                    rects=[] if bb is None else [[int(f[0]),int(f[1]),int(f[2])-int(f[0]),int(f[3])-int(f[1])] for f in bb]
                    with contextlib.redirect_stdout(sys.stderr):faces=model.detect_emotions(frame,face_rectangles=rects)
                    counts[i]=len(faces)
                    if not faces:continue
                    box=boxes[i];pick=max(range(len(faces)),key=lambda k:_overlap(faces[k]['box'],box) if box else faces[k]['box'][2]*faces[k]['box'][3])
                    detected[i]=True;probs[i]=[faces[pick]['emotions'][k] for k in EMOTIONS]
                    association[i]=_overlap(faces[pick]['box'],box) if box else 0.
            np.savez(req['output'],fer=probs,fer_detected=detected,face_iou=association,fer_face_count=counts)
            reply=dict(ok=True,output=req['output'])
        except Exception:
            import traceback
            reply=dict(ok=False,error=traceback.format_exc())
        print(json.dumps(reply),flush=True)


if __name__=='__main__':
    import glare_shared_reactions as native
    if sys.argv[1]=='fer-server':
        os.environ['CUDA_VISIBLE_DEVICES']=os.environ['GLARE_FER_GPU']
        fer_server()
    else:
        # Native Detector launches this file for its FER child; MediaPipe remains CPU.
        native.__file__=str(Path(__file__).resolve())
        native.server()
