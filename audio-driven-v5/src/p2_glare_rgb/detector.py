"""MTCNN consumes RGB; FER's grayscale emotion branch consumes original BGR."""
import csv,pickle
from pathlib import Path
from p2_glare_eval.common import ROOT,read,sha,atomic_json,complete,artifact_valid,code_identity as original_code
from p2_glare_eval.detector import EMOTIONS,valid_face,valid_emotion,face_iou,run_rules,runtime_identity
from p2_glare_eval.upstream import DETECTOR,module,verify_sources
from p2_glare_eval.media import probe

def policy():
    p=read(ROOT/'configs/p2_glare_detector_v1.json')
    return dict(p,schema='glare-rgb-face-detector-v2',fer_input='BGR for emotion; RGB for MTCNN face rectangles',missing_frame_policy='reject_clip_no_interpolation_no_zero_fill')

def code_identity():
    return {**original_code(),**{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}}

def rgb_emotions(model,bgr):
    import cv2
    rectangles=model.find_faces(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    return model.detect_emotions(bgr,face_rectangles=rectangles)

class Engine:
    def __init__(self):
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
        import torch,tensorflow as tf
        from fer import FER
        torch.set_num_threads(2);tf.config.set_visible_devices([],'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
        self.runtime=dict(runtime_identity(),fer_api='find_faces(RGB), detect_emotions(BGR, explicit_rectangles)',model_lifetime='persistent_stateless_FER_per_worker')
        self.sources=verify_sources();self.policy=policy();self.code=code_identity();self.emotion=FER(mtcnn=True)
    def identity(self,video_hash):
        return dict(video_sha256=video_hash,policy=self.policy,runtime=self.runtime,upstream=self.sources,code=self.code)
    def detect(self,video,folder,video_hash):
        import cv2
        video,folder=Path(video),Path(folder);ident=self.identity(video_hash)
        if sha(video)!=video_hash:raise ValueError('Input video changed')
        if artifact_valid(folder,ident):return dict(status='complete',reused=True)
        folder.mkdir(parents=True,exist_ok=True);geometry=probe(video,self.policy['fps'])
        face=module('_p2_glare_rgb_face',DETECTOR/'nod_headshake/face_detection/detect_face_multithread.py')
        face.process_video(video,folder/'face.pkl',str(DETECTOR/'nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task'))
        with (folder/'face.pkl').open('rb') as f:faces=pickle.load(f)
        if len(faces)!=geometry['frame_count']:raise ValueError('Face extractor lost frames')
        capture=cv2.VideoCapture(str(video));raw=[];bad=[];face_count=emotion_count=0
        try:
            with (folder/'fer.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['frame_index',*EMOTIONS]);writer.writeheader()
                for i,face_frame in enumerate(faces):
                    ok,bgr=capture.read()
                    if not ok:raise ValueError('FER decoder lost frames')
                    found=rgb_emotions(self.emotion,bgr)
                    serial=[dict(box=[float(x) for x in f['box']],emotions={k:float(v) for k,v in f['emotions'].items()}) for f in found]
                    raw.append(dict(frame_index=i,faces=serial));fv,ev=valid_face(face_frame),valid_emotion(serial)
                    face_count+=int(fv);emotion_count+=int(ev)
                    same=fv and ev and face_iou(face_frame['bbox_px'],serial[0]['box'])>=self.policy['min_face_iou']
                    if not same:bad.append(dict(frame_index=i,mediapipe_valid=fv,fer_valid=ev,association_valid=bool(same),fer_faces=len(serial)))
                    writer.writerow(dict(frame_index=i,**(serial[0]['emotions'] if ev else {k:'' for k in EMOTIONS})))
                if capture.read()[0]:raise ValueError('FER decoder has extra frames')
        finally:capture.release()
        atomic_json(folder/'fer_raw.json',raw)
        quality=dict(frames=len(faces),mediapipe_valid_frames=face_count,fer_valid_frames=emotion_count,jointly_valid_frames=len(faces)-len(bad),invalid_frames=bad,required_frame_coverage=1.0)
        atomic_json(folder/'quality.json',quality)
        if bad:
            atomic_json(folder/'failed.json',dict(identity=ident,error_type='IncompleteDetection',error=f'{len(bad)}/{len(faces)} invalid frames',quality=quality))
            # Invalid is a completed diagnostic, not a fabricated negative CSV.
            complete(folder,ident,[folder/n for n in ('face.pkl','fer.csv','fer_raw.json','quality.json','failed.json')],geometry=geometry,valid=False)
            return dict(status='invalid',invalid_frames=len(bad))
        run_rules(video,folder,self.policy,geometry['frame_count'])
        complete(folder,ident,[folder/n for n in ('scores.csv','face.pkl','fer.csv','fer_raw.json','quality.json')],geometry=geometry,valid=True,human_verified=False,detector_based=True)
        return dict(status='complete',invalid_frames=0)
