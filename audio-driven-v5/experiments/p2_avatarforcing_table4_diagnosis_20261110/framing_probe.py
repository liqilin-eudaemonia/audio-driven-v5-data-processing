"""Same author pixels/landmarks, changed field of view, static coefficient shift."""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts')]
from p2_avatarforcing_table4_generate import guard,save,sha
from p2_spectre_coefficients import Encoder,source_crop_function


def main():
    import cv2,face_alignment
    from skimage.transform import warp
    guard(.08,32768);cv2.setNumThreads(1)
    model=Encoder();detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device='cuda',flip_input=False)
    transform=source_crop_function();rows=[]
    for base in ['03_listening_01','03_listening_02']:
        cap=cv2.VideoCapture(str(WORK/'published_supplement'/(base+'_avatar.mp4')));okay,bgr=cap.read();cap.release();assert okay
        frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);lm=detector.get_landmarks_from_image(frame)[0]
        # Geometry chosen before this probe: round the observed local face-width
        # fraction (~0.848) to0.85. No score-based crop search.
        width=int(round(np.ptp(lm[:,0])/.85));centre=(lm.min(axis=0)+lm.max(axis=0))/2
        x1=int(round(centre[0]-width/2));y1=int(round(centre[1]-width/2));x2=x1+width;y2=y1+width
        assert 0<=x1<x2<=512 and 0<=y1<y2<=512
        tight=frame[y1:y2,x1:x2];shifted=lm-np.array([x1,y1])
        crops=[]
        for name,image,points in [('wide',frame,lm),('tight',tight,shifted)]:
            out=warp(image,transform(image,points,scale=1.6).inverse,output_shape=(224,224)).transpose(2,0,1).astype('float32')
            cv2.imwrite(str(WORK/(base+'_'+name+'_spectre_crop.png')),cv2.cvtColor((out.transpose(1,2,0)*255).round().astype('uint8'),cv2.COLOR_RGB2BGR))
            crops.append(model.chunks(np.repeat(out[None],5,axis=0))[2])
        delta=crops[0]-crops[1]
        rows.append(dict(video=base,tight_rectangle=[x1,y1,x2,y2],target_face_width_fraction=.85,
            landmarks='same original landmarks translated exactly, no second detection',
            expression_squared_L2_shift=float(np.square(delta[6:]).sum()),pose_squared_L2_shift=float(np.square(delta[:6]).sum()),
            expression_mean_abs_shift=float(np.abs(delta[6:]).mean())))
    save(WORK/'FRAMING_PROBE.json',dict(rows=rows,script_sha256=sha(__file__),
        scope='Static identical-face-pixels control; only surrounding field of view changes. No full-sequence rescore and no asserted author quantitative crop policy.'))
    print(rows,flush=True)


if __name__=='__main__':main()
