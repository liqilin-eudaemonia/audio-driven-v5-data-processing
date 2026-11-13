"""Recover visible author speaker crop/time mapping by image registration only."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

WORK=Path(__file__).resolve().parent
DIAG=WORK.parent/'p2_avatarforcing_table4_diagnosis_20261110'


def read_frame(cap,index):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));okay,frame=cap.read()
    if not okay:raise ValueError(f'No source frame{index}')
    return frame


def match(target,source):
    sift=cv2.SIFT_create(nfeatures=4000)
    mask=np.full(target.shape[:2],255,np.uint8);mask[:85]=0
    k1,d1=sift.detectAndCompute(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY),mask)
    k2,d2=sift.detectAndCompute(cv2.cvtColor(source,cv2.COLOR_BGR2GRAY),None)
    if d1 is None or d2 is None:return None
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=100))
    good=[a for a,b in matcher.knnMatch(d1,d2,k=2) if a.distance<.7*b.distance]
    if len(good)<8:return None
    x=np.float32([k1[m.queryIdx].pt for m in good]);y=np.float32([k2[m.trainIdx].pt for m in good])
    matrix,inliers=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=5000,confidence=.999)
    if matrix is None:return None
    valid=inliers.ravel().astype(bool);error=np.linalg.norm(cv2.transform(x[None],matrix)[0]-y,axis=1)
    corners=cv2.transform(np.array([[[0,0],[512,0],[0,512],[512,512]]],np.float32),matrix)[0]
    return dict(matrix=matrix.tolist(),inliers=int(valid.sum()),matches=len(good),median_error=float(np.median(error[valid])),
        source_corners=corners.tolist(),scale=float(np.linalg.norm(matrix[0,:2])),rotation_deg=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))))


def main():
    cv2.setNumThreads(2);cv2.setRNGSeed(42)
    manifest=json.loads((WORK/'ORIGINAL_TEST43_MANIFEST.json').read_text())['records']
    rows=[]
    for base,uid in [('03_listening_01','ClNUxPsUByg_000210_000221.0'),('03_listening_02','Ew2z_sYABE0_000063_000083.0')]:
        record=next(r for r in manifest if r['id']==uid and r['role']=='speaker')
        source=cv2.VideoCapture(record['path']);sfps=source.get(cv2.CAP_PROP_FPS)
        published=DIAG/'published_supplement'/(base+'_speaker.mp4');cap=cv2.VideoCapture(str(published))
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));probes=[]
        for frame in [0,25,75,n-26]:
            target=read_frame(cap,frame);candidates=[]
            # Probe the declared25fps grid and the possible native-frame-index
            # interpretation. This reconstructs observable frames, not FD/GT lag.
            expected=frame*sfps/25
            indices=set(max(0,int(round(expected))+delta) for delta in [-2,-1,0,1,2])
            indices.update(max(0,frame+delta) for delta in [-1,0,1])
            for i in sorted(indices):
                original=read_frame(source,i);answer=match(target,original)
                if answer is not None:candidates.append(dict(source_index=i,**answer))
            if not candidates:raise ValueError('No geometric registration: '+uid)
            best=max(candidates,key=lambda row:(row['inliers'],-row['median_error']))
            probes.append(dict(published_index=frame,published_time=frame/25,source_fps=sfps,best=best,candidates=candidates))
            print(uid,'pubframe',frame,'source',best['source_index'],'inliers',best['inliers'],'error',best['median_error'],'corners',best['source_corners'],flush=True)
        rows.append(dict(id=uid,published=str(published),original=record['path'],original_sha256=record['sha256'],probes=probes))
        source.release();cap.release()
    result=dict(rows=rows,method='SIFT descriptor ratio.7 and similarity RANSAC2.5px; title area masked; selection by source-image inliers, no generated/GT metric selection',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (WORK/'AUTHOR_CROP_REGISTRATION.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
