"""Infer observable reference framing from published static background pixels."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
PUBLISHED=ROOT/'experiments/p2_avatarforcing_table4_diagnosis_20261110/published_supplement'


def match(target,source):
    sift=cv2.SIFT_create(nfeatures=10000)
    mask=np.full(target.shape[:2],255,np.uint8)
    mask[:90]=0
    mask[90:430,120:400]=0  # Exclude the generated face and central torso.
    ka,da=sift.detectAndCompute(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY),mask)
    kb,db=sift.detectAndCompute(cv2.cvtColor(source,cv2.COLOR_BGR2GRAY),None)
    if da is None or db is None:return dict(valid=False,reason='No descriptors')
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=150))
    pairs=[a for a,b in matcher.knnMatch(da,db,k=2) if a.distance<.7*b.distance]
    if len(pairs)<8:return dict(valid=False,reason='Too few ratio-filtered background matches',matches=len(pairs))
    x=np.float32([ka[m.queryIdx].pt for m in pairs]);y=np.float32([kb[m.trainIdx].pt for m in pairs])
    matrix,good=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=10000,confidence=.999)
    if matrix is None:return dict(valid=False,reason='No similarity transform',matches=len(pairs))
    good=good.ravel().astype(bool)
    error=np.linalg.norm(cv2.transform(x[None],matrix)[0]-y,axis=1)
    corners=cv2.transform(np.array([[[0,0],[512,0],[0,512],[512,512]]],np.float32),matrix)[0]
    scale=float(np.linalg.norm(matrix[0,:2]))
    return dict(valid=int(good.sum())>=12,matches=len(pairs),inliers=int(good.sum()),
        median_error=float(np.median(error[good])),matrix=matrix.tolist(),source_corners=corners.tolist(),
        source_square_side=512*scale,rotation_deg=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))))


def selected_frames(path):
    cap=cv2.VideoCapture(str(path));out={};index=0
    while index<=75:
        okay,frame=cap.read()
        if not okay:break
        if index in [0,25,75]:out[index]=frame
        index+=1
    cap.release();return out


def main():
    cv2.setNumThreads(2);cv2.setRNGSeed(42)
    manifest=json.loads((WORK/'ORIGINAL_TEST43_MANIFEST.json').read_text())['records']
    rows=[]
    for base,uid in [('03_listening_01','ClNUxPsUByg_000210_000221.0'),('03_listening_02','Ew2z_sYABE0_000063_000083.0')]:
        record=next(r for r in manifest if r['id']==uid and r['role']=='listener')
        cap=cv2.VideoCapture(record['path']);okay,source=cap.read();cap.release();assert okay
        control=WORK/'inputs'/uid/'listener/reference.png'
        probes=[dict(kind='current_reference',frame=0,**match(cv2.imread(str(control)),source))]
        for kind,path in [('published_AF',PUBLISHED/(base+'_avatar.mp4')),
                          ('current_AF',WORK/'native_generated'/uid/'generated.mp4')]:
            for index,target in selected_frames(path).items():
                probes.append(dict(kind=kind,frame=index,**match(target,source)))
        rows.append(dict(id=uid,original=record['path'],probes=probes))
        print(uid,[(r['kind'],r['frame'],r.get('inliers'),round(r.get('source_square_side',0),2),r['valid']) for r in probes],flush=True)
    result=dict(rows=rows,method='Static background SIFT ratio.7, similarity RANSAC2px; face and title excluded',
        limitations=['Animated background warp may affect the affine estimate',
                     'Static background cannot determine which source frame was the reference',
                     'No generated/GT motion metric or Table4 value used to choose framing'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (WORK/'PUBLISHED_REFERENCE_BACKGROUND_REGISTRATION.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
