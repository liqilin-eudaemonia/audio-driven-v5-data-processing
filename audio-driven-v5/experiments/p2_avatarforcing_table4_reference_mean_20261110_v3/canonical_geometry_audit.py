from pathlib import Path
import cv2,numpy as np,json,time,hashlib
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');w=root/'experiments/p2_avatarforcing_table4_reference_mean_20261110_v3';p=json.loads((w/'NATIVE_GENERATION_PROTOCOL.json').read_text());cv2.setNumThreads(1);cv2.setRNGSeed(42);sift=cv2.SIFT_create(nfeatures=4000);rows=[]
for s in p['samples']:
 for role,key in [('listener','gt_video'),('speaker','speaker_video')]:
  cap=cv2.VideoCapture(s[key]);ok,a=cap.read();cap.release();assert ok
  b=cv2.imread(str(w/'inputs'/s['id']/role/'raw25/00001.jpg'));assert b is not None
  ka,da=sift.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None);kb,db=sift.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
  r=dict(id=s['id'],role=role,canonical_video=s[key],canonical_shape=list(a.shape),source_shape=list(b.shape))
  if da is None or db is None:r.update(valid=False,reason='no descriptors')
  else:
   matches=[x for x,y in cv2.BFMatcher().knnMatch(da,db,k=2) if x.distance<.7*y.distance]
   if len(matches)<8:r.update(valid=False,reason='few matches',matches=len(matches))
   else:
    x=np.float32([ka[m.queryIdx].pt for m in matches]);y=np.float32([kb[m.trainIdx].pt for m in matches]);matrix,good=cv2.estimateAffine2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=5000,confidence=.999)
    if matrix is None:r.update(valid=False,reason='no affine',matches=len(matches))
    else:
     error=np.linalg.norm(cv2.transform(x[None],matrix)[0]-y,axis=1);sv=np.linalg.svd(matrix[:,:2],compute_uv=False);r.update(valid=int(good.sum())>=8,matches=len(matches),inliers=int(good.sum()),median_error=float(np.median(error[good.ravel().astype(bool)])),source_pixels_per_canonical_x=float(np.linalg.norm(matrix[:,0])),source_pixels_per_canonical_y=float(np.linalg.norm(matrix[:,1])),anisotropy_ratio=float(sv[0]/sv[1]),matrix=matrix.tolist())
  rows.append(r)
valid=[r for r in rows if r['valid']];out=dict(rows=rows,pairs=43,valid_streams=len(valid),total_streams=86,anisotropy_median=float(np.median([r['anisotropy_ratio'] for r in valid])),anisotropy_above_1_10=sum(r['anisotropy_ratio']>1.1 for r in valid),scope='First-frame identity and affine crop geometry diagnostic; no metric scores used; no new scoring config selected',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checked_unix=time.time());(w/'CANONICAL_CROP_GEOMETRY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='rows'},flush=True)
for r in rows:
 if r['id']=='UCA1A5GqCdQ_000610_000619.1':print(r)
