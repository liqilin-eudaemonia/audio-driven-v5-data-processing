"""Official DECA coarse encoder and crop; no FLAME geometry/renderer required."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/"experiments/p2_compare_vico43_20261106/learning2listen"
REF=ROOT/"experiments/p2_vico_motion8000_20261106"
VENDOR=ROOT/"compare/deployment/vendor/DECA"
WEIGHT=ROOT/"compare/deployment/assets/deca_model.tar"


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+"\n")


def main():
    import torch
    import cv2
    from skimage.transform import estimate_transform,warp
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="7"
    sys.path.insert(0,str(VENDOR))
    from decalib.models.encoders import ResnetEncoder
    torch.set_num_threads(2);cv2.setNumThreads(1)
    model=ResnetEncoder(236);checkpoint=torch.load(WEIGHT,map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["E_flame"],strict=True);model.eval().cuda()
    p=json.loads((REF/"protocol.json").read_text())
    source={str(q):sha(q) for q in [WEIGHT,VENDOR/"decalib/models/encoders.py",VENDOR/"decalib/models/resnet.py",VENDOR/"decalib/datasets/datasets.py",Path(__file__)]}
    save(WORK/"DECA_PROTOCOL.json",dict(extractor="official DECA E_flame; strict checkpoint load; encoder only, no FLAME model or renderer",
        source=source,parameter_order="shape100,tex50,exp50,pose6,cam3,light27; motion=exp[150:200]+pose[200:206]",
        crop="official TestData landmarks path: FAN68 bbox, old_size=mean(width,height)*1.1, scale1.25, similarity crop224, skimage warp",
        image_input="original256 RGB via OpenCV; no JPEG intermediate",landmarks="same cached official FAN as previous run; all speaker and GT frames detected",
        resolution=224,fps=30,precision="FP32",roles=["speaker","gt"],inference_gt_access=False))
    start=time.time();records=[]
    for sample in p["samples"]:
        uid=sample["id"]
        for role in ["speaker","gt"]:
            dest=WORK/"features"/role/(uid+".npz");receipt=dest.with_suffix(".json")
            if receipt.exists():records.append(json.loads(receipt.read_text()));continue
            info=sample["streams"][role]
            landmark=np.load(REF/"features"/role/(uid+".npz"))["landmarks"]
            assert info["sha256"]==sha(info["path"])
            cap=cv2.VideoCapture(info["path"]);frames=[]
            while True:
                ok,im=cap.read()
                if not ok:break
                frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            cap.release();assert len(frames)==len(landmark)==info["frames"]
            def crop(pair):
                im,lm=pair;left,top=lm.min(0);right,bottom=lm.max(0)
                old_size=(right-left+bottom-top)/2*1.1;center=np.array([right-(right-left)/2,bottom-(bottom-top)/2])
                size=int(old_size*1.25);src=np.array([[center[0]-size/2,center[1]-size/2],[center[0]-size/2,center[1]+size/2],[center[0]+size/2,center[1]-size/2]])
                dst=np.array([[0,0],[0,223],[223,0]])
                tform=estimate_transform("similarity",src,dst)
                result=warp(im/255.,tform.inverse,output_shape=(224,224)).transpose(2,0,1).astype(np.float32)
                return result,tform.params.astype(np.float32)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:cropped=list(pool.map(crop,zip(frames,landmark)))
            coeff=[]
            with torch.no_grad():
                for i in range(0,len(cropped),128):
                    x=torch.from_numpy(np.stack([x[0] for x in cropped[i:i+128]])).cuda()
                    coeff.append(model(x).cpu().numpy())
            coeff=np.concatenate(coeff);motion=np.concatenate([coeff[:,150:200],coeff[:,200:206]],axis=1)
            assert motion.shape==(len(frames),56) and np.isfinite(motion).all()
            dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,motion=motion,coeff=coeff,transform_params=np.stack([x[1] for x in cropped]))
            r=dict(id=uid,role=role,frames=len(frames),source_sha256=info["sha256"],feature_sha256=sha(dest),deca_protocol_sha256=sha(WORK/"DECA_PROTOCOL.json"))
            save(receipt,r);records.append(r)
            save(WORK/"deca_status.json",dict(completed=len(records),total=86,seconds=time.time()-start))
            print("extracted",len(records),uid,role,flush=True)
    save(WORK/"DECA_COMPLETE.json",dict(streams=86,frames=sum(x["frames"] for x in records),seconds=time.time()-start,records=records))


if __name__=="__main__":main()
