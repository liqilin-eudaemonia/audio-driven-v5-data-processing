"""Offline cross-check against upstream constructors and existing FVD pipeline."""
import argparse
import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import torch
from .backends import Networks, Faces
from .media import decode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--face-video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    assets=json.loads(args.assets.read_text())
    torch.set_num_threads(2); cv2.setNumThreads(2)
    network=Networks(assets,batch_size=2)
    report={}
    rng=np.random.default_rng(5)
    g=rng.integers(0,256,(2,96,128,3),dtype=np.uint8)
    p=np.clip(g.astype(int)+20,0,255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); (root/'checkpoints').mkdir()
        for key in ('alexnet','fid_inception'):
            src=Path(assets[key]).resolve()
            (root/'checkpoints'/src.name).symlink_to(src)
        old=torch.hub.get_dir(); torch.hub.set_dir(str(root))
        try:
            import lpips
            ours=np.asarray(network.lpips(g,p))
            reference=lpips.LPIPS(net='alex',version='0.1',pnet_rand=False,verbose=False).eval()
            with torch.inference_mode():
                ref=reference(network.tensor(g)*2-1,network.tensor(p)*2-1).flatten().numpy()
            error=float(np.max(np.abs(ours-ref)))
            assert error<1e-7,error
            for name,param in reference.state_dict().items():
                assert torch.equal(param,network.models['lpips'].state_dict()[name]),name
            report['lpips']={'max_abs_error':error,'all_weights_equal':True}
            print('LPIPS reference passed',flush=True)
            from PIL import Image
            from pytorch_fid.inception import InceptionV3
            from pytorch_fid.fid_score import get_activations
            files=[]
            for i,frame in enumerate(g):
                dest=root/f'{i}.png'; Image.fromarray(frame).save(dest);files.append(str(dest))
            ours=network.fid(g)
            reference=InceptionV3([3]).eval()
            ref=get_activations(files,reference,batch_size=2,dims=2048,device='cpu',num_workers=0)
            error=float(np.max(np.abs(ours-ref)))
            assert error<1e-6,error
            report['fid_features']={'max_abs_error':error,'dimension':int(ours.shape[1])}
            print('FID PNG reference passed',flush=True)
        finally:
            torch.hub.set_dir(old)
    from ..fvd.frechet_video_distance import batch_to_bcthw_uint8,load_torchscript_detector
    frames=rng.integers(0,256,(32,320,480,3),dtype=np.uint8)
    ours=network.fvd(frames)
    detector=load_torchscript_detector(assets['i3d'],torch.device('cpu'))
    with torch.inference_mode():
        reference=detector(batch_to_bcthw_uint8([torch.from_numpy(frames)]),
                           rescale=True,resize=False,return_features=True).numpy()
    error=float(np.max(np.abs(ours-reference)))
    assert error<1e-6,error
    # Quantify why the old area-resize path was incompatible.
    old_area=np.stack([cv2.resize(f,(224,224),interpolation=cv2.INTER_AREA) for f in frames])
    old_tensor=torch.from_numpy(old_area).permute(3,0,1,2)[None]
    new_tensor=batch_to_bcthw_uint8([torch.from_numpy(frames)])
    report['fvd_features']={'max_abs_error':error,'input_shape':list(frames.shape),
        'old_area_vs_bilinear_mean_pixel_difference':float((old_tensor.float()-new_tensor.float()).abs().mean())}
    print('FVD legacy non-224 reference passed',flush=True)
    # Match the official DeepFace analyze API on the exact same detected crop.
    face=Faces(assets,emotion=True,threads=2)
    rgb=decode(args.face_video)[0][:1]
    _,boxes,labels,_=face.extract(rgb)
    assert labels[0]>=0,'No face in supplied test video'
    bgr=cv2.cvtColor(rgb[0],cv2.COLOR_RGB2BGR)
    x1,y1,x2,y2=np.rint(boxes[0]).astype(int)
    crop=bgr[max(y1,0):min(y2,len(bgr)),max(x1,0):min(x2,bgr.shape[1])]
    from deepface import DeepFace
    from .core import EMOTIONS
    analysis=DeepFace.analyze(crop,actions=['emotion'],detector_backend='skip',
                              align=False,enforce_detection=False,silent=True)[0]
    assert EMOTIONS[labels[0]]==analysis['dominant_emotion']
    from deepface.modules.preprocessing import resize_image
    scores=np.atleast_2d(face.emotion.predict(resize_image(crop.astype(np.float64)/255,(224,224))[0]))[0]
    scores=scores/scores.sum()*100
    expected=np.array([analysis['emotion'][label] for label in EMOTIONS])
    error=float(np.max(np.abs(scores-expected)))
    assert error<1e-4,error
    report['emotion_same_crop']={'probability_percent_max_abs_error':error,'label':EMOTIONS[labels[0]],
                                 'note':'Checks classifier preprocessing; not equivalence of face detectors.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
