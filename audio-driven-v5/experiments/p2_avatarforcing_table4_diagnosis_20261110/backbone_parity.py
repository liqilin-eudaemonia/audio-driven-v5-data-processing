"""Independent original torchvision v0.8.1 backbone versus deployed encoder."""
import ast
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts')]
from p2_spectre_coefficients import Encoder
from p2_avatarforcing_table4_generate import WORK as RUN,read,save,sha


def main():
    import cv2
    torch.set_num_threads(2);cv2.setNumThreads(1)
    assert not torch.cuda.is_initialized()
    source=WORK/'mobilenet_v0_8_1.py'
    tree=ast.parse(source.read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    ns=dict(nn=torch.nn);exec(compile(tree,str(source),'exec'),ns)
    old=ns['MobileNetV2']().eval()
    encoder=Encoder('cpu');modern=encoder.model.E_expression.encoder
    old.load_state_dict(modern.state_dict(),strict=True)
    sample=read(RUN/'protocol.json')['samples'][0]
    cap=cv2.VideoCapture(sample['gt_video']);frames=[]
    for _ in range(8):
        okay,frame=cap.read();assert okay
        frames.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(224,224)).transpose(2,0,1).astype('float32')/255)
    cap.release();images=torch.from_numpy(np.stack(frames))
    with torch.inference_mode():
        feature_delta=float((old.features(images)-modern.features(images)).abs().max())
        deployed=encoder.encode(images)
        encoder.model.E_expression.encoder=old
        original=encoder.encode(images)
    delta=float((deployed-original).abs().max())
    assert feature_delta<1e-5 and delta<1e-5
    result=dict(passed=True,device='CPU',frames=len(frames),features_max_abs=feature_delta,
        coefficient_max_abs=delta,original_source_sha256=sha(source),
        original_source_url='https://raw.githubusercontent.com/pytorch/vision/v0.8.1/torchvision/models/mobilenet.py',
        checkpoint_sha256=sha(RUN/'assets/spectre_model.tar'),script_sha256=sha(__file__),
        scope='Original old backbone source, strict same states and same inputs; this excludes backbone architecture substitution on tested inputs, not crop equivalence')
    save(WORK/'BACKBONE_PARITY.json',result);print(result)


if __name__=='__main__':main()
