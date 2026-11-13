"""Same scientific readout on sealed/replayed/counterfactual short clip."""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_avatarforcing_table4_generate import WORK as RUN,guard,read,save,sha
from p2_spectre_coefficients import Encoder,decode25,crop_frames
from vico_literature_metrics import load_author


def main():
    import torch,cv2,face_alignment
    device=guard(.08,32768);cv2.setNumThreads(1)
    model=Encoder();detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device='cuda',flip_input=False)
    info=read(WORK/'INFERENCE_PARITY.json');uid=info['sample'];count=info['frames']
    sample=next(s for s in read(RUN/'protocol.json')['samples'] if s['id']==uid)
    gt=np.load(RUN/'features/gt'/(uid+'.npz'))['motion'];sealed=np.load(RUN/'features/avatarforcing'/(uid+'.npz'))['motion']
    raw,_=decode25(sample['gt_video'],count)
    images={
        'sealed_video_reextracted':decode25(RUN/'generated'/uid/'generated.mp4',count)[0],
        'official_seed137_replay':decode25(WORK/'official_preprocessor_seed137.mp4',count)[0],
        'counterfactual_avatar_cfg0':decode25(WORK/'counterfactual_avatar_cfg0.mp4',count)[0],
        'GT_first_frame_repeated':np.repeat(raw[:1],count,axis=0),
    }
    for key,path in [('preprocessed_reference_repeated',RUN/'inputs'/uid/'reference.png'),('reference_autoencoder_repeated',WORK/'reference_autoencoder_reconstruction.png')]:
        a=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);images[key]=np.repeat(a[None],count,axis=0)
    metrics={};values={};author=load_author()
    for key,frames in images.items():
        cropped,audit=crop_frames(frames,detector);v=model.chunks(cropped);values[key]=v
        np.savez_compressed(WORK/(key+'_spectre.npz'),motion=v)
        metrics[key]={}
        for group,sl in [('exp',slice(6,56)),('pose',slice(0,6))]:
            metrics[key][group]=dict(FD_to_GT=float(author.calculate_frechet_distance(*author.calculate_activation_statistics(gt[:,sl]),*author.calculate_activation_statistics(v[:,sl]))),
                temporal_variance_mean=float(np.var(v[:,sl],axis=0).mean()),
                coeff_MSE_to_sealed=float(np.mean((v[:,sl]-sealed[:,sl])**2)))
        print(key,metrics[key],flush=True)
    parity_error=float(np.abs(values['sealed_video_reextracted']-sealed).max())
    preprocessing_shift={group:float(np.square(values['preprocessed_reference_repeated'][:,sl]-values['GT_first_frame_repeated'][:,sl]).sum(axis=1).mean())
                         for group,sl in [('exp',slice(6,56)),('pose',slice(0,6))]}
    result=dict(sample=uid,frames=count,metrics=metrics,sealed_readout_reextract_max_abs=parity_error,
        reference_crop_only_squared_coefficient_shift=preprocessing_shift,device=device,
        source_sha256=sha(__file__),scope='predeclared shortest sample only; counterfactual is not a claimed Table4 setting')
    save(WORK/'READOUT_PROBE.json',result);print('READOUT_REPLAY_MAX_ABS',parity_error,flush=True)


if __name__=='__main__':main()
