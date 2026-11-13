"""Read official released qualitative outputs through the same SPECTRE pipeline."""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
RUN=ROOT/'experiments/p2_avatarforcing_table4_recheck_20261110'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_avatarforcing_table4_generate import guard,read,save,sha
from p2_spectre_coefficients import Encoder,decode25,crop_frames
from vico_literature_metrics import load_author


def main():
    import cv2,face_alignment
    device=guard(.08,32768);cv2.setNumThreads(1)
    model=Encoder();detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device='cuda',flip_input=False)
    author=load_author();rows=[]
    for base,uid in [('03_listening_01','ClNUxPsUByg_000210_000221.0'),('03_listening_02','Ew2z_sYABE0_000063_000083.0')]:
        gt=np.load(RUN/'features/gt'/(uid+'.npz'))['motion'];speaker=np.load(RUN/'features/speaker'/(uid+'.npz'))['motion'];ours=np.load(RUN/'features/avatarforcing'/(uid+'.npz'))['motion']
        encoded={};audits={};face_width={}
        for role in ['speaker','avatar']:
            video=WORK/'published_supplement'/(base+'_'+role+'.mp4');frames,_=decode25(video,len(gt));cache=WORK/(base+'_'+role+'_spectre.npz')
            crop,audit=crop_frames(frames,detector);values=model.chunks(crop);np.savez_compressed(cache,motion=values)
            landmarks=detector.get_landmarks_from_image(frames[0])[0]
            face_width['author_'+role]=float(np.ptp(landmarks[:,0])/frames.shape[2])
            encoded[role]=values;audits[role]=audit;print('author demo extracted',base,role,len(values),flush=True)
        cap=cv2.VideoCapture(str(RUN/'generated'/uid/'generated.mp4'));okay,frame=cap.read();cap.release();assert okay
        landmarks=detector.get_landmarks_from_image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))[0]
        face_width['our_generated']=float(np.ptp(landmarks[:,0])/frame.shape[1])
        scores={}
        for name,v in [('sealed_local_generation',ours),('released_author_video',encoded['avatar'])]:
            scores[name]={}
            for group,sl in [('exp',slice(6,56)),('pose',slice(0,6))]:
                scores[name][group]=dict(FD=float(author.calculate_frechet_distance(*author.calculate_activation_statistics(gt[:,sl]),*author.calculate_activation_statistics(v[:,sl]))),
                    temporal_variance_mean=float(np.var(v[:,sl],axis=0).mean()),pooled_flattened_variance=float(np.var(v[:,sl])),
                    rPCC_same_canonical_speaker=float(abs(np.corrcoef(gt[:,sl].ravel(),speaker[:,sl].ravel())[0,1]-np.corrcoef(v[:,sl].ravel(),speaker[:,sl].ravel())[0,1])))
        # A same-source signal check, independent of FD. Constant coefficients are excluded.
        a=speaker.astype(float)-speaker.mean(axis=0);b=encoded['speaker'].astype(float)-encoded['speaker'].mean(axis=0)
        denom=np.sqrt(np.sum(a*a,axis=0)*np.sum(b*b,axis=0));valid=denom>1e-15
        corr=np.full(56,np.nan);corr[valid]=np.sum(a*b,axis=0)[valid]/denom[valid]
        rows.append(dict(id=uid,frames=len(gt),scores=scores,face_landmark_width_fraction=face_width,
            original_vs_author_speaker_per_coefficient_correlations=corr.tolist(),face_audit=audits))
        save(WORK/'AUTHOR_DEMO_PARTIAL.json',dict(rows=rows,device=device))
    result=dict(pairs=2,rows=rows,device=device,encoder=model.info,script_sha256=sha(__file__),
        audio_alignment=read(WORK/'AUTHOR_DEMO_AUDIO_ALIGNMENT.json'),
        scope='Two official qualitative examples whose audio matches canonical test clips; not the full Table4 outputs, not proof these exact encodings were quantitatively scored; overlay/cropping/compression differences disclosed',
        original_results_sha256=sha(RUN/'RESULTS.json'))
    save(WORK/'AUTHOR_DEMO_READOUT.json',result);print('AUTHOR DEMO READOUT COMPLETE',flush=True)


if __name__=='__main__':main()
