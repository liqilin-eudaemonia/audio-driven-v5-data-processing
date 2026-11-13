"""CPU annotation-pipeline candidate rules; candidates never become truth labels."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

from .contracts import read, save, sha
from .media import ROOT
from .hints import smile_trace


def measure(entry, fer_python):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'scripts'))
    # These original modules intentionally disable CUDA; use ONLY in this CPU subprocess.
    from run_glare_nodding_pilot import module, FACE, MODEL, NOD, VENDOR, PROFILES, intervals
    from listen_j_corrected_reaction import _vendor_defaults, can_possibly_pass_expression_gate
    from check_glare_headshake_direction import PARAMS as SH, assess as assess_shake
    from check_glare_expression_gate import PARAMS as EX, blendshapes, assess_frown
    video = Path(entry['visual25']); folder = Path(entry['folder']); pose = folder / 'pose.pkl'
    if sha(video) != entry['visual25_sha256']: raise ValueError('Prepared video changed')
    face = module('_presence_face', FACE); nod = module('_presence_nod', NOD)
    with contextlib.redirect_stdout(sys.stderr): face.process_video(video, pose, str(MODEL))
    detected, pitch, yaw = nod.load_pose_from_pkl(pose)
    with pose.open('rb') as f: raw = pickle.load(f)  # Created locally immediately above.
    duration = entry['duration_s']; result = dict(schema='annotation-visual-candidates-v2',
        media_id=entry['media_id'], face_frames=len(raw), detected_frames=int(detected.sum()),
        frame_coverage=float(detected.mean()) if len(raw) else 0., fps=25,
        framewise_face_detected=[bool(v) for v in detected],
        face_model_sha256=sha(MODEL), parameters=dict(nodding=PROFILES['recall'], headshake_gate=SH, expression_gate=EX),
        cues=[], candidate_only=True, missing_candidates_are_not_absence=True)
    if not detected.any():
        result['error'] = 'no_detected_face'; save(folder/'candidates.json', result); return result
    score = nod.compute_nodding_intensity(detected, pitch, yaw, argparse.Namespace(**PROFILES['recall']))
    result['cues'] += [dict(e, type='nodding') for e in intervals(score, detected, 25, duration)]
    shake = module('_presence_shake', VENDOR/'reaction_detector/head_shaking/run_head_shaking_intensity_mt.py')
    cfg = _vendor_defaults(shake); result['parameters']['head_shaking'] = cfg
    score = shake.compute_head_shaking_intensity(detected, pitch, yaw, argparse.Namespace(**cfg))
    yp = shake.moving_average(shake.fill_pose_gaps(np.asarray(yaw), detected), SH['smooth_frames'])
    pp = shake.moving_average(shake.fill_pose_gaps(np.asarray(pitch), detected), SH['smooth_frames'])
    for e in intervals(score, detected, 25, duration):
        a,b = e['start_frame'],e['end_frame_exclusive']
        if assess_shake(yp[a:b],pp[a:b],detected[a:b],SH)['retain']: result['cues'].append(dict(e,type='head_shaking'))
    signal = blendshapes(pose)
    possible, _ = can_possibly_pass_expression_gate(signal, EX)
    result['frown_fer_status'] = 'skipped_by_original_blendshape_bound'
    if possible:
        request = dict(video=str(video), bboxes=[x.get('bbox_px') for x in raw])
        command = [str(fer_python),str(ROOT/'scripts/listen_j_corrected_reaction.py'),'--fer-worker']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        response = subprocess.run(command,input=json.dumps(request)+'\n',text=True,capture_output=True,env=env,timeout=300)
        (folder/'fer.log').write_text(response.stderr)
        reply = json.loads(response.stdout.strip().splitlines()[-1])
        if not reply.get('ok') or reply['frames'] != len(raw): raise ValueError('FER candidate backend failed')
        result['frown_fer_status'] = 'measured'
        frown = module('_presence_frown',VENDOR/'reaction_detector/frowning/run_frowning_intensity_mt.py')
        cfg = _vendor_defaults(frown); result['parameters']['frowning'] = cfg
        m, mouth, brow, au = frown.load_blendshape_signals(pose)
        score = frown.compute_frowning_intensity(dict(enumerate(reply['sad'])),dict(enumerate(reply['angry'])),m,mouth,brow,au,**cfg)
        for e in intervals(score,detected,25,duration):
            if assess_frown(signal['brow_down'],score,e['start_frame'],e['end_frame_exclusive'],EX)['retain']:
                result['cues'].append(dict(e,type='frowning'))
    # Preserve smile evidence without inventing a new binary smile threshold.
    result['smile_blendshape'] = {'framewise_mean_left_right': smile_trace(raw)}
    save(folder/'candidates.json',result); return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--media',type=Path,required=True);p.add_argument('--fer-python',required=True)
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    measure(read(args.media),args.fer_python)


if __name__=='__main__':main()
