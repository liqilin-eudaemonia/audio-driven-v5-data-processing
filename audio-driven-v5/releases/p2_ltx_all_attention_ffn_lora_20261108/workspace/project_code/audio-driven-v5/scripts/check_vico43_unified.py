"""Check actual outputs and independently recompute errors, not only markers."""
import csv
import json
import os
from pathlib import Path
import sys
import time
import argparse

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_avatarforcing import WORK,read,save,sha,timestamp_map


def check_media():
    import cv2
    cv2.setNumThreads(1)
    af=WORK/'avatarforcing';p=read(af/'protocol.json');rows=[]
    for s in p['samples']:
        uid=s['id'];r=read(af/'generated'/uid/'complete.json')
        assert r['protocol_sha256']==sha(af/'protocol.json')
        native=af/'generated'/uid/'generated.mp4';score=Path(r['score_path']);decoded=[]
        for path,size,fps,count,expected in [(native,512,25,r['native_frames'],r['native_sha256']),
                                           (score,256,30,s['frames'],r['score_sha256'])]:
            assert sha(path)==expected
            cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path
            assert cap.get(cv2.CAP_PROP_FRAME_WIDTH)==cap.get(cv2.CAP_PROP_FRAME_HEIGHT)==size
            assert abs(cap.get(cv2.CAP_PROP_FPS)-fps)<1e-6
            n=0
            while True:
                ok,img=cap.read()
                if not ok:break
                assert img.shape==(size,size,3);n+=1
            cap.release();assert n==count,(uid,n,count);decoded.append(n)
        assert timestamp_map(s['frames'],r['native_frames']).tolist()==r['mapping']
        rows.append(dict(id=uid,decoded_native=decoded[0],decoded_score=decoded[1],
            native_sha256=r['native_sha256'],score_sha256=r['score_sha256']))
        print('decoded',len(rows),uid,flush=True)
    assert len(rows)==43 and sum(r['decoded_score'] for r in rows)==16288
    save(WORK/'MEDIA_AUDIT.json',dict(passed=True,pairs=43,videos_fully_decoded=86,
        protocol_sha256=sha(af/'protocol.json'),rows=rows,at=time.time()))


def check():
    import cv2
    cv2.setNumThreads(1)
    af=WORK/'avatarforcing';p=read(af/'protocol.json');result=read(WORK/'RESULTS.json')
    assert read(af/'WEIGHTS_LOADED.json')['wav2vec_exact_tensor_equality']
    assert read(af/'WEIGHTS_LOADED.json')['all_non_wav2vec_parameters_loaded']
    for f,h in {**p['assets'],**p['code'],**p['old_results']}.items():assert sha(f)==h,f
    ep=read(WORK/'evaluation_protocol.json')
    for f,h in {**ep['code_and_deca_assets'],**ep['bfm_assets']}.items():assert sha(f)==h,f
    for f,h in ep['fid_weights'].items():assert sha(f)==h,f
    assert sha(ep['arcface_path'])==ep['arcface_sha256']
    ids={s['id'] for s in p['samples']};assert len(ids)==43
    assert len({u.rsplit('_',2)[0] for u in ids})==10
    assert len({u.split('.')[0] for u in ids})==40
    rows={r['id']:r for r in result['deca']['models']['avatarforcing']['per_clip']}
    if not (WORK/'MEDIA_AUDIT.json').exists():check_media()
    media=read(WORK/'MEDIA_AUDIT.json')
    assert media['passed'] and media['protocol_sha256']==sha(af/'protocol.json')
    media_rows={r['id']:r for r in media['rows']};assert set(media_rows)==ids
    report=[];error=0.
    for s in p['samples']:
        uid=s['id'];r=read(af/'generated'/uid/'complete.json');f=read(af/'features'/(uid+'.json'))
        assert r['protocol_sha256']==sha(af/'protocol.json')
        assert r['source_only'] and not r['listener_gt_audio_read'] and not r['listener_future_video_read'] and r['user_visual_zero']
        assert sha(s['image'])==r['input_image_sha256']==s['image_sha256']
        from scipy.io import wavfile
        sr,wave=wavfile.read(s['speaker']);assert sr==16000
        import hashlib
        assert hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()==r['speaker_waveform_sha256']==s['speaker_waveform_sha256']
        native=af/'generated'/uid/'generated.mp4';score=Path(r['score_path'])
        assert sha(native)==r['native_sha256'] and sha(score)==r['score_sha256']==f['source_sha256']
        for path,h in f['outputs'].items():assert sha(path)==h
        prior=media_rows[uid]
        assert prior['native_sha256']==r['native_sha256'] and prior['score_sha256']==r['score_sha256']
        counts=[prior['decoded_native'],prior['decoded_score']]
        assert counts==[r['native_frames'],s['frames']]
        mapping=timestamp_map(s['frames'],r['native_frames'])
        assert mapping.tolist()==r['mapping']
        for space,dim in [('bfm',70),('deca',56)]:
            pred=np.load(af/'features'/space/(uid+'.npz'))['motion']
            assert pred.shape==(s['frames'],dim) and np.isfinite(pred).all()
        gtpath=ROOT/'experiments/p2_compare_vico43_20261106/learning2listen/features/gt'/(uid+'.npz')
        gt=np.load(gtpath)['motion'].astype(float);pred=np.load(af/'features/deca'/(uid+'.npz'))['motion'].astype(float)
        for g,sl in [('exp',slice(0,50)),('head_rotation',slice(50,53)),('jaw_rotation',slice(53,56))]:
            direct=float(np.sum(np.abs(gt[:,sl]-pred[:,sl]))/gt[:,sl].size)
            error=max(error,abs(direct-rows[uid]['metrics']['L1'][g]))
        report.append(dict(id=uid,decoded_native=counts[0],decoded_score=counts[1],missing_face_frames=f['missing_face_frames']))
    assert error<1e-12,error
    verified_outputs=0
    for name,model in result['deca']['models'].items():
        assert {r['id'] for r in model['per_clip']}==ids,name
        for row in model['per_clip']:
            uid=row['id']
            if name=='avatarforcing':path=af/'features/deca'/(uid+'.npz')
            elif name.startswith('learning2listen/'):
                path=ROOT/'experiments/p2_compare_vico43_20261106/learning2listen/generated'/name.split('/')[1]/(uid+'.npz')
            else:path=ROOT/'experiments/common_method_comparison_20261106/vico_deca/features'/name/(uid+'.npz')
            assert sha(path)==row['prediction_sha256'],path
            pred=np.load(path)['motion'].astype(float)
            gt=np.load(ROOT/'experiments/p2_compare_vico43_20261106/learning2listen/features/gt'/(uid+'.npz'))['motion'].astype(float)
            assert pred.shape==gt.shape==(row['frames'],56)
            assert np.isfinite(pred).all()
            for g,sl in [('exp',slice(0,50)),('head_rotation',slice(50,53)),('jaw_rotation',slice(53,56))]:
                error=max(error,abs(float(np.abs(pred[:,sl]-gt[:,sl]).mean())-row['metrics']['L1'][g]))
            verified_outputs+=1
        for metric,groups in model['coverage'].items():
            assert set(groups.values())=={42 if metric=='RTLCC' else 41 if metric=='RWTLCC' else 43}
        for metric,groups in model['metrics'].items():
            for g,v in groups.items():
                direct=np.mean([r['metrics'][metric][g] for r in model['per_clip'] if r['metrics'][metric][g] is not None])
                assert abs(direct-v)<1e-12
    # Independently check each paired point estimate and its stated denominator.
    joint={r['id']:r['metrics'] for r in result['deca']['models']['joint']['per_clip']}
    for name,metrics in result['paired_deca_difference_joint_minus_baseline'].items():
        baseline={r['id']:r['metrics'] for r in result['deca']['models'][name]['per_clip']}
        for m,groups in metrics.items():
            for g,v in groups.items():
                vals=[joint[u][m][g]-baseline[u][m][g] for u in ids if joint[u][m][g] is not None and baseline[u][m][g] is not None]
                assert len(vals)==v['clips'] and abs(np.mean(vals)-v['mean'])<1e-12
    for path,h in result['source_sha256'].items():assert sha(path)==h
    bfm_error=0.;bfm_outputs=0
    base=ROOT/'experiments/p2_vico_motion8000_20261106'
    for name,model in result['bfm'].items():
        assert {r['id'] for r in model['per_clip']}==ids
        for row in model['per_clip']:
            uid=row['id'];gt=np.load(base/'features/gt'/(uid+'.npz'))['motion'].astype(float)
            if name=='joint':source=base/'features/prediction'/(uid+'.npz')
            elif name=='dystream':source=ROOT/'experiments/p2_compare_vico43_20261106/dystream/motion/features/prediction'/(uid+'.npz')
            else:source=af/'features/bfm'/(uid+'.npz')
            if name!='avatarforcing':assert sha(source)==read(source.with_suffix('.json'))['feature_sha256']
            pred=np.load(source)['motion'].astype(float)
            assert pred.shape==gt.shape and pred.shape[1]==70 and np.isfinite(pred).all()
            for group,sl in [('exp',slice(0,64)),('angle',slice(64,67)),('trans',slice(67,70))]:
                bfm_error=max(bfm_error,abs(float(np.abs(pred[:,sl]-gt[:,sl]).mean())-row['metrics_raw']['FD'][group]))
            bfm_outputs+=1
    assert bfm_error<1e-12
    # Validate the image aggregation independently from the reporting builder.
    image=read(af/'IMAGE_RESULTS.json')
    counts={s['id']:s['frames'] for s in p['samples']}
    spec=read(ROOT/'experiments/p2_vico256_combined8000_20261106/protocol.json')['datasets']['vico']
    with open(spec['annotation']) as handle:annotation=list(csv.DictReader(handle))
    assert {r['index'] for r in image['cpbd']}==set(range(43))
    cpbd=sum(r['values']['cpbd']*counts[annotation[r['index']]['uuid']] for r in image['cpbd'])/16288
    assert abs(cpbd-image['metrics']['cpbd'])<1e-12
    for metric,value in image['metrics'].items():assert value==result['images']['avatarforcing'][metric]
    for record in image['reports']:
        for metric,value in record['values'].items():assert value==image['metrics'][metric]
    data=read(ROOT/'data_previews/listen_benchmark_review/vico43_unified_20261107/data.json')
    assert {c['id'] for c in data['cases']}==ids and data['count']==43
    output=dict(passed=True,at=time.time(),pairs=43,source_dialogues=10,speaker_intervals=40,
        native_and_score_videos_fully_decoded=86,score_frames=sum(r['decoded_score'] for r in report),
        independent_L1_max_error=error,all43_new_output_features_verified=True,all_motion_outputs_hash_verified=verified_outputs,
        independent_BFM_L1_max_error=bfm_error,bfm_video_outputs_verified=bfm_outputs,
        previous_results_unchanged=True,coverage_verified=True,pairwise_point_estimates_verified=True,
        source_only_code_path='frozen vico_avatarforcing.py: speaker + initial image + synthesized zero tensors; no GT file path passed to model',
        browser_check='separate BROWSER_AUDIT.json required',rows=report,results_sha256=sha(WORK/'RESULTS.json'))
    save(WORK/'NUMERICAL_AND_MEDIA_AUDIT.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--media-only',action='store_true');args=a.parse_args()
    check_media() if args.media_only else check()
