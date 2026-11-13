"""Independent coverage, provenance and numerical audit; no success by markers alone."""
import math
from pathlib import Path
import subprocess
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
PARENT = WORK.parent / 'p2_avatarforcing_table4_full_reproduction_20261110'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'compare/deployment')]
from p2_avatarforcing_table4_generate import read, save, sha


def main():
    import cv2
    import numpy as np
    import soundfile as sf
    from sklearn.cluster import KMeans
    from vico_literature_metrics import load_author
    cv2.setNumThreads(1)
    p = read(WORK/'NATIVE_GENERATION_PROTOCOL.json')
    required = read(WORK/'REQUIREMENTS.json')
    result = read(WORK/'OFFICIAL_RESULTS.json')
    extraction = read(WORK/'OFFICIAL_EXTRACTION_COMPLETE.json')
    assert len(p['samples']) == 43 and extraction['streams'] == 129
    assert p['target'] == required['target']
    assert result['generation_protocol_sha256'] == sha(WORK/'NATIVE_GENERATION_PROTOCOL.json')
    assert result['extraction_protocol_sha256'] == sha(WORK/'OFFICIAL_SPECTRE_PROTOCOL.json')
    assert result['metrics_protocol_sha256'] == sha(WORK/'OFFICIAL_METRICS_PROTOCOL.json')
    for path, digest in {**p['assets'], **p['code']}.items():
        assert sha(path) == digest, path
    frontend = read(WORK/'OFFICIAL_SPECTRE_PROTOCOL.json')
    assert sha(WORK/'official_spectre.py') == frontend['script_sha256']
    spectre_asset = ROOT/'experiments/p2_avatarforcing_table4_recheck_20261110/assets/spectre_model.tar'
    assert sha(spectre_asset) == frontend['encoder']['checkpoint_sha256']
    for path, digest in frontend['encoder']['source'].items():
        assert sha(path) == digest
    for name in ['SOURCE_MANIFEST.json','WEIGHT_MANIFEST.json']:
        for record in read(WORK/'official_tracker_audit'/name):
            assert sha(ROOT.parents[1]/record['path']) == record['sha256']
    parallel_path = WORK/'OFFICIAL_PARALLEL_PROTOCOL.json'
    if parallel_path.exists():
        parallel = read(parallel_path)
        assert sha(WORK/'official_spectre_workers.py') == parallel['script_sha256']
        assert sha(WORK/'official_spectre.py') == parallel['source_helper_sha256']
        parity = read(WORK/'official_tracker_audit/PARALLEL_PARITY.json')
        assert parity['passed'] and parity['exact_motion_bytes']
        assert parity['motion_max_abs'] <= 1e-6 and parity['landmarks_max_abs'] <= 1e-6
    for name in ['GENERATION_MATH_CALL_AUDIT.json','FRONTEND_SOURCE_REUSE_AUDIT.json']:
        assert read(WORK/name)['passed']
    assert read(WORK/'ORIGINAL_INPUTS_INDEPENDENT_AUDIT.json')['passed']
    previous = WORK.parent/'p2_avatarforcing_table4_framegrid_20261110_v2'
    assert (WORK/'official_spectre.py').read_bytes() == (previous/'official_spectre.py').read_bytes()
    old_frontend = read(previous/'OFFICIAL_SPECTRE_PROTOCOL.json')
    assert {k:v for k,v in frontend.items() if k!='generation_protocol_sha256'} == {k:v for k,v in old_frontend.items() if k!='generation_protocol_sha256'}
    reused = read(WORK/'REUSED_GT_SPEAKER_FEATURES.json')
    assert reused['streams'] == len(reused['records']) == 86
    assert read(previous/'FINAL_AUDIT.json')['passed_execution_audit']
    for item in reused['records']:
        receipt = read(Path(item['path']).with_suffix('.json'))
        assert receipt['reused_identical_video_features']
        assert sha(item['path']) == item['sha256'] == receipt['sha256']
        assert sha(receipt['parent_feature_receipt']) == receipt['parent_feature_receipt_sha256']
        parent = read(receipt['parent_feature_receipt'])
        assert parent['protocol_sha256'] == receipt['parent_feature_protocol_sha256'] == sha(previous/'OFFICIAL_SPECTRE_PROTOCOL.json')
        for key in ['id','role','video_sha256','frames','sha256','grid','face_audit']:
            assert parent[key] == receipt[key], (key,item['id'],item['role'])
        assert Path(parent['video']).resolve() == Path(receipt['video']).resolve()
    assert len({(r['id'],r['role']) for r in reused['records']}) == 86
    recs = {(r['id'],r['role']): r for r in extraction['records']}
    assert len(recs) == 129
    arrays = {}
    decoded, rng_cpu, rng_cuda, tails = [], set(), set(), []
    for sample in p['samples']:
        uid = sample['id']
        audio = sf.info(sample['speaker'])
        assert audio.samplerate == 16000
        frames = math.ceil(audio.frames/640)
        assert (frames-1)/25 < audio.frames/16000
        generation = read(WORK/'native_generated'/uid/'complete.json')
        video = WORK/'native_generated'/uid/'generated.mp4'
        assert generation['frames'] == frames and sha(video) == generation['sha256']
        assert generation['protocol_sha256'] == sha(WORK/'NATIVE_GENERATION_PROTOCOL.json')
        assert generation['future_listener_pixels_conditioned'] is False
        assert generation['reference_ROI_from_full_track'] is True
        reference = p['mean_reference_images'][uid]
        assert sha(reference['path']) == reference['sha256'] == generation['reference_sha256']
        runtime = WORK/'runtime_history'/(generation['runtime_sha256']+'.json')
        assert sha(runtime) == generation['runtime_sha256']
        rng_cpu.add(generation['initial_cpu_rng'])
        rng_cuda.add(generation['initial_cuda_rng'])
        if generation.get('reused_without_rerender'):
            assert sha(generation['parent_receipt']) == generation['parent_receipt_sha256']
            parent = read(generation['parent_receipt'])
            assert parent['sha256'] == generation['sha256'] and parent['frames'] == frames
        cap = cv2.VideoCapture(str(video))
        assert cap.get(cv2.CAP_PROP_FPS) == 25
        count = 0
        while True:
            okay, frame = cap.read()
            if not okay: break
            assert frame.shape == (512,512,3)
            count += 1
        cap.release()
        assert count == frames, (uid,count,frames)
        decoded.append(dict(id=uid,frames=count,sha256=generation['sha256']))
        for role in ['speaker','gt','avatarforcing']:
            r = recs[uid,role]
            assert r['frames'] == frames and sha(r['path']) == r['sha256']
            assert sha(r['video']) == r['video_sha256']
            assert r['protocol_sha256'] == sha(WORK/'OFFICIAL_SPECTRE_PROTOCOL.json')
            if 'executor_protocol_sha256' in r:
                assert r['executor_protocol_sha256'] == sha(parallel_path)
            with np.load(r['path']) as data:
                motion = data['motion'].copy()
                assert motion.shape == (frames,56) and np.isfinite(motion).all()
                assert np.array_equal(data['frame_indices'],np.arange(frames))
                assert float(data['fps']) == 25
                assert data['landmarks'].shape == (frames,68,2) and np.isfinite(data['landmarks']).all()
            grid = r['grid']
            expected = np.floor(np.arange(frames)*grid['source_fps']/25+.5).astype(int).clip(0,grid['source_frames']-1)
            assert np.array_equal(grid['source_indices'],expected)
            if role != 'avatarforcing' and grid['source_frames'] < frames:
                assert frames-grid['source_frames'] == 1
                tails.append(dict(id=uid,role=role,source_frames=grid['source_frames'],
                    evaluation_frames=frames,last_timestamp=(frames-1)/25,actual_audio_duration=audio.frames/16000))
            arrays[uid,role] = motion
        print('Audit',len(decoded),'/43',uid,flush=True)
    assert len(rng_cpu) == len(rng_cuda) == 1
    assert len(tails) == 4  # Two fractional-duration clips, both GT and speaker.
    ids = [s['id'] for s in p['samples']]
    author = load_author()
    independently_computed = {m:{} for m in ['FD','rPCC','SID','Var']}
    for group, sl, k in [('exp',slice(6,56),15),('pose',slice(0,6),9)]:
        gs = [arrays[u,'gt'][:,sl] for u in ids]
        ps = [arrays[u,'avatarforcing'][:,sl] for u in ids]
        xs = [arrays[u,'speaker'][:,sl] for u in ids]
        g,a,x = [np.concatenate(values) for values in [gs,ps,xs]]
        fd = [author.calculate_frechet_distance(*author.calculate_activation_statistics(gg),
                    *author.calculate_activation_statistics(aa)) for gg,aa in zip(gs,ps)]
        independently_computed['FD'][group] = float(np.mean(fd))
        def correlation(left,right):
            l = left.ravel().astype(np.float64);r = right.ravel().astype(np.float64)
            l -= l.mean();r -= r.mean()
            return float(np.dot(l,r)/math.sqrt(float(np.dot(l,l)*np.dot(r,r))))
        independently_computed['rPCC'][group] = abs(correlation(g,x)-correlation(a,x))
        independent_labels = KMeans(n_clusters=k,random_state=0,n_init='auto').fit(g).predict(a)
        hist = np.bincount(independent_labels,minlength=k).astype(np.float64)
        hist /= hist.sum()
        independently_computed['SID'][group] = float(-np.dot(hist,np.log2(hist+1e-6)))
        independently_computed['Var'][group] = float(np.var(a,axis=0).mean())
    gold = read(PARENT/'BASELINE_PROVENANCE.json')
    assert gold['other_rows_source_audited'] and gold['inherited_cell_count'] == 24
    assert sha(ROOT/'writing/43_Avatar_Forcing_CVPR2026.pdf') == gold['af_paper_sha256']
    assert len(gold['table']) == 5 and all(len(row)==8 for row in gold['table'].values())
    order = [('FD','exp'),('FD','pose'),('rPCC','exp'),('rPCC','pose'),('SID','exp'),('SID','pose'),('Var','exp'),('Var','pose')]
    cells = []
    for metric, group in order:
        value = independently_computed[metric][group]
        assert abs(value-result['primary'][metric][group]) < 1e-6
        target = gold['table']['Ours'][metric+'_'+group]
        assert target == required['target'][metric][group]
        cells.append(dict(metric=metric,group=group,independent_value=value,published=target,
                          same_at_2dp=f'{value:.2f}'==f'{target:.2f}'))
    matches = sum(c['same_at_2dp'] for c in cells)
    assert matches == result['matched_cells_at_published_precision']
    save(WORK/'FINAL_AUDIT.json',dict(passed_execution_audit=True,decoded_videos=43,
        decoded_frames=sum(r['frames'] for r in decoded),coefficient_streams=129,
        independent_metrics=independently_computed,cells=cells,matched_cells=matches,total_cells=8,
        numerical_table_match=matches==8,goal_complete_proven=matches==8,
        private_author_configuration_confirmed=False,remaining_gaps=result['known_gaps'],
        fractional_tail_checks=tails,video_audit=decoded,
        code_sha256=sha(__file__),results_sha256=sha(WORK/'OFFICIAL_RESULTS.json'),
        baseline_provenance_sha256=sha(PARENT/'BASELINE_PROVENANCE.json')))
    print('Independent audit:',matches,'/8 published cells match',flush=True)
    # This complete published-code candidate was frozen before full results,
    # and is reported separately rather than selecting per-cell alternatives.
    subprocess.run([sys.executable,str(WORK/'score_dim_style.py'),'score'],cwd=ROOT,check=True)


if __name__ == '__main__':
    main()
