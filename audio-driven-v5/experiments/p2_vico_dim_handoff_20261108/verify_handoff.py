"""Read-only verification of the sealed DIM/EMOCA run and GT-scope diagnostics.

Writes only into this handoff folder; never changes the sealed run or allocates CUDA.
"""
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEALED = ROOT / 'experiments/p2_vico_literature_protocol_20261108'
sys.path.insert(0, str(ROOT/'compare/deployment'))
sys.path.insert(0, str(ROOT/'scripts'))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    import numpy as np
    from vico_literature_report import verify_existing
    protocol = read(SEALED/'protocol.json')
    complete = read(SEALED/'COMPLETE.json')
    for path, expected in complete['files'].items():
        assert sha(path) == expected, path
    implementation = read(SEALED/'IMPLEMENTATION_FILES.json')['files']
    for path, expected in implementation.items():
        assert sha(path) == expected, path
    for path, expected in protocol['source_files'].items():
        assert sha(path) == expected, path
    encoder = read(SEALED/'ENCODER.json')
    assert sha(encoder['checkpoint']) == encoder['checkpoint_sha256']
    assert sha(encoder['config']) == encoder['config_sha256']
    assert encoder['full_encoder_state_strict']
    assert sum(encoder['encoder_tensors'].values()) == 644
    meta = Path('/public/0_DATA/2_Avatar/ViCo/dataset/metadata.csv')
    assert sha(meta) == protocol['metadata_sha256']
    with meta.open() as stream:
        test_ids = [row['uuid'] for row in csv.DictReader(stream) if row['data_split'] == 'test']
    assert len(test_ids) == len(set(test_ids)) == 43
    assert set(test_ids) == set(protocol['ids'])
    tasks = {(r['id'],r['role']):r for r in protocol['tasks']}
    assert len(tasks) == 172
    for task in tasks.values():
        assert sha(task['video']) == task['video_sha256'], task['video']
    features = {}
    for receipt in read(SEALED/'EXTRACTION_COMPLETE.json')['records']:
        key = receipt['id'],receipt['role']
        assert key in tasks and key not in features
        assert sha(receipt['path']) == receipt['sha256']
        with np.load(receipt['path']) as data:
            a = data['motion'].copy()
            assert a.dtype == np.float32 and a.shape == (tasks[key]['expected_frames'],56)
            assert np.array_equal(data['frame_indices'], np.arange(len(a)))
            valid = np.isfinite(a).all(1)
            assert np.array_equal(valid, data['valid'])
            assert int(valid.sum()) == receipt['valid_frames']
            assert np.where(~valid)[0].tolist() == receipt['missing_frames']
        features[key] = a
    assert set(features) == set(tasks)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        verify_existing()
    recompute = output.getvalue().strip()
    assert '548 scalars; max_abs= 0.0' in recompute
    result = read(SEALED/'RESULTS.json')
    ids = protocol['ids']
    scopes = {
        'full43_all_gt_frames': [features[u,'gt'] for u in ids],
        'full43_listener_t1_onward_no_generation_mask': [features[u,'gt'][1:] for u in ids],
        'actual_shared_valid_mask': [features[c['id'],'gt'][c['listener_frame_indices']] for c in result['coverage']],
    }
    diagnostic = read(SEALED/'LOADER_SCOPE_DIAGNOSTIC.json')
    scopes['released_loader_video_only_projection'] = [features[u,'gt'][1:] for u in diagnostic['ids_with_repetitions']]
    rows = {}
    for name, arrays in scopes.items():
        a = np.concatenate(arrays)
        assert np.isfinite(a).all()
        rows[name] = dict(rows=len(arrays),frames=len(a),
            Var_exp=float(np.var(a[:,6:].reshape(-1))), Var_pose=float(np.var(a[:,:6].reshape(-1))))
    for group in ['exp','pose']:
        assert rows['actual_shared_valid_mask']['Var_'+group] == result['gt_metrics']['Var'][group]
    write('GT_SCOPE_DIAGNOSTICS.json', dict(
        scopes=rows, published_dim_gt_Var=dict(exp=.93,pose=.01),
        exact_author_scope_known=False,
        interpretation='GT variance mismatch persists on all GT frames, without generation failures. '
            'The video-only loader projection omits unknown author audio/feature lengths and is diagnostic only.',
        no_scaling_or_protocol_tuning=True))
    exit_receipt = read(SEALED/'jobs/emoca_extract/exit.json')
    assert exit_receipt['returncode'] == 0
    checks = dict(passed=True,verified_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        engineering_complete=True, strict_published_table_equivalence=False,
        sealed_files_verified=len(complete['files']), implementation_files_verified=len(implementation),
        original_source_files_verified=len(protocol['source_files']), input_videos_verified=len(tasks),
        feature_arrays_verified=len(features),encoder_tensors_strictly_loaded=644,
        named_model=encoder['model'], checkpoint_sha256=encoder['checkpoint_sha256'],
        official_test43_ids_verified=True, methods=protocol['selected_methods'],
        author_metric_parity=read(SEALED/'EMOCA_METRIC_PARITY.json'),
        full_metric_recomputation_stdout=recompute, coverage=read(SEALED/'COVERAGE.json'),
        extraction_process_successful_exit=exit_receipt,
        result_sha256=sha(SEALED/'RESULTS.json'), source_completion_sha256=sha(SEALED/'COMPLETE.json'),
        new_gpu_execution=False, pending='Author confirmation of published protocol; see DOUBLE_CHECK.md')
    write('VERIFICATION.json',checks)
    print(json.dumps({k:v for k,v in checks.items() if k not in ['author_metric_parity','coverage']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
