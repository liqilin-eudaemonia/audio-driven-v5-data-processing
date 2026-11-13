"""Independent post-run coverage, provenance, media and statistic audit (CPU only)."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    protocol = read(WORK / 'protocol.json')
    controller = read(WORK / 'CONTROLLER_PROTOCOL.json')
    extraction = read(WORK / 'EXTRACTION_COMPLETE.json')
    result = read(WORK / 'RESULTS.json')
    assert read(WORK / 'CONTROLLER_COMPLETE.json')['passed_execution']
    assert read(WORK / 'GENERATION_COMPLETE.json')['passed']
    assert extraction['passed'] and extraction['streams'] == 129
    assert read(WORK / 'COMPLETE.json')['results_sha256'] == digest(WORK / 'RESULTS.json')
    checksums = {}
    for group in [protocol['source_files'], protocol['assets'], controller['files'], controller['runtime_overlay']]:
        for path, expected in group.items():
            assert digest(path) == expected, path
            checksums[path] = expected
    assert digest(WORK / 'protocol.json') == controller['generation_protocol_sha256']
    assert digest(WORK / 'assets/spectre_model.tar') == controller['spectre_weight_sha256']
    assert digest(ROOT / 'experiments/p2_vico43_unified_20261107/COMPLETE.json') == protocol['source_run_sha256']
    dim_stage = read(ROOT / 'experiments/p2_vico_dim_best10_20261108/status.json')['stage']
    assert dim_stage == 'paused_by_user_for_31gpu_training'
    ids = [s['id'] for s in protocol['samples']]
    assert len(ids) == len(set(ids)) == 43
    videos = []
    for uid in ids:
        folder = WORK / 'generated' / uid
        receipt = read(folder / 'complete.json')
        video = folder / 'generated.mp4'
        assert digest(video) == receipt['sha256']
        assert receipt['real_user_visual'] and not receipt['listener_future_conditioned']
        assert receipt['avatar_audio_zero']
        assert receipt['protocol_sha256'] == digest(WORK / 'protocol.json')
        info = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=width,height,avg_frame_rate,nb_frames,duration', '-of', 'json', str(video)
        ], text=True))['streams'][0]
        assert (info['width'], info['height'], info['avg_frame_rate']) == (512, 512, '25/1')
        assert int(info['nb_frames']) == receipt['frames']
        # Decode every frame; metadata alone cannot establish file integrity.
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-threads', '2', '-i', str(video),
                        '-map', '0:v:0', '-f', 'null', '-'], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        videos.append(dict(id=uid, frames=receipt['frames'], decoded=True))
    arrays = {}
    feature_keys = set()
    for receipt in extraction['records']:
        key = receipt['id'], receipt['role']
        assert key not in feature_keys
        feature_keys.add(key)
        assert digest(receipt['path']) == receipt['sha256']
        assert receipt['protocol_sha256'] == digest(WORK / 'SPECTRE_PROTOCOL.json')
        with np.load(receipt['path']) as data:
            value = data['motion'].copy()
            assert int(data['fps']) == 25
            assert np.array_equal(data['frame_indices'], np.arange(receipt['frames']))
        assert value.shape == (receipt['frames'], 56) and np.isfinite(value).all()
        arrays[key] = value
    assert feature_keys == {(uid, role) for uid in ids for role in ['gt', 'speaker', 'avatarforcing']}
    for uid in ids:
        assert len({arrays[uid, role].shape for role in ['gt', 'speaker', 'avatarforcing']}) == 1
    independent = {}
    for group, sl in [('pose', slice(0, 6)), ('exp', slice(6, 56))]:
        g, p, x = [np.concatenate([arrays[uid, role][:, sl] for uid in ids]).astype(np.float64)
                   for role in ['gt', 'avatarforcing', 'speaker']]
        # Population variance via moments, independently from scorer np.var float32.
        variance = float(np.mean(np.mean(p * p, axis=0) - np.mean(p, axis=0) ** 2))
        assert np.isclose(variance, result['primary']['Var'][group], rtol=2e-5, atol=2e-6)

        def pearson(a, b):
            ac, bc = a.ravel() - a.mean(), b.ravel() - b.mean()
            return float(np.dot(ac, bc) / np.sqrt(np.dot(ac, ac) * np.dot(bc, bc)))

        residual = abs(pearson(g, x) - pearson(p, x))
        assert abs(residual - result['primary']['rPCC'][group]) < 1e-10
        independent[group] = dict(Var_moment_formula=variance, rPCC_centered_dot=residual)
    parity = read(WORK / 'SPECTRE_GPU_PARITY.json')
    assert parity['passed'] and parity['full_vs_context_chunk_max_abs'] < 1e-4
    assert parity['source_vs_independent_coefficient_max_abs'] == 0
    for cell in result['comparison']:
        assert cell['same_at_published_2dp'] == (f"{cell['measured']:.2f}" == f"{cell['published']:.2f}")
    audit = dict(passed=True, at=time.time(), videos=videos, video_count=43, feature_streams=129,
                 frames_per_role=sum(v['frames'] for v in videos), finite_full_coverage=True,
                 real_speaker_visual_all_43=True, listener_future_conditioned=False,
                 independent_statistics=independent, strict_gpu_encoder_parity=parity,
                 immutable_source_asset_files=len(checksums), old_run_marker_unchanged=True,
                 DIM_stage=dim_stage, matched_table_cells=result['matched_cells_at_published_precision'],
                 execution_pass_does_not_imply_paper_reproduction=True,
                 auditor_sha256=digest(__file__), results_sha256=digest(WORK / 'RESULTS.json'))
    destination = WORK / 'FINAL_AUDIT.json'
    destination.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'videos'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
