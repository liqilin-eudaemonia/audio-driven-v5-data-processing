"""Finite, real-video same-input arithmetic and CPU-pipeline speed check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read, save, sha, METRIC_PYTHON


def run(work, output):
    if 'js-infer-machine-ws3qe' not in os.uname().nodename:
        raise ValueError('Run on node11 only')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '7':
        raise ValueError('Physical GPU7 only')
    import numpy as np
    p = read(work/'protocol.json')
    arm = p['arms'][0]['name']
    output.mkdir(parents=True, exist_ok=True)
    report = dict(at=time.time(), trials={}, datasets={})
    for ds in ['seamless', 'responsenet']:
        src = work/'bindings'/arm/ds/'pairs.jsonl'
        rows = [json.loads(s) for s in src.read_text().splitlines() if s.strip()]
        rows.sort(key=lambda r: (sum(b-a for a, b in r['listening_intervals']), r['id']))
        chosen = [rows[round(i*(len(rows)-1)/3)] for i in range(4)]
        manifest = output/(ds+'.jsonl')
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in chosen))
        report['datasets'][ds] = dict(ids=[r['id'] for r in chosen], manifest_sha256=sha(manifest))
        order = ['baseline', 'pipeline'] if ds == 'seamless' else ['pipeline', 'baseline']
        for mode in order:
            dest = output/ds/mode
            common = ['--manifest', str(manifest), '--output', str(dest), '--assets', p['metric_assets'],
                      '--device', 'cuda:0', '--face-device', 'cuda:0', '--emotion-device', 'cuda:0',
                      '--threads', '4', '--pixel-workers', '4', '--batch-size', '64',
                      '--face-padding', str(p['datasets'][ds]['face_padding']),
                      '--gt-cache', p['metrics_gt_cache'], '--feature-output', str(dest/'features.npz'), '--allow-partial']
            if mode == 'pipeline':
                cli = [str(METRIC_PYTHON), '-u', str(ROOT/'scripts/p2_pipelined_video_metrics.py'), '--prefetch-depth', '2']
            elif ds == 'seamless':
                cli = [str(METRIC_PYTHON), '-u', str(ROOT/'scripts/p2_cached_video_metrics.py')]
            else:
                cli = [str(METRIC_PYTHON), '-u', '-m', 'evaluation.video.paired_metrics.run']
            if ds == 'seamless':
                cli += ['--decoded-gt-cache', p['metrics_decoded_gt_cache']]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='7', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='2',
                       PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python')
            start = time.monotonic()
            save(output/'STATUS.json', dict(stage='running', dataset=ds, mode=mode, at=time.time()))
            with (output/(ds+'_'+mode+'.log')).open('w') as log:
                result = subprocess.run(cli+common, cwd=ROOT.parent/'OmniResponse', env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Benchmark {ds}/{mode} failed; see log')
            pair_rows = [json.loads(s) for s in (dest/'pairs.jsonl').read_text().splitlines()]
            report['trials'][ds+'/'+mode] = dict(wall_seconds=time.monotonic()-start,
                   pair_seconds=sum(r['seconds'] for r in pair_rows), pairs=len(pair_rows))
            save(output/'IN_PROGRESS.json', report)
        a = output/ds/'baseline'
        b = output/ds/'pipeline'
        ar = [json.loads(s) for s in (a/'pairs.jsonl').read_text().splitlines()]
        br = [json.loads(s) for s in (b/'pairs.jsonl').read_text().splitlines()]
        assert len(ar) == len(br) == 4
        keys = ['id', 'metrics', 'errors', 'pixels', 'fid', 'fvd', 'face_detection', 'emotion', 'response']
        for x, y in zip(ar, br):
            for k in keys:
                if x.get(k) != y.get(k):
                    raise AssertionError(f'Native value differs: {ds}/{x["id"]}/{k}')
        fa = np.load(a/'features.npz')
        fb = np.load(b/'features.npz')
        assert fa.files == fb.files
        for k in fa.files:
            np.testing.assert_array_equal(fa[k], fb[k], err_msg=f'{ds}/{k}')
        assert read(a/'summary.json')['metrics'] == read(b/'summary.json')['metrics']
        assert read(a/'summary.json')['coverage'] == read(b/'summary.json')['coverage']
        assert read(a/'contract.json')['code'] == read(b/'contract.json')['code']
        assert read(a/'contract.json')['assets'] == read(b/'contract.json')['assets']
        assert all(row['gt_cache_hits'] == other['gt_cache_hits'] for row, other in zip(ar, br))
        native = report['trials'][ds+'/baseline']
        piped = report['trials'][ds+'/pipeline']
        cpu = read(b/'cpu_pipeline.json')
        # Count preparation that overlaps backend initialization, rather than
        # hiding it behind per-row wall times.
        report['datasets'][ds].update(exact_pair_metrics=True, exact_features=True,
                    exact_summary=True, gt_feature_cache_preserved=True,
                    speedup_wall=native['wall_seconds']/piped['wall_seconds'],
                    speedup_pair_loop=native['pair_seconds']/piped['pair_seconds'],
                    producer_decode_seconds=sum(x.get('decode_cpu_seconds', 0) for x in cpu['events']),
                    producer_pixel_seconds=sum(x.get('pixels_cpu_seconds', 0) for x in cpu['events']),
                    input_wait_seconds=sum(x['wait_seconds'] for x in cpu['events']))
    report.update(success=True, finished_at=time.time(), physical_gpu=7,
                  note='Finite eight-video one-pass comparison; opposite mode order across datasets. Not a global optimum or whole400 speed estimate.')
    save(output/'RESULTS.json', report)
    save(output/'STATUS.json', dict(stage='complete', success=True, at=time.time()))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.work.resolve(), args.output.resolve())
    except Exception as exc:
        save(args.output/'FAILURE.json', dict(error=repr(exc), at=time.time()))
        raise
