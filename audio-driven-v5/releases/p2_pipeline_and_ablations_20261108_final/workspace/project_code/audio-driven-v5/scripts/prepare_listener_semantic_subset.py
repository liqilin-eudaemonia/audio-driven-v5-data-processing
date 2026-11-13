#!/usr/bin/env python3
"""Freeze postprocessed subset membership into semantic runner jobs (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.semantic_prompt import source_payload


def prepare(subset, output):
    subset = Path(subset).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    targets = json.loads((subset / 'manifests/targets.json').read_text())
    observations = json.loads((subset / 'manifests/observations.json').read_text())
    ids = [row['sample_id'] for row in targets]
    if len(set(ids)) != len(ids) or set(ids) != {r['sample_id'] for r in observations}:
        raise ValueError('Duplicate or inconsistent subset membership')
    expected, seen = set(ids), set()
    source = subset / 'integrated_labels.jsonl'
    sha = hashlib.sha256()
    job_sha = hashlib.sha256()
    tmp = output / 'inputs.jsonl.partial'
    with source.open('rb') as src, tmp.open('wb') as dst:
        for raw in src:
            sha.update(raw)
            if not raw.strip():
                continue
            record = json.loads(raw)
            sid = record['sample_id']
            if sid not in expected or sid in seen:
                raise ValueError('Unexpected or duplicate label: ' + sid)
            if 'audio_postprocess' not in record:
                raise ValueError('Missing audio postprocessing provenance: ' + sid)
            source_payload(record)
            job_id = 'current__' + sid
            if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in job_id):
                raise ValueError('Unsafe job ID')
            job = dict(job_id=job_id, record=record, provenance=dict(
                role='current_postprocessed', source_file=str(source),
                source_line_sha256=hashlib.sha256(raw).hexdigest()))
            encoded = (json.dumps(job, ensure_ascii=False) + '\n').encode()
            dst.write(encoded)
            job_sha.update(encoded)
            seen.add(sid)
    if seen != expected:
        raise ValueError(f'Missing labels: {len(expected - seen)}')
    tmp.replace(output / 'inputs.jsonl')
    report = dict(status='prepared', samples=len(seen), subset=str(subset),
                  source_sha256=sha.hexdigest(), inputs_file_sha256=job_sha.hexdigest(),
                  manifest_sha256={name: hashlib.sha256((subset / 'manifests' / name).read_bytes()).hexdigest()
                                   for name in ('targets.json', 'observations.json')},
                  inference_started=False, original_records_unchanged=True)
    (output / 'PREPARED.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.subset, args.output), ensure_ascii=False, indent=2))
