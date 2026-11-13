"""Export seven-class candidates from frozen open labels; no media inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.seven_class import TAXONOMY, map_sample, policy


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def export(run, output, preview=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    preview = Path(preview).resolve() if preview else None
    targets = [output] + ([preview] if preview else [])
    for target in targets:
        if target == run or run in target.parents or target in run.parents:
            raise ValueError('Output must be separate from frozen input run')
        if target.exists() and any(target.iterdir()):
            raise ValueError(f'Use a new empty output directory: {target}')
    if preview and (output == preview or output in preview.parents or preview in output.parents):
        raise ValueError('Run output and preview must be separate directories')
    source_names = ['labels.jsonl', 'provenance.json', 'input_review.json']
    raw = {name: (run / name).read_bytes() for name in source_names}
    labels = [json.loads(line) for line in raw['labels.jsonl'].decode().splitlines() if line.strip()]
    provenance = json.loads(raw['provenance.json'])
    media = json.loads(raw['input_review.json'])
    ids = [s['sample_id'] for s in labels]
    media_by_id = {s['id']: s for s in media['samples']}
    if (not ids or len(ids) != len(set(ids)) or len(media_by_id) != len(media['samples'])
            or set(ids) != set(provenance['samples']) or set(ids) != set(media_by_id)):
        raise ValueError('Sample identity mismatch between labels, provenance and media metadata')
    results = [map_sample(s, provenance['samples'][s['sample_id']]) for s in labels]
    statuses, counts, pending, reasons = Counter(), Counter(), Counter(), Counter()
    per_class_samples = {k: 0 for k in TAXONOMY}
    for r in results:
        counts.update(e['type'] for e in r['events'])
        for cls in {e['type'] for e in r['events']}:
            per_class_samples[cls] += 1
        for d in r['decisions']:
            statuses[d['status']] += 1
            reasons.update(d['reasons'])
            if d['status'] == 'needs_review':
                pending.update(d['candidate_types'])
    summary = dict(run_id=output.name, source_run=run.name, samples=len(results),
                   open_events=sum(len(r['decisions']) for r in results), status_counts=dict(statuses),
                   mapped_by_class={k: counts[k] for k in TAXONOMY},
                   clips_by_class=per_class_samples,
                   review_candidates_by_class={k: pending[k] for k in TAXONOMY},
                   review_reasons=dict(reasons),
                   word_anchors_preserved=sum(e['event_unit'] == 'word' for r in results for e in r['events']),
                   upstream_exclusions=sum(len(r['upstream_exclusions']) for r in results),
                   taxonomy=list(TAXONOMY), human_ground_truth=False, training_admitted=False,
                   K_train=[], negative_coverage_inferred=False, gpu_inference_rerun=False)
    exports = [{k: v for k, v in r.items() if k not in ('decisions', 'upstream_exclusions', 'upstream_audit')}
               for r in results]
    trace = {'run_id': output.name, 'policy': policy(), 'samples': {
        r['sample_id']: {k: r[k] for k in ('decisions', 'upstream_exclusions', 'upstream_audit')}
        for r in results}}
    samples = []
    for r in results:
        original = media_by_id[r['sample_id']]
        if original['duration'] != r['duration_s']:
            raise ValueError('Media/label duration mismatch')
        samples.append(dict(id=r['sample_id'], duration=r['duration_s'],
                            video=original['video'], audio=original['audio'],
                            events=r['events'], decisions=r['decisions']))
    review = dict(summary=summary, policy=policy(), samples=samples)
    template = (ROOT / 'scripts/listen_seven_class_review.html').read_text()
    html = template.replace('__DATA__', json.dumps(review, ensure_ascii=False).replace('</', '<\\/'))
    # All validation precedes writes. Keep immutable inputs for independent replay.
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        (target / 'labels.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in exports))
        dump(target / 'labels.json', dict(schema='listen-seven-class-dataset-v1', samples=exports))
        dump(target / 'provenance.json', trace)
        dump(target / 'summary.json', summary)
        dump(target / 'mapping_policy.json', policy())
        dump(target / 'review.json', review)
        (target / 'index.html').write_text(html)
    (output / 'input').mkdir()
    for name, content in raw.items():
        (output / 'input' / name).write_bytes(content)
    code_paths = ['src/listen_benchmark/seven_class.py', 'scripts/build_listen_seven_class.py',
                  'scripts/listen_seven_class_review.html', 'tests/test_listen_seven_class.py']
    for rel in code_paths:
        target = output / 'code' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), source_run=str(run),
                    input_sha256={k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
                    code_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in code_paths})
    dump(output / 'manifest.json', manifest)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', help='Frozen open-merge input for explicitly requested legacy rules')
    p.add_argument('--semantic-run', help='Completed model semantic-classification run (current workflow)')
    p.add_argument('--baseline', help='Optional old rule result for comparison')
    p.add_argument('--legacy-rules', action='store_true', help='Explicitly reproduce old keyword-rule mapping')
    p.add_argument('--output', required=True, help='New seven-class result directory')
    p.add_argument('--preview', help='Optional new preview directory, next to open_merge/')
    a = p.parse_args()
    if a.semantic_run:
        if a.legacy_rules or a.run:p.error('Do not mix semantic and legacy-rule inputs')
        from build_listen_semantic_seven_class import export_semantic
        export_semantic(a.semantic_run,a.output,a.preview,a.baseline)
    elif a.legacy_rules and a.run:
        export(a.run, a.output, a.preview)
    else:
        p.error('Provide --semantic-run; old rules require explicit --legacy-rules --run')


if __name__ == '__main__':
    main()
