"""Export response episodes while retaining the complete semantic observation layer."""
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
from listen_benchmark.response_episodes import VERSION, consolidate_responses


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def export(source, output, preview=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    targets = [output] + ([Path(preview).resolve()] if preview else [])
    for i, target in enumerate(targets):
        for other in [source] + targets[:i]:
            if target == other or target in other.parents or other in target.parents:
                raise ValueError('Use separate export directories')
        if target.exists() and any(target.iterdir()):
            raise ValueError('Use an empty output directory: ' + str(target))
    raw = (source / 'labels.jsonl').read_bytes()
    observations = [json.loads(line) for line in raw.splitlines() if line.strip()]
    old_review = json.loads((source / 'review.json').read_text())
    if not old_review['policy'].get('semantic') or old_review['policy'].get('episodes'):
        raise ValueError('Input must be semantic observations, not an already consolidated export')
    media = {s['id']: s for s in old_review['samples']}
    ids = [s['sample_id'] for s in observations]
    if len(ids) != len(set(ids)) or set(ids) != set(media):
        raise ValueError('Sample identity mismatch')
    results, traces, samples = [], {}, []
    for original in observations:
        result, trace = consolidate_responses(original)
        results.append(result)
        traces[result['sample_id']] = trace
        sample = dict(media[result['sample_id']])
        sample['events'] = [e for e in result['events'] if e['mapping_status'] == 'mapped_candidate']
        sample['unresolved_observations'] = result['unresolved_observations']
        samples.append(sample)
    events = [e for s in results for e in s['events']]
    status = Counter(e['mapping_status'] for e in events)
    counts = Counter(k for e in events if e['mapping_status'] == 'mapped_candidate' for k in e['categories'])
    pending = Counter(k for e in events if e['mapping_status'] == 'needs_review' for k in e['categories'])
    episodes = [e for e in events if e.get('event_unit') == 'vocal_response_episode']
    taxonomy = old_review['policy']['taxonomy']
    summary = dict(run_id=output.name, source_run=source.name, samples=len(results),
        open_events=sum(len(s['events']) for s in observations), derived_events=len(events),
        status_counts=dict(status), mapped_by_class={k: counts[k] for k in taxonomy},
        review_candidates_by_class={k: pending[k] for k in taxonomy},
        audio_observations_before=old_review['summary']['mapped_by_class']['listener_vocalization'],
        audio_response_episodes_after=len(episodes), merged_episode_count=sum(len(e['member_event_ids']) > 1 for e in episodes),
        unresolved_grouping_observations=sum(len(s['unresolved_observations']) for s in results),
        word_observations_preserved=sum(e.get('event_unit') == 'word' for s in observations for e in s['events']),
        visual_events_unchanged=True, all_original_observations_preserved=True,
        media_inference_rerun=False, human_ground_truth=False, training_admitted=False, K_train=[])
    policy = dict(old_review['policy'], version=VERSION, episodes=True,
        main_labels_preserve_all_open_events=False, observations_file='observations.jsonl',
        changes_timestamps=True, merges_occurrences=True,
        category_mapping_unchanged=True, raw_observation_times_unchanged=True,
        episode_rule='Same literal content; ASR intervals overlap/touch, loose model intervals overlap. No gap threshold.',
        boundary_rule='ASR component extent when present, otherwise union of overlapping model intervals.',
        separation_rule='A loose interval cannot bridge separated ASR components; ambiguous observations are retained separately.',
        visual_rule='Visual events unchanged; simultaneous gestures do not split audio.',
        repetition_rule='Do not infer word repetitions from observation count. Internal word records are alignment evidence only.',
        limitations='Episode grouping is a candidate derived from existing intervals; acoustic pauses and true word repetition have not been revalidated.')
    review = dict(summary=summary, policy=policy, samples=samples)
    html = (ROOT / 'scripts/listen_seven_class_review.html').read_text().replace('__DATA__', json.dumps(review, ensure_ascii=False).replace('</', '<\\/'))
    candidates = [dict(s, events=[e for e in s['events'] if e['mapping_status'] == 'mapped_candidate']) for s in results]
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        (target / 'observations.jsonl').write_bytes(raw)
        for name, rows in [('labels.jsonl', results), ('category_candidates.jsonl', candidates)]:
            (target / name).write_text(''.join(json.dumps(s, ensure_ascii=False, allow_nan=False) + '\n' for s in rows))
        dump(target / 'labels.json', dict(schema='listen-response-episode-dataset-v1', samples=results))
        dump(target / 'provenance.json', dict(version=VERSION, samples=traces,
            source_provenance_file='source_provenance.json', raw_observations_file='observations.jsonl'))
        shutil.copy2(source / 'provenance.json', target / 'source_provenance.json')
        for name, value in [('summary.json', summary), ('mapping_policy.json', policy), ('review.json', review)]:
            dump(target / name, value)
        (target / 'index.html').write_text(html)
        if (source / 'review.woff').exists():
            shutil.copy2(source / 'review.woff', target / 'review.woff')
    code_paths = ['src/listen_benchmark/response_episodes.py', 'scripts/build_listen_response_episodes.py',
                  'scripts/listen_seven_class_review.html', 'tests/test_listen_response_episodes.py',
                  'scripts/check_listen_response_episodes.py']
    for rel in code_paths:
        dest = output / 'code' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    dump(output / 'manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(), source_run=str(source),
         input_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                       for name in ['labels.jsonl', 'provenance.json', 'review.json']},
         code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code_paths}))
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Completed semantic label export directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--preview')
    args = parser.parse_args()
    export(args.source, args.output, args.preview)
