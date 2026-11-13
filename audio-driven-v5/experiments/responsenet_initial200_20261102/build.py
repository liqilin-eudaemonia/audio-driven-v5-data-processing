"""Freeze 48 nonzero GT tracks + the best 152 zero-track pairs; reuse scored outputs."""
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RANKING = ROOT / 'experiments/responsenet_top200_analysis_20261102'
RUN = ROOT / 'experiments/p2_testsets_4ckpt_24gpu_20261102'
ARMS = ['av_event_6000', 'av_event_8927', 'av_only_6000', 'av_only_8927']


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in data))


def main():
    inputs = {}

    def tracked(path):
        inputs[str(path)] = sha(path)
        return path

    protocol = read(tracked(RUN / 'protocol.json'))
    refpath = Path(protocol['datasets']['responsenet']['prepared']) / 'references.jsonl'
    references = rows(tracked(refpath))
    assert sha(refpath) == protocol['datasets']['responsenet']['references_sha256']
    ref = {r['id']: r for r in references}
    ranking = rows(tracked(RANKING / 'ranked_all586.jsonl'))
    rank_method = read(tracked(RANKING / 'summary.json'))['method']
    audit_path = ROOT / 'experiments/planning_sft_responsenet_handoff_20261102/responsenet_gt_audio_audit.jsonl'
    audit = {r['id']: r for r in rows(tracked(audit_path))}
    assert len(ref) == len(references) == len(ranking) == 586
    assert set(ref) == {r['id'] for r in ranking} == set(audit)
    assert ranking == sorted(ranking, key=lambda r: (-r['score'], r['id']))
    assert all(r['gt_nonzero'] == audit[r['id']]['nonzero'] for r in ranking)
    nonzero = [r for r in ranking if r['gt_nonzero']]
    zero = [r for r in ranking if not r['gt_nonzero']]
    assert len(nonzero) == 48 and len(zero) == 538
    selected = nonzero + zero[:152]
    ids = [r['id'] for r in selected]
    idset = set(ids)
    assert len(idset) == 200
    zero_rank = {r['id']: n for n, r in enumerate(zero, 1)}
    selection = []
    for n, r in enumerate(selected, 1):
        selection.append({
            'selection_order': n, 'id': r['id'],
            'selection_reason': 'all_nonzero_gt_audio' if r['gt_nonzero'] else 'top152_zero_gt_audio',
            'global_visual_rank': r['rank'], 'zero_audio_visual_rank': zero_rank.get(r['id']),
            **{k: r[k] for k in ('source_id', 'source_start', 'source_end', 'duration_s',
                                'score', 'gt_nonzero', 'gt_acoustic', 'gt_speech')},
        })
    jsonl(BASE / 'references.jsonl', [ref[sid] for sid in ids])
    jsonl(BASE / 'selection.jsonl', selection)
    jsonl(BASE / 'excluded386.jsonl', [
        {'id': r['id'], 'global_visual_rank': r['rank'],
         'zero_audio_visual_rank': zero_rank[r['id']], 'reason': 'outside_top152_zero_gt_audio'}
        for r in zero[152:]
    ])
    with (BASE / 'selection.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(selection[0]))
        writer.writeheader()
        writer.writerows(selection)

    # Use the previous analysis' aggregation, already checked against the full-run results.
    spec = importlib.util.spec_from_file_location('existing_analysis', RANKING / 'analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracked(RANKING / 'analyze.py')
    metric_summaries = {}
    reuse = {}
    for arm in ARMS:
        map_path = RUN / 'maps' / arm / 'responsenet.jsonl'
        mapping_rows = rows(tracked(map_path))
        mapping = {r['id']: r for r in mapping_rows}
        metric_path = RUN / 'metrics' / arm / 'responsenet' / 'pairs.jsonl'
        metric_rows = rows(tracked(metric_path))
        metrics = {r['id']: r for r in metric_rows}
        audio_path = RUN / 'metrics' / arm / 'responsenet' / 'audio_pairs.jsonl'
        audio_rows = rows(tracked(audio_path))
        audio = {r['id']: r for r in audio_rows}
        for raw, keyed in ((mapping_rows, mapping), (metric_rows, metrics), (audio_rows, audio)):
            assert len(raw) == len(keyed) == 586 and set(keyed) == set(ref)
        for sid in ids:
            m = metrics[sid]
            assert not m['errors'] and not audio[sid]['errors']
            assert m['input']['gt'] == ref[sid]['gt']
            assert m['input']['generated'] == mapping[sid]['generated']
            assert m['input']['generated_audio'] == mapping[sid]['generated_audio']
            assert (audio[sid]['audio']['gt']['peak'] > 0) == audit[sid]['nonzero']
            for key in ('source_start', 'source_end'):
                assert abs(m['input'][key] - ref[sid][key]) < 1e-9
            for path in (mapping[sid]['generated'], mapping[sid]['generated_audio'],
                         ref[sid]['gt'], ref[sid]['listener']['audio']):
                assert Path(path).is_file(), path
        jsonl(BASE / 'maps' / f'{arm}.jsonl', [mapping[sid] for sid in ids])
        jsonl(BASE / 'metric_manifests' / f'{arm}.jsonl', [metrics[sid]['input'] for sid in ids])
        metric_summaries[arm] = module.aggregate([metrics[sid] for sid in ids])
        reuse[arm] = {'pairs': 200, 'original_visual_scores': str(metric_path),
                     'original_audio_scores': str(audio_path),
                     'generation_map': str(BASE / 'maps' / f'{arm}.jsonl'),
                     'bound_metric_manifest': str(BASE / 'metric_manifests' / f'{arm}.jsonl')}

    summary = {
        'purpose': 'Initial exploratory evaluation subset; original full586 default is unchanged.',
        'selection_rule': 'Keep all 48 exactly nonzero GT listener tracks; add the top 152 exactly-zero tracks by the existing four-checkpoint mean visual composite.',
        'ranking_method': rank_method,
        'pairs': 200, 'gt_nonzero': 48, 'gt_zero': 152,
        'source_count': len({r['source_id'] for r in selected}),
        'total_minutes': sum(r['duration_s'] for r in selected) / 60,
        'mean_duration_s': sum(r['duration_s'] for r in selected) / 200,
        'gt_acoustic_pairs': sum(r['gt_acoustic'] for r in selected),
        'gt_speech_pairs': sum(r['gt_speech'] for r in selected),
        'asr_groups': dict(Counter(r['asr_group'] for r in selected)),
        'overlap_with_previous_visual_top200': len(idset & {r['id'] for r in ranking[:200]}),
        'last_selected_zero_global_rank': zero[151]['rank'],
        'reused_generations': 800, 'new_generations': 0,
        'per_arm_per_pair_metric_aggregates': metric_summaries,
        'aggregation_note': 'Original frame/duration weighted aggregation. FID/FVD are not recomputed for this subset; no full586 distribution score is substituted. UTMOS remains unavailable.',
        'reuse': reuse,
        'caveats': ['Nonzero audio is not manually verified speech or backchannel.',
                    'The 152 zero-track pairs are selected by evaluated model performance; this is not an unbiased test set or the natural full586 distribution.'],
    }
    write(BASE / 'summary.json', summary)
    outputs = [BASE / name for name in ('references.jsonl', 'selection.jsonl', 'selection.csv',
                                        'excluded386.jsonl', 'summary.json')]
    outputs += list((BASE / 'maps').glob('*.jsonl')) + list((BASE / 'metric_manifests').glob('*.jsonl'))
    assert all(sha(Path(p)) == h for p, h in inputs.items())
    assert rows(BASE / 'references.jsonl') == [ref[sid] for sid in ids]
    assert {r['id'] for r in selected if r['gt_nonzero']} == {r['id'] for r in nonzero}
    assert {r['id'] for r in selected if not r['gt_nonzero']} == {r['id'] for r in zero[:152]}
    for arm in ARMS:
        for folder in ('maps', 'metric_manifests'):
            assert [r['id'] for r in rows(BASE / folder / f'{arm}.jsonl')] == ids
    write(BASE / 'seal.json', {'schema': 'responsenet-initial200-nonzero48-bestzero152-v1',
          'inputs_sha256': inputs, 'outputs_sha256': {str(p): sha(p) for p in outputs},
          'builder_sha256': sha(Path(__file__)), 'parent_references_sha256': sha(refpath)})
    write(BASE / 'validation.json', {'passed': True, 'unique_pairs': 200,
          'all48_nonzero_retained': True, 'exact_top152_zero_retained': True,
          'reference_rows_unchanged': True, 'four_arm_ids_aligned': True,
          '800_generated_video_and_audio_files_exist': True,
          'all_read_source_files_unchanged': True, 'new_gpu_work': False})
    print(json.dumps({k: v for k, v in summary.items() if k not in
                      ('ranking_method', 'reuse', 'per_arm_per_pair_metric_aggregates')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
