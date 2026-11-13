"""Verify the finalized Seamless metadata run without modifying admission rules."""
import collections
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data_corpora/seamless_listen_local_full'


def main():
    read = lambda name: json.loads((OUT / name).read_text())
    runtime = read('runtime.json')
    summary = read('screening_summary.json')
    face = read('face_size_distribution.json')
    crop = read('crop_size_distribution.json')
    index = read('manifests/index.json')
    assert runtime['state'] in ('finished', 'finished_with_errors')
    assert summary['final'] and summary['pending_pairs'] == 0
    assert summary['complete_pairs'] + summary['error_pairs'] == summary['total_pairs']
    assert summary['utc'] == face['utc'] == crop['utc'] == index['utc']
    assert index['export']['counts_verified'] and index['counts'] == summary['counts']
    assert not summary['visual_filter_applied']
    done = {p.stem for p in (OUT / 'results').glob('*.json')}
    errors = {p.stem for p in (OUT / 'errors').glob('*.json')} - done
    assert len(done) == summary['complete_pairs'] and len(errors) == summary['error_pairs']
    grouped = collections.Counter(read('errors/' + pair + '.json')['error'] for pair in errors)
    for key, pool in crop['pools'].items():
        for role, stats in pool['roles'].items():
            assert abs(sum(stats['side_seconds']) + stats['unmeasured_seconds'] - pool['seconds']) < 1e-5, (key, role)
            f = face['pools'][key]['roles'][role]
            assert abs(sum(f['median_short_side_seconds']) + f['unmeasured_seconds'] - pool['seconds']) < 1e-5, (key, role)
    for key in index['manifest_sha256']:
        assert (OUT / 'manifests' / (key + '.jsonl')).is_file()
    nodes = {}
    for path in (OUT / 'cluster/nodes').glob('*.json'):
        node = json.loads(path.read_text())
        assert node['state'] == 'finished'
        assert all(child['exit_code'] == 0 for child in node['children'].values())
        nodes[path.stem] = dict(state=node['state'], workers_exited=len(node['children']))
    assert len(nodes) == 4
    start = datetime.datetime.fromisoformat('2026-10-23T12:48:46+00:00')
    end = datetime.datetime.fromisoformat(runtime['utc'].replace('Z', '+00:00'))
    record = dict(start_utc=start.isoformat(), finish_utc=runtime['utc'],
                  elapsed_minutes=(end-start).total_seconds()/60,
                  total_pairs=summary['total_pairs'], complete_pairs=len(done),
                  error_pairs=len(errors), source_invalid_pairs=summary['source_invalid_pairs'],
                  pending_pairs=0, error_reasons=dict(grouped),
                  final_snapshot_utc=summary['utc'], export=index['export'],
                  duration_conservation_verified=True, manifest_files_present=True,
                  nodes=nodes, visual_filter_applied=False, training_started=False)
    (Path(__file__).parent / 'completion_verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
