"""Read-only validation of the completed native pilot; independent count arithmetic."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from p2_event_presence.contracts import CLASSES, sha
from p2_event_presence.media import verify
from p2_event_presence.inference import request

WORK = Path(__file__).parent / 'run'


def load(path):
    return json.loads(path.read_text())


def run():
    verify(WORK)
    records = [json.loads(line) for line in (WORK/'event_pairs.jsonl').read_text().splitlines()]
    prepared = [json.loads(line) for line in (WORK/'prepared_pairs.jsonl').read_text().splitlines()]
    summary = load(WORK/'summary.json')
    complete = load(WORK/'DETECTION_COMPLETE.json')
    assert len(records) == len(prepared) == 8
    assert sum(not r['self_check_only'] for r in records) == 4
    unique = {m['media_id']: m for r in prepared for m in r['prepared_media'].values()}
    assert len(unique) == complete['media'] == 8
    unknown = []
    for mid, media in unique.items():
        folder = Path(media['folder'])
        assert not media['errors']
        assert sha(folder/'detection.json') == complete['detections'][mid]
        d = load(folder/'detection.json')
        assert not d['errors']
        assert set(d['labels']) == set(CLASSES)
        for mod, classes in CLASSES.items():
            assert set(d['labels'][mod]) == set(classes)
            for c, label in d['labels'][mod].items():
                assert label['state'] in {'present', 'absent', 'unknown'}
                assert label['evidence']
                if label['state'] == 'unknown':
                    unknown.append(dict(media_id=mid, modality=mod, event=c, evidence=label['evidence']))
        for field, digestfield in [('visual25','visual25_sha256'), ('audio','audio_file_sha256')]:
            assert sha(media[field]) == media[digestfield]
        assert load(folder/'candidates.json')['frame_coverage'] == 1.0
        for window in media['windows']:
            assert sha(window['frames']) == window['frames_sha256']
        for raw in folder.glob('*raw.json'):
            assert load(raw)['attempts'][-1]['finish_reason'] == 'stop'
    checked_counts = 0
    for dataset, data in summary['datasets'].items():
        for method, result in data['methods'].items():
            source = [r for r in records if r['dataset'] == dataset and r['method'] == method]
            for mod, classes in CLASSES.items():
                draw = result['draws'][0][mod]
                any_unknown = False
                f1s = []
                for c in classes:
                    gt = [r['reference'][mod][c]['state'] for r in source]
                    pred = [r['generated'][mod][c]['state'] for r in source]
                    pairs = [(g,p) for g,p in zip(gt,pred) if g in {'present','absent'} and p in {'present','absent'}]
                    any_unknown |= len(pairs) != len(source)
                    counts = dict(tp=sum(g=='present' and p=='present' for g,p in pairs),
                        fp=sum(g=='absent' and p=='present' for g,p in pairs),
                        fn=sum(g=='present' and p=='absent' for g,p in pairs),
                        tn=sum(g=='absent' and p=='absent' for g,p in pairs))
                    measured = draw['per_class'][c]
                    assert all(measured[k] == v for k,v in counts.items())
                    assert measured['evaluated_pairs'] == len(pairs)
                    checked_counts += 1
                    if c in data['reference_supported_classes'][mod]:
                        tp, fp, fn = (counts[k] for k in ('tp','fp','fn'))
                        f1s.append(2*tp/(2*tp+fp+fn))
                    else:
                        assert not measured['macro_eligible']
                        assert measured['false_positive_rate'] is not None
                assert (draw['full_dataset_metrics'] is None) == any_unknown
                if f1s:
                    assert abs(draw['available_subset_metrics']['macro_f1']-sum(f1s)/len(f1s)) < 1e-12
                if method == 'gt_self_control':
                    assert all(r['reference'] == r['generated'] for r in source)
                    assert all(v == 1 for v in f1s)
    # Validate the exact runtime request constructor without loading a second model.
    class Processor:
        class Image:
            temporal_patch_size = 2
        image_processor = Image()
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)
    for mod, kind in [('visual','video'), ('audio','audio')]:
        req = request(Processor(),mod,object(),'blind instruction','opaque')
        assert set(req['multi_modal_data']) == {kind}
        assert req['mm_processor_kwargs']['use_audio_in_video'] is False
    for r in prepared:
        assert r['audio_audit']['generated_audio']['gt_event_gate_applied'] is False
        assert r['audio_audit']['generated_audio']['own_full_waveform'] is True
    browser = load(ROOT/'data_previews/listen_benchmark_review/event_presence_pilot_20261108/BROWSER_CHECK.json')
    assert browser['passed'] and browser['video_players'] == browser['audio_players'] == 8
    assert summary['pilot_only'] and not summary['formal_benchmark_ready']
    assert summary['human_reliability_validation'] == 'pending'
    result = dict(passed=True, bound_pairs=8, actual_model_pairs=4, gt_self_controls=4,
        unique_media=8, technical_failures=0, explicit_class_units=56, unknown_units=unknown,
        independently_checked_class_count_tables=checked_counts,
        separate_modality_request_check=True, generated_gt_gate_applied=False,
        source_and_media_hashes_checked=True, browser_check_passed=True,
        summary_sha256=sha(WORK/'summary.json'), formal_benchmark_ready=False,
        human_reliability_validation='pending')
    (WORK.parent/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    run()
