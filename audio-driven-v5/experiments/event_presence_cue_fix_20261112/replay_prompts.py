"""CPU replay of real cached cues into v2 prompts; does not generate event labels."""
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from p2_event_presence.hints import audio_hints, smile_trace, visual_hints
from p2_event_presence.prompts import VERSION, prompt
from p2_event_presence.scoring import score
from p2_event_presence.media import source_identity
from run_listen_native_caption_vocal_pilot import normalize_asr_cue_for_english_prompt


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'experiments/p2_event_presence_pilot_20261108/run'
    output = Path(__file__).resolve().parent
    # Freeze the artifacts read below; never write into the historical run.
    paths = [source/'event_pairs.jsonl', source/'summary.json', source/'frozen_pairs.jsonl']
    paths += sorted((source/'media').glob('*/*.json'))
    paths += sorted((source/'media').glob('*/pose.pkl'))
    before = {str(p.relative_to(source)): sha(p) for p in paths}
    report = []
    for p in sorted((source/'media').glob('*/media.json')):
        media = read(p)
        candidates = read(p.parent/'candidates.json')
        # Trusted local cache from the previous pipeline; no detector rerun.
        with (p.parent/'pose.pkl').open('rb') as handle:
            poses = pickle.load(handle)
        candidates['framewise_face_detected'] = [bool(x.get('detected')) for x in poses]
        candidates['smile_blendshape'] = dict(framewise_mean_left_right=smile_trace(poses))
        dest = output/'prompt_replay'/p.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        for wi, window in enumerate(media['windows']):
            hints = visual_hints(candidates, window)
            text = prompt('visual', window['end_s']-window['start_s'], hints)
            assert 'smile_blendshape' in text
            assert hints['coverage_scope'] == 'current_window'
            target = dest/f'visual_{wi:03d}_prompt.txt'
            target.write_text(text+'\n')
            report.append(dict(media=p.parent.name, window=wi, modality='visual',
                               face_coverage=hints['frame_coverage'],
                               smile_bins=len(hints['smile_blendshape']['bins']),
                               prompt_characters=len(text), prompt_sha256=sha(target)))
            asr_path = p.parent/f'audio_{wi:03d}_asr_raw.json'
            if asr_path.exists():
                attempts = read(asr_path)['attempts']
                raw = json.loads(next(a['text'] for a in reversed(attempts) if a['finish_reason'] == 'stop'))
                hints = audio_hints(raw, normalize_asr_cue_for_english_prompt)
                assert not {'transcript', 'nonlexical_syllables', 'raw_asr_text'} & hints.keys()
                text = prompt('audio', window['end_s']-window['start_s'], asr=hints)
                target = dest/f'audio_{wi:03d}_prompt.txt'
                target.write_text(text+'\n')
                report.append(dict(media=p.parent.name, window=wi, modality='audio',
                                   prompt_characters=len(text), prompt_sha256=sha(target)))
    records = [json.loads(line) for line in (source/'event_pairs.jsonl').read_text().splitlines()]
    expected = [json.loads(line) for line in (source/'frozen_pairs.jsonl').read_text().splitlines()]
    rescored, old = score(records, expected), read(source/'summary.json')
    for dataset, data in rescored['datasets'].items():
        for method, result in data['methods'].items():
            previous = old['datasets'][dataset]['methods'][method]
            for field in ('draws', 'common_subset_draws', 'full_dataset_draw_aggregate', 'common_subset_draw_aggregate'):
                assert result[field] == previous[field], (dataset, method, field)
    assert before == {str(p.relative_to(source)): sha(p) for p in paths}
    identity = source_identity()
    assert 'src/p2_event_presence/hints.py' in identity
    result = dict(prompt_version=VERSION, source_files=identity,
                  historical_artifacts_unchanged=True, historical_scores_reproduced=True,
                  media=len({r['media'] for r in report}),
                  visual_prompts=sum(r['modality']=='visual' for r in report),
                  audio_prompts=sum(r['modality']=='audio' for r in report), prompts=report,
                  gpu_inference_run=False, new_event_labels_generated=False,
                  semantic_accuracy_improvement_verified=False)
    (output/'PROMPT_REPLAY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_files','prompts')}))


if __name__ == '__main__':
    main()
