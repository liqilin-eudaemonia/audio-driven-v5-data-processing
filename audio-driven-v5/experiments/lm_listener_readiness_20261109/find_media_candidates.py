"""Find text-anchor candidates in existing RealTalk ASR; never certify media."""
import collections
import csv
import json
from pathlib import Path
import re
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
CORPUS = Path('/public/0_DATA/2_Avatar/realtalk/asr')


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def grams(ts, width=8):
    return {tuple(ts[i:i + width]) for i in range(len(ts) - width + 1)}


rows = torch.load(ROOT / 'evaluation/lm-listener/dataset/trevorconanstephen/segments_test_no_overlap.pth',
                  map_location='cpu', weights_only=False)
anchors = collections.defaultdict(set)
for row in rows:
    anchors[row['fname']].update(grams(tokens(' '.join(w['text'] for w in row['during_words']))))
reverse = collections.defaultdict(set)
for fname, phrases in anchors.items():
    for phrase in phrases:
        reverse[phrase].add(fname)
scores = collections.defaultdict(collections.Counter)
files = sorted(CORPUS.glob('*.json'))
for path in files:
    datum = json.loads(path.read_text())
    text = datum.get('text', '')
    for phrase in grams(tokens(text)):
        for fname in reverse.get(phrase, ()):
            scores[fname][path.stem] += 1
matches = []
for fname in sorted(anchors):
    candidates = [{'realtalk_video_id': vid, 'distinct_shared_8word_anchors': n}
                  for vid, n in scores[fname].most_common(3)]
    matches.append({'source_fname': fname, 'reference_distinct_anchors': len(anchors[fname]),
                    'candidates': candidates, 'verified_media_mapping': False})
report = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'searched_corpus': str(CORPUS), 'asr_files_checked': len(files),
          'reference_videos': len(anchors), 'videos_with_any_candidate': sum(bool(m['candidates']) for m in matches),
          'method': 'Exact 8-word alphanumeric anchors. Candidate discovery only; roles/time/crops unverified.',
          'matches': matches}
(WORK / 'MEDIA_CANDIDATES.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
with (WORK / 'source_videos.csv').open('w') as f:
    w = csv.writer(f)
    w.writerow(['source_fname', 'reference_distinct_8word_anchors', 'best_realtalk_candidate', 'shared_anchors', 'verified'])
    for m in matches:
        first = m['candidates'][0] if m['candidates'] else {}
        w.writerow([m['source_fname'], m['reference_distinct_anchors'], first.get('realtalk_video_id', ''),
                    first.get('distinct_shared_8word_anchors', 0), False])
print(json.dumps({k: v for k, v in report.items() if k != 'matches'}))
