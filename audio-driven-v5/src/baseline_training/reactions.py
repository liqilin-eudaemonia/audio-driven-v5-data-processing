"""Training-only unknown supervision mask; never alter benchmark detection."""
import csv
from pathlib import Path
import pickle
import shutil

import numpy as np

from .data import save


def extract(video, output, policy):
    from p2_glare_eval.detector import detect, run_rules, valid_face
    output = Path(output)
    try:
        result = detect(video, output, policy)
        count = result['geometry']['frame_count']
        mask = np.ones((count, 6), dtype=bool)
    except ValueError as error:
        if not str(error).startswith('Incomplete independent detection:'):
            raise
        import json
        quality = json.loads((output / 'quality.json').read_text())
        count = quality['frames']
        with (output / 'face.pkl').open('rb') as f:
            face = pickle.load(f)
        face_complete = all(valid_face(x) for x in face)
        fer_complete = quality['fer_valid_frames'] == count and quality['jointly_valid_frames'] == count
        # Original nod/shake rules consume face motion; FER classes consume
        # emotion scores. If a modality fails anywhere, mark its entire clip
        # unknown instead of manufacturing temporal labels by interpolation.
        mask = np.tile([fer_complete] * 4 + [face_complete] * 2, (count, 1))
        original = output / 'fer_original.csv'
        if not original.exists():
            shutil.copyfile(output / 'fer.csv', original)
        with original.open() as f:
            reader = csv.DictReader(f)
            names = reader.fieldnames
            rows = list(reader)
        # The FER-based classes are wholly masked if incomplete. Filling blanks
        # only lets unchanged six-rule code run; these filled targets never
        # contribute to the loss. Nod/shake inputs are the untouched face.pkl.
        for row in rows:
            for key in names[1:]:
                if not row[key]:
                    row[key] = '0'
        with (output / 'fer.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        run_rules(Path(video), output, policy, count)
    np.save(output / 'training_valid_mask.npy', mask, allow_pickle=False)
    save(output / 'TRAINING_LABELS.json', dict(frames=count, known_per_class=mask.sum(0).tolist(),
          policy='incomplete FER -> all four FER classes unknown; incomplete face -> nod/shake unknown',
          benchmark_detector_policy_unchanged=True, interpolated_targets_used=False))
