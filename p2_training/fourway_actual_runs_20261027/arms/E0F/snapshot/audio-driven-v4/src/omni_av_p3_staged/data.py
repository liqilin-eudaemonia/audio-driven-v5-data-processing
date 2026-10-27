"""No P2 hidden reuse: join P3 native targets with text/AV by exact split and ID."""
import hashlib
import json
import math
from pathlib import Path
import torch
from omni_av.data import TextPlan
from omni_av.ltx_runtime import load_av_targets, audit_target_window
from omni_av_p3.distributed import sha
from .integrity import load_record, verified_load


def read(path):
    return json.loads(Path(path).read_text())


def unique(rows, key, split):
    result = {}
    for row in rows:
        sid = key(row)
        if row['split'] != split or sid in result:
            raise ValueError('Duplicate ID or split mismatch')
        if not sid or Path(sid).name != sid or sid in ('.', '..'):
            raise ValueError('Unsafe sample ID')
        result[sid] = row
    return result


def join_rows(native, text, examples, metadata, split, expected):
    groups = [unique(native, lambda x:x['sample_id'], split),
              unique(text, lambda x:x['sample_id'], split),
              unique(examples, lambda x:x['observation']['sample_id'], split),
              unique(metadata, lambda x:x['sample_id'], split)]
    ids = set(groups[0])
    if len(ids) != expected or ids != set(groups[1]) or ids != set(groups[2]) or not ids <= set(groups[3]):
        raise ValueError('Full split count/ID coverage mismatch')
    result = []
    for sid in sorted(ids):
        n, t, e, m = [g[sid] for g in groups]
        h, w = m['target_height'], m['target_width']
        if (h, w) != (e['target_height'], e['target_width']) or h % 32 or w % 32:
            raise ValueError('Original geometry mismatch; no resizing/cropping allowed')
        if not math.isclose(n['duration_s'], e['duration_s'], abs_tol=1e-4):
            raise ValueError('Native target duration mismatch')
        audit_target_window(dict(example=e), dict(metadata=m, tiny=False))
        result.append(dict(native=n, text=t['text'], example=e, metadata=m,
                           sample_id=sid, split=split))
    return result


def condition_seed(seed, sid):
    return int.from_bytes(hashlib.sha256(f'{seed}:{sid}'.encode()).digest()[:4], 'big')


def condition_mode(seed, ordinal):
    # Exactly one Oracle and one Pred25 per pair; deterministic across rank/resume.
    pair = ordinal // 2
    flip = condition_seed(seed, str(pair)) % 2
    return 'pred25' if (ordinal % 2) ^ flip else 'oracle'


def cache_identity(manifest_sha, entry):
    return dict(manifest_sha256=manifest_sha, sample_id=entry['sample_id'], split=entry['split'])


def cache_path(root, entry):
    sid = entry['sample_id']
    return Path(root) / entry['split'] / sid[:3] / (sid + '.pt')


def load_readouts(root, entry, manifest_sha):
    path = cache_path(root, entry)
    marker = read(path.with_suffix('.json'))
    expected = cache_identity(manifest_sha, entry)
    if marker['identity'] != expected or marker['sha256'] != sha(path):
        raise ValueError('Readout cache identity/hash mismatch')
    row = torch.load(path, map_location='cpu', weights_only=True)
    if row['identity'] != expected or set(row['readouts']) != {'oracle', 'pred25'}:
        raise ValueError('Missing Oracle/Pred25 readouts')
    return row['readouts']


def load_training_sample(entry, root, manifest_sha, data_root, mode, device, allow_unsealed=False):
    record=load_record(Path(root)/'inputs',entry,manifest_sha,data_root,allow_unsealed)
    readouts = load_readouts(root, entry, manifest_sha)
    hidden = {k:v.to(device) for k,v in readouts[mode].items()}
    text = verified_load(entry['text'],record['files'],map_location=device)
    if text['prompt'] != TextPlan(**entry['example']['text_plan']).render_prompt():
        raise ValueError('Cached text plan differs')
    av, meta = load_av_targets(data_root, entry['sample_id'], device=device, split=entry['split'],
                              tensor_loader=lambda p:verified_load(p,record['files']))
    if meta['metadata'] != entry['metadata']:
        raise ValueError('VAE metadata changed after preparation')
    audit_target_window(entry, meta)
    return hidden, text, av
