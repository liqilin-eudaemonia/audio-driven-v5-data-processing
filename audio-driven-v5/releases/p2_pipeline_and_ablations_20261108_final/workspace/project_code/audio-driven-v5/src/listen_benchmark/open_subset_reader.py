"""Read all original event descriptions; never substitute seven-class event targets."""
import hashlib
import json
from pathlib import Path
import sqlite3


def load_open_label(root,sample_id):
    root=Path(root)
    with sqlite3.connect(f'file:{root}/open_label_index.sqlite?mode=ro',uri=True) as db:
        row=db.execute('SELECT offset,size,sha256 FROM labels WHERE sample_id=?',(sample_id,)).fetchone()
    if row is None:raise KeyError(sample_id)
    with (root/'integrated_labels.jsonl').open('rb') as f:f.seek(row[0]);raw=f.read(row[1])
    if hashlib.sha256(raw).hexdigest()!=row[2]:raise ValueError('Open label changed')
    label=json.loads(raw)
    if label['sample_id']!=sample_id or label['schema']!='listen-open-all-verbatim-v1' or label['new_semantic_classification']:
        raise ValueError('Unexpected open-label contract')
    return label


def load_open_training_sample(root,observation,target,device='cpu'):
    """Return input, layout, None, AV latent target, and OPEN annotation dict.

The last result is not the stock seven-class event tensor. Consumers must accept
raw descriptions and intervals explicitly; the existing fixed head is not changed.
"""
    from p2_listen.config import load,activate
    from p2_listen import features
    from p2_listen.cache import load_part
    from listen_audio_postprocess.overlay import sample_contract,compose
    root=Path(root);cfg=load(root/'media_cache_config.json');activate(cfg)
    _,base_folder,expected,recipe,seven_reference,identity=sample_contract(cfg,observation,target)
    x,layout=features.load_input(cfg,observation,device)
    base=load_part(base_folder,expected,device)
    patch=load_part(root/'cache/audio_postprocess'/observation['sample_id'],identity,device)
    return x,layout,None,compose(base,patch),load_open_label(root,observation['sample_id'])
