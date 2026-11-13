"""Explicit per-sample conditions; never substitute a global fixed text cache."""
from pathlib import Path
import torch

from p2_listen.cache import load_part, tree_to
from p2_listen.contracts import digest
from .protocol import open_protocol, sample, select
from .runtime import dependencies, identity, validate_payload


class PlanCacheReader:
    def __init__(self, work, *, verify_sources=True):
        self.work = Path(work)
        self.protocol = open_protocol(work, verify_sources=verify_sources)
        self.config = self.protocol['config']
        from p2_listen.config import activate
        activate(self.config)
        self._dependencies = {}
        self._native_text = None

    def part(self, sid, part, device='cpu'):
        item = sample(self.work, self.protocol, sid)
        if part not in self._dependencies:
            self._dependencies[part] = dependencies(self.config, part)
        expected = identity(self.protocol, item, part, self._dependencies[part])
        payload = load_part(self.work/'cache'/part/sid, expected, device)
        validate_payload(payload, item, part)
        return payload

    def conditions(self, sid, variant='A2', device='cpu'):
        from p2_listen.features import load_input, load_text
        from omni_av_p2.connector import ConditionInputs
        if variant not in ('A0', 'A1', 'A2'):
            raise ValueError('Expected A0, A1 or A2')
        item = sample(self.work, self.protocol, sid)
        if self._native_text is None:
            self._native_text = load_text(self.config)
        text = self._native_text if variant == 'A0' else dict(
            positive=self.part(sid, 'text')['positive'], negative=self._native_text['negative'])
        if variant == 'A2':
            planned = self.part(sid, 'input_plan')
            if planned['prompt'] != text['positive']['prompt']:
                raise ValueError('Qwen/Gemma plan text differs')
            x = ConditionInputs(**planned['condition']); layout = planned['layout']
        else:
            x, layout = load_input(self.config, item['observation'])
        return dict(condition=x.to(device), layout=tree_to(layout, device), text=tree_to(text, device),
                    metadata=dict(sample_id=sid, variant=variant,
                                  prompt_sha256=digest(text['positive']['prompt']),
                                  target_annotation_access=variant != 'A0', qwen_frozen=True))

    def training_sample(self, sid, target, variant='A2', device='cpu'):
        """Compose accepted AV/events without loading teacher or the old A2 input."""
        from listen_audio_postprocess.overlay import sample_contract, compose
        from p2_listen import features
        from p2_listen.config import activate
        activate(self.config)
        item = sample(self.work, self.protocol, sid); obs = item['observation']
        if target['sample_id'] != sid:
            raise ValueError('Target/sample mismatch')
        result = self.conditions(sid, variant, device)
        root, base_folder, expected, recipe, label, overlay_identity = sample_contract(self.config, obs, target)
        base = load_part(base_folder, expected, device)
        patch = load_part(root/'cache/audio_postprocess'/sid, overlay_identity, device)
        events = load_part(features.folder(self.config, 'events', sid),
                           features.identity(self.config, 'events', obs, label=label), device)
        result.update(av=compose(base, patch), events=events, teacher=None)
        return result


def verify(work, sample_ids=None, shard_index=0, shards=1, parts=('text', 'input_plan')):
    from p2_listen.contracts import atomic_json
    reader = PlanCacheReader(work); p = reader.protocol
    ids = select(p, sample_ids, shard_index, shards)
    if not parts or set(parts)-{'text', 'input_plan'}:
        raise ValueError('Invalid verification parts')
    rows = []
    for sid in ids:
        row = dict(sample_id=sid, success=True, parts={})
        for part in parts:
            try:
                value = reader.part(sid, part)
                row['parts'][part] = dict(success=True, prompt_sha256=value['prompt_sha256'])
                del value
            except Exception as exc:
                row['success'] = False
                row['parts'][part] = dict(success=False, error_type=type(exc).__name__, error=str(exc))
        if row['success'] and len({r['prompt_sha256'] for r in row['parts'].values()}) != 1:
            row['success'] = False
        rows.append(row)
    report = dict(success=all(r['success'] for r in rows), samples=len(ids), results=rows,
                  parts=list(parts), shard_index=shard_index, shards=shards, protocol_sha256=digest(p))
    path = Path(work)/'reports'/f'verify.{shard_index}-of-{shards}.{digest([ids,list(parts)])[:12]}.json'
    atomic_json(path, report)
    return report
