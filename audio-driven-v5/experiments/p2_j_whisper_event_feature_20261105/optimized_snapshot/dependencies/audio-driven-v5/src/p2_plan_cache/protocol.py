"""Stream cleaned annotations into an immutable, independently stored protocol."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from p2_listen.contracts import ROOT, atomic_json, digest, observation, path_at, read, safe_id, sha
from p2_listen.config import load as load_config, feature_loss_enabled
from .formatting import render_verbatim
from . import VERSION


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def code_identity():
    paths = list(Path(__file__).parent.glob('*.py')) + [
        ROOT/'scripts/p2_plan_cache.py', ROOT/'src/p2_guided_prompt_en/formatting.py',
        ROOT/'src/p2_guided_prompt_en/encoding.py', ROOT/'src/p3_listen/prompts.py']
    return {str(p): sha(p) for p in sorted(paths)}


def render_clean(record):
    value = render_verbatim(record)
    # Model control strings are not literal listener speech or visual descriptions.
    if any(token in value['prompt'] for token in ('<|', '|>')):
        raise ValueError('Model control tokens in guidance are forbidden')
    return value


def prepare(config, work, sample_ids=None, max_text_tokens=8192, max_qwen_tokens=32768):
    if max_text_tokens not in (1024, 2048, 4096, 8192):
        raise ValueError('Gemma limit must be 1024, 2048, 4096 or 8192')
    if type(max_qwen_tokens) is not int or max_qwen_tokens <= 0:
        raise ValueError('Qwen limit must be positive')
    config, work = Path(config).resolve(), Path(work).resolve()
    c = load_config(config); root = path_at(c['output'])
    if work == root or work.is_relative_to(root):
        raise ValueError('Use a separate work directory; the accepted dataset is immutable')
    if feature_loss_enabled(c) or c['data']['size'] != 512 or 'audio_postprocess' not in c['data']:
        raise ValueError('Expected 512px postprocessed data with feature supervision disabled')
    completion = read(root/'COMPLETE.json')
    if completion['status'] != 'passed':
        raise ValueError('Dataset has not passed acceptance')
    source_paths = [config, root/'COMPLETE.json', root/'manifests/observations.json',
                    root/'integrated_labels.jsonl', root/'dataset.jsonl']
    sources = {str(p): sha(p) for p in source_paths}
    for p in source_paths:
        name = str(p.relative_to(root)) if p.is_relative_to(root) else None
        if name in completion.get('files', {}) and sources[str(p)] != completion['files'][name]:
            raise ValueError('Accepted dataset file changed: '+str(p))
    rows = read(root/'manifests/observations.json')
    raw = {safe_id(o['sample_id']): o for o in rows}
    if len(raw) != len(rows) or len(raw) != completion['samples']:
        raise ValueError('Duplicate/missing dataset observations')
    selected = list(raw) if sample_ids is None else list(sample_ids)
    if not selected or len(set(selected)) != len(selected) or set(selected)-raw.keys():
        raise ValueError('Invalid selected sample IDs')
    selected = sorted(selected)
    observations = {}
    for sid in selected:
        o = observation(raw[sid], root/'manifests', verify=False)
        if o['experiment_split'] != 'train' or o['official_split'] != 'train':
            raise ValueError('This entry point prepares training annotations only')
        if o['geometry']['height'] != 512 or o['geometry']['width'] != 512:
            raise ValueError('Expected 512x512 geometry')
        observations[sid] = o
    contract = dict(schema=VERSION, config=c, sources=sources, code=code_identity(),
                    sample_ids=selected, max_text_tokens=max_text_tokens,
                    max_qwen_tokens=max_qwen_tokens, plan_source='postprocessed_train_annotation',
                    annotation_text_policy='verbatim_including_non_english',
                    target_annotation_access=True, target_media_access=False,
                    qwen_frozen=True, qwen_lora=False, truncation=False)
    with lock(work.parent/(work.name+'.prepare.lock')):
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work/'prepare_identity.json', contract, immutable=True)
        entries = {}; seen = set(); errors = []
        # The source is >1GB: never materialize all annotation records at once.
        with (root/'integrated_labels.jsonl').open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line); sid = record['sample_id']
                if sid not in observations:
                    continue
                if sid in seen:
                    raise ValueError('Duplicate selected annotation: '+sid)
                seen.add(sid); obs = observations[sid]
                if abs(record['duration_s']-obs['requested_duration_s']) > 1e-6:
                    raise ValueError('Annotation/observation duration mismatch')
                try:
                    guidance = render_clean(record)
                except ValueError as exc:
                    errors.append(dict(sample_id=sid, error=str(exc)))
                    continue
                item = dict(observation=obs, guidance=guidance)
                atomic_json(work/'samples'/(sid+'.json'), item, immutable=True)
                entries[sid] = digest(item)
                if len(entries)%1000==0:
                    print(json.dumps(dict(stage='prepare',ready=len(entries),total=len(selected))),flush=True)
        if seen != set(selected):
            raise ValueError('Missing selected annotations: '+str(sorted(set(selected)-seen)))
        if errors:
            atomic_json(work/'preparation_errors.json', dict(success=False, samples=len(selected),
                        failed_samples=len(errors), errors=errors,
                        policy='no_translation_no_silent_drop_no_protocol_publication'))
            raise ValueError(f'{len(errors)} invalid guidance samples; see {work}/preparation_errors.json')
        # Bind each selected source record to the dataset's accepted export.
        indexed = set()
        with (root/'dataset.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line); sid = row['sample_id']
                if sid not in entries:
                    continue
                if sid in indexed:
                    raise ValueError('Duplicate dataset index entry')
                indexed.add(sid)
                label = root/'labels/canonical'/(sid+'.json')
                recipe = root/'audio_postprocess/recipes'/(sid+'.json')
                if sha(label) != row['canonical_sha256'] or sha(recipe) != row['recipe_sha256']:
                    raise ValueError('Current label/recipe differs from accepted export: '+sid)
        if indexed != set(selected):
            raise ValueError('Selected samples missing from accepted export')
        if any(sha(p) != h for p, h in sources.items()):
            raise ValueError('Dataset changed during preparation')
        protocol = dict(contract, entries=entries)
        atomic_json(work/'protocol.json', protocol, immutable=True)
    return dict(success=True, samples=len(entries), work=str(work), protocol_sha256=digest(protocol))


def open_protocol(work, verify_sources=True):
    p = read(Path(work)/'protocol.json')
    if p['schema'] != VERSION or p['code'] != code_identity():
        raise ValueError('Plan cache code changed; prepare a new work directory')
    if verify_sources:
        for path, h in p['sources'].items():
            if sha(path) != h:
                raise ValueError('Frozen dataset source changed: '+path)
    return p


def sample(work, p, sid):
    safe_id(sid)
    if sid not in p['entries']:
        raise ValueError('Sample outside prepared protocol')
    item = read(Path(work)/'samples'/(sid+'.json'))
    if digest(item) != p['entries'][sid]:
        raise ValueError('Frozen sample changed')
    if item['observation']['sample_id'] != sid or item['guidance']['sample_id'] != sid:
        raise ValueError('Sample identity mismatch')
    if render_clean(item['guidance']['source_record']) != item['guidance']:
        raise ValueError('Prompt differs from the existing English formatter')
    return item


def select(p, sample_ids=None, shard_index=0, shards=1):
    if type(shards) is not int or type(shard_index) is not int or not 0 <= shard_index < shards:
        raise ValueError('Invalid shard')
    ids = list(p['sample_ids'] if sample_ids is None else sample_ids)
    if len(ids) != len(set(ids)) or set(ids)-p['entries'].keys():
        raise ValueError('Invalid selected IDs')
    return sorted(ids)[shard_index::shards]
