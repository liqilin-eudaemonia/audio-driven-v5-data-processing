"""Caption-cache identity, completion status, and prefetch shutdown order.

Detection measurements are recorded, not recomputed. An identity mismatch writes
a new caption directory and leaves the previous file in place.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from listen_benchmark.j_annotation import read, result_path, write

ROOT = Path(__file__).resolve().parents[2]
GATE_VERSION = 'headshake-direction-v1+expression-gate-v3'
CROP_REVISION = 'quality-bounded-crop-v1-20261030'
PROMPT_VERSION = 'native-av-first-plus-nod-vocal-reaction-v1'
SCHEMA = 'listen-j-caption-cache-identity-v1'
_FACE_MODEL = (
    ROOT
    / 'related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task'
)
CAPTION_PREP_PER_GPU = 2
CAPTION_PREP_PER_GPU_MAX = 3
CAPTION_BACKLOG_FLOOR = 25000
CAPTION_GPU_MEMORY = '0.78'
GPU_BUSY_MIB = 2000
PREPARE_LIMIT_S = 180
INFERENCE_LIMIT_S = 120
WRITE_LIMIT_S = 30
ATTEMPT_LIMIT = 3

_DETECTOR_FILES = (
    'scripts/listen_j_corrected_reaction.py',
    'scripts/check_glare_headshake_direction.py',
    'scripts/check_glare_expression_gate.py',
)
_CODE_FILES = _DETECTOR_FILES + (
    'scripts/run_glare_reaction_omni_pilot.py',
    'scripts/run_listen_native_caption_vocal_pilot.py',
    'src/listen_benchmark/j_omni_reaction_vocal.py',
    'src/listen_benchmark/native_caption_quality.py',
    'src/listen_benchmark/caption_guard.py',
)


def _canon(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(_canon(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _files_sha(relative_paths) -> str:
    h = hashlib.sha256()
    for relative in relative_paths:
        h.update(relative.encode())
        h.update(_file_sha(ROOT / relative).encode())
    return h.hexdigest()


def _assigned_dict(relative: str, name: str) -> dict:
    tree = ast.parse((ROOT / relative).read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError(f'{relative} {name} is not a dict')
            return value
    raise ValueError(f'{relative} has no {name}')


@lru_cache(maxsize=1)
def shared_identity() -> dict:
    return {
        'schema': SCHEMA,
        'crop_revision': CROP_REVISION,
        'detector_code_sha256': _files_sha(_DETECTOR_FILES),
        'model_file_sha256': _file_sha(_FACE_MODEL),
        'parameters': {
            'prompt_version': PROMPT_VERSION,
            'headshake_gate': _assigned_dict('scripts/check_glare_headshake_direction.py', 'PARAMS'),
            'expression_gate': _assigned_dict('scripts/check_glare_expression_gate.py', 'PARAMS'),
            'sampling': {'temperature': 0, 'max_tokens': 4096},
        },
        'gate_version': GATE_VERSION,
        'code_sha256': _files_sha(_CODE_FILES),
    }


def clip_identity(row_source, measurement_sha256: str | None) -> dict:
    identity = dict(shared_identity())
    identity['source'] = row_source
    identity['crop'] = {
        'revision': (row_source.get('crop_proposal') or {}).get('revision', CROP_REVISION),
        'proposal': row_source.get('crop_proposal'),
        'video_transform': row_source.get('video_transform'),
    }
    identity['detector_measurement_sha256'] = measurement_sha256
    return identity


def _guard_work(work) -> Path:
    path = Path(work).resolve()
    if 'releases' in path.parts:
        raise ValueError('Refusing to write a caption cache into a release archive')
    return path


def caption_file(work, row_source, measurement_sha256, sample_id) -> Path:
    """Versioned caption path. The directory changes when the shared identity or this row changes."""
    root = _guard_work(work)
    shared = digest(shared_identity())[:16]
    clip = digest({'source': row_source, 'measurement': measurement_sha256})[:16]
    path = result_path(root, f'captions/{shared}/{clip}', sample_id).resolve()
    if root != path and root not in path.parents:
        raise ValueError('Caption cache escaped the work directory')
    return path


def measurement_sha256(work, sample_id) -> str | None:
    cfg_path = Path(work) / 'config.json'
    if not cfg_path.exists():
        return None
    cache = (read(cfg_path).get('reaction_input') or {}).get('reaction_cache')
    if not cache:
        return None
    path = result_path(cache, 'reaction', sample_id)
    if not path.exists():
        return None
    return _file_sha(path)


def identity_matches(stored, expected) -> bool:
    """Extra audit fields may sit beside the required identity. A required field that differs does not match."""
    if not isinstance(stored, dict):
        return False
    return all(stored.get(key) == value for key, value in expected.items())


def row_needs_caption(work, row) -> bool:
    sample_id = row['sample_id']
    if result_path(work, 'held', sample_id).exists() or result_path(work, 'failed', sample_id).exists():
        return False
    sha = measurement_sha256(work, sample_id)
    path = caption_file(work, row['source'], sha, sample_id)
    if not path.exists():
        return True
    record = read(path)
    return not identity_matches(record.get('cache_identity'), clip_identity(row['source'], sha))


def ordered_prefetch_shutdown(stop_reader, settle_pending, release_claims):
    """Stop claiming first, wait until in-flight work has stopped, then release locks."""
    stop_reader()
    settle_pending()
    release_claims()


def record_stage_timeout(work, sample_id, stage, detail, limit=ATTEMPT_LIMIT) -> str:
    """Save the diagnostic before deciding whether this sample is isolated."""
    work = Path(work)
    attempts_path = work / 'caption_attempts' / f'{sample_id}.json'
    attempts = read(attempts_path) if attempts_path.exists() else {'sample_id': sample_id, 'attempts': []}
    attempts['attempts'].append({'stage': stage, 'at': time.time()})
    write(attempts_path, attempts)
    diagnostic = {
        'sample_id': sample_id,
        'stage': stage,
        'attempt': len(attempts['attempts']),
        'at': time.time(),
        'detail': detail,
    }
    write(work / 'diagnostics' / f'{sample_id}-{stage}-{time.time_ns()}.json', diagnostic)
    if len(attempts['attempts']) >= limit:
        write(
            result_path(work, 'failed', sample_id),
            {
                'sample_id': sample_id,
                'stage': stage,
                'attempts': len(attempts['attempts']),
                'reason': 'stage_timeout',
                'at': time.time(),
                'training_admitted': False,
            },
        )
        return 'isolate'
    return 'restart'


def reconcile_completion(work, rows) -> dict:
    """Classify every manifest row. Failures and omissions are not a successful completion."""
    ids = [row['sample_id'] for row in rows]
    duplicates = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    buckets = {
        'pass': [],
        'generation_failed': [],
        'insufficient_evidence': [],
        'processing_failed': [],
        'not_processed': [],
    }
    seen = set()
    for row in rows:
        sample_id = row['sample_id']
        if sample_id in seen:
            continue
        seen.add(sample_id)
        buckets[_classify(work, row)].append(sample_id)
    problems = duplicates or any(buckets[name] for name in buckets if name != 'pass')
    if not rows:
        problems = True
    status = {
        'status': 'completed_with_failures' if problems else 'succeeded',
        'counts': {name: len(items) for name, items in buckets.items()},
        'duplicate_ids': duplicates,
        'missing_ids': list(buckets['not_processed']),
        'pass': buckets['pass'],
        'generation_failed': buckets['generation_failed'],
        'insufficient_evidence': buckets['insufficient_evidence'],
        'processing_failed': buckets['processing_failed'],
        'not_processed': buckets['not_processed'],
    }
    return status


def _classify(work, row) -> str:
    sample_id = row['sample_id']
    if not row.get('source'):
        if result_path(work, 'held', sample_id).exists():
            return 'insufficient_evidence'
        if result_path(work, 'failed', sample_id).exists():
            return 'processing_failed'
        return 'not_processed'
    sha = measurement_sha256(work, sample_id)
    path = caption_file(work, row['source'], sha, sample_id)
    if path.exists():
        record = read(path)
        identity = clip_identity(row['source'], sha)
        if not identity_matches(record.get('cache_identity'), identity):
            return 'not_processed'
        flags = record.get('review_flags') or []
        if record.get('generation_checks_passed') and 'coverage_uncertain' not in flags and 'invented_vocal_on_silence' not in flags:
            return 'pass'
        return 'generation_failed'
    if result_path(work, 'held', sample_id).exists():
        return 'insufficient_evidence'
    if result_path(work, 'failed', sample_id).exists():
        return 'processing_failed'
    return 'not_processed'


def load_fixed_manifest(work):
    """The checklist is the retained manifest, not whichever jobs have been created."""
    work = Path(work)
    local = work / 'fixed_manifest.jsonl'
    path = local if local.exists() else None
    if path is None and (work / 'config.json').exists():
        configured = read(work / 'config.json').get('retained_manifest')
        if configured and Path(configured).exists():
            path = Path(configured)
    if path is None:
        return None
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def completion_rows(work):
    fixed = load_fixed_manifest(work)
    if fixed is None:
        return None
    by_id = {}
    for row in job_rows(work):
        by_id.setdefault(row['sample_id'], row)
    out = []
    for item in fixed:
        sample_id = item['sample_id'] if isinstance(item, dict) else item
        if sample_id in by_id:
            out.append(by_id[sample_id])
        elif isinstance(item, dict) and item.get('source'):
            out.append(item)
        else:
            out.append({'sample_id': sample_id})
    return out


def job_rows(work):
    rows = []
    folder = Path(work) / 'jobs'
    if folder.exists():
        for path in sorted(folder.glob('*.json')):
            rows.extend(read(path).get('rows') or [])
    return rows


def write_completion_status(work, rows=None) -> dict:
    from .j_annotation import locked
    with locked(Path(work) / 'completion_audit.lock'):
        saved = Path(work) / 'completion_status.json'
        version = digest(shared_identity())
        queue = Path(work) / 'queues/caption/state.json'
        journal = Path(work) / 'ready.jsonl'
        queue_version = digest({'state': read(queue) if queue.exists() else None, 'ready_bytes': journal.stat().st_size if journal.exists() else None})
        if saved.exists() and read(saved).get('identity_version') == version and read(saved).get('audit_queue_identity') == queue_version:
            return read(saved)
        result = _write_completion_status(work, rows)
        result['identity_version'] = version
        result['audit_queue_identity'] = queue_version
        write(saved, result)
        return result

def _write_completion_status(work, rows=None) -> dict:
    if rows is None:
        rows = completion_rows(work)
    if rows is None:
        status = {
            'status': 'completed_with_failures',
            'reason': 'fixed_manifest_missing',
            'counts': {},
            'duplicate_ids': [],
            'missing_ids': [],
        }
    else:
        status = reconcile_completion(work, rows)
    write(Path(work) / 'completion_status.json', status)
    return status


def due_prepares(pending, now, limit):
    """Futures that have not finished and have already exceeded the prepare limit."""
    return [
        fut
        for fut, meta in pending.items()
        if not fut.done() and now - meta['submitted_at'] >= limit
    ]


def reap_processes(processes, grace_s=2.0):
    """Terminate child processes and return only those still alive afterward."""
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    deadline = time.monotonic() + grace_s
    for proc in processes:
        proc.join(timeout=max(0.0, deadline - time.monotonic()))
    for proc in processes:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
    return [proc.pid for proc in processes if proc.is_alive()]


def release_after_children_exit(processes, release_claims, grace_s=2.0):
    """Release claim locks only after prepare workers have exited."""
    still = reap_processes(processes, grace_s=grace_s)
    if still:
        return still
    release_claims()
    return []


def reaction_cache_problems(record) -> list[str]:
    """A measurement is usable only when its stored gate and face-model hash match the current ones."""
    expected_shake = _assigned_dict('scripts/check_glare_headshake_direction.py', 'PARAMS')
    expected_expr = _assigned_dict('scripts/check_glare_expression_gate.py', 'PARAMS')
    params = record.get('parameters') or {}
    problems = []
    if params.get('headshake_direction_gate') != expected_shake:
        problems.append('headshake_gate')
    if params.get('expression_gate_v3') != expected_expr:
        problems.append('expression_gate')
    if record.get('model_sha256') != _file_sha(_FACE_MODEL):
        problems.append('model_sha256')
    return problems


def republish_caption_jobs(work):
    """Put caption jobs back on the queue when their receipt is stale. Detection files stay."""
    work = Path(work)
    folder = work / 'caption' / 'completed'
    republished = []
    if not folder.exists():
        return republished
    journal = work / 'ready.jsonl'
    for receipt_path in sorted(folder.glob('*.json')):
        if receipt_path.name.endswith('.stale'):
            continue
        job_id = receipt_path.stem
        job_path = work / 'jobs' / f'{job_id}.json'
        if not job_path.exists():
            continue
        job = read(job_path)
        if not any(row_needs_caption(work, row) for row in job.get('rows') or []):
            continue
        receipt_path.replace(receipt_path.with_name(receipt_path.name + '.stale'))
        with journal.open('a') as handle:
            handle.write(json.dumps({'job_id': job_id}) + '\n')
        republished.append(job_id)
    if republished:
        (work / 'completion_status.json').unlink(missing_ok=True)
    return republished


def archive_status(archive_dir, workspace_root) -> dict:
    """Report archive integrity separately from workspace drift. Does not rewrite the archive."""
    archive_dir = Path(archive_dir)
    manifest = json.loads((archive_dir / 'manifest.json').read_text())
    archive_root = archive_dir / 'code'
    archive_bad = []
    workspace_bad = []
    for rel, expected in manifest['archive_sha256'].items():
        archived = archive_root / rel
        if not archived.exists() or _file_sha(archived) != expected:
            archive_bad.append(rel)
            continue
        live = Path(workspace_root) / rel
        if not live.exists() or _file_sha(live) != expected:
            workspace_bad.append(rel)
    return {
        'archive_unchanged': not archive_bad,
        'archive_mismatches': archive_bad,
        'workspace_consistent': not workspace_bad,
        'workspace_mismatches': workspace_bad,
    }


def reaction_ready(work, sample_id) -> bool:
    cfg_path = Path(work) / 'config.json'
    if not cfg_path.exists():
        return False
    cache = (read(cfg_path).get('reaction_input') or {}).get('reaction_cache')
    if not cache:
        return False
    return result_path(cache, 'reaction', sample_id).exists()


def partition_caption_rows(work, rows):
    """Rows with a reaction measurement can be captioned. The rest wait for detection."""
    ready = []
    awaiting = []
    for row in rows:
        if not row_needs_caption(work, row):
            continue
        if reaction_ready(work, row['sample_id']):
            ready.append(row)
        else:
            awaiting.append(row)
    return ready, awaiting


def caption_job_eligible(work, job_id) -> bool:
    path = Path(work) / 'jobs' / f'{job_id}.json'
    if not path.exists():
        return False
    ready, awaiting = partition_caption_rows(work, read(path).get('rows') or [])
    # A worker may die after writing its final clip but before the pair receipt.
    # Such a pair must still be claimable so the next worker can commit it.
    return bool(ready) or not awaiting


def detected_uncaptioned(work) -> int:
    def clips(branch):
        path = Path(work) / 'queues' / branch / 'state.json'
        if not path.exists():
            return 0
        return int(read(path).get('completed_clips') or 0)

    return max(0, clips('detect') - clips('caption'))


def caption_prep_workers(preparation_wait_s, inference_s, backlog) -> int:
    """Keep two preparers per GPU. Add a third only while detection is ahead and GPUs are waiting."""
    if backlog < CAPTION_BACKLOG_FLOOR:
        return CAPTION_PREP_PER_GPU
    if inference_s >= 30 and preparation_wait_s > inference_s * 0.25:
        return CAPTION_PREP_PER_GPU_MAX
    return CAPTION_PREP_PER_GPU


def detect_worker_ceiling(cores, backlog, caption_gpus) -> int:
    """Leave room for caption preparation while detected clips are waiting. Give the cores back once that backlog is small."""
    if caption_gpus <= 0:
        return max(1, cores)
    # Even when detection falls behind, never starve the GPU input pipeline.
    # A detector has native-library threads too; this is a ceiling, not a target.
    reserve = 8 + caption_gpus * CAPTION_PREP_PER_GPU
    if backlog >= CAPTION_BACKLOG_FLOOR:
        reserve += caption_gpus
    return max(1, cores - min(max(0, cores - 1), reserve))
