"""Frozen planning-SFT H overlay for J AV+Event, isolated from active runs."""
import copy
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time
import torch
from p2_listen.contracts import ROOT, read, sha, digest, atomic_json
from p2_clip_events import backend as base
from p2_clip_events.labels import tensors


def base_config(c):
    if (c.get('qwen_hidden_source') != 'planning_sft_frozen' or c['text_condition'] != 'guided'
            or c['stage'] != 1 or c['loss'] != 'av_event' or c['qwen_lora']['enabled']
            or c.get('teacher_cache') or c.get('event_head_training') is not True
            or Path(c['plan_cache']).resolve() != Path(c['sft_hidden_cache']).resolve()):
        raise ValueError('Expected frozen planning-SFT guided H with AV+Event and no Feature teacher')
    result = copy.deepcopy(c)
    result.pop('qwen_hidden_source'); result['plan_cache'] = result['j_cache']
    return result


@lru_cache(maxsize=2)
def overlay(work):
    from p2_j_sft_hidden import Reader, require_ready
    r = Reader(work); require_ready(r)
    done = read(r.work / 'COMPLETE.json')
    if (not done['success'] or done['samples'] != r.protocol['samples']
            or done['protocol_sha256'] != r.protocol_sha or done['adapter'] != r.protocol['adapter']
            or digest(r.protocol['sample_ids']) != r.protocol['ids_sha256']
            or len(r.ids) != r.protocol['samples']):
        raise ValueError('SFT H final audit/member/adapter mismatch')
    return r


def data_config(c):
    d = base.data_config(base_config(c))
    r = overlay(str(Path(c['sft_hidden_cache']).resolve()))
    if Path(r.protocol['source']).resolve() != Path(c['j_cache']).resolve():
        raise ValueError('SFT H and target J cache have different sources')
    return d


def data_identity(c):
    data_config(c)
    r = overlay(str(Path(c['sft_hidden_cache']).resolve()))
    return dict(base.data_identity(base_config(c)), qwen_hidden_source='planning_sft_frozen',
        sft_hidden=dict(path=str(r.work.resolve()), protocol_sha256=r.protocol_sha,
            complete_sha256=sha(r.work / 'COMPLETE.json'), adapter=r.protocol['adapter']))


def source_identity():
    from p2_j_sft_hidden import code_identity
    result = base.source_identity(); result.update(code_identity())
    for path in (Path(__file__), ROOT / 'scripts/p2_j_sft_event_train.py',
                 ROOT / 'scripts/p2_j_duration_execution.py', ROOT / 'scripts/p2_duration_execution.py'):
        result[str(path)] = sha(path)
    return result


class Dataset(base.Dataset):
    def __init__(self, c):
        data_config(c)
        super().__init__(base_config(c))
        self.spec = c
        self.overlay = overlay(str(Path(c['sft_hidden_cache']).resolve()))
        if set(self.overlay.protocol['sample_ids']) != set(self.reader.ids('train')):
            raise ValueError('SFT H full train membership differs')
        if not {r['sample_id'] for r in self.rows} <= self.overlay.ids:
            raise ValueError('Selected labels missing SFT H')

    def sample(self, index, device='cpu'):
        sid = self.rows[index]['sample_id']; label = self.labels.row(sid)
        if label['source_digest'] != self.reader.manifest['entries'][sid]:
            raise ValueError('Label/target source mismatch')
        cached = self.overlay.training_sample(sid, device, include_teacher=False)
        return dict(condition=cached['condition'], layout=cached['layout'], text=cached['text'],
            av=dict(av=cached['av'], support=cached['support']), teacher=None,
            events=tensors(label, self.names, self.labels.manifest['absence_weight'], device))


_context = None


def _child():
    from p2_clip_events.labels import LabelReader
    ds = _context
    ds.labels.conn.close(); ds.labels = LabelReader(ds.spec['clip_labels'], verify=False)


def _row(row):
    from p2_j_cache.runtime import cache_identity
    ds = _context; sid = row['sample_id']; item = ds.overlay.item(sid)
    if ds.labels.row(sid)['source_digest'] != ds.reader.manifest['entries'][sid]:
        raise ValueError('Stale event labels: ' + sid)
    for part in ('vae', 'text_guided'):
        expected = cache_identity(ds.reader.manifest, item, part, ds.reader.assets[part])
        folder = ds.reader.work / 'cache' / part / sid
        marker = read(folder / 'complete.json')
        if marker['identity'] != digest(expected) or marker['contract'] != expected or not (folder / 'payload.pt').stat().st_size:
            raise ValueError('Invalid reused cache: ' + sid + '/' + part)
    expected = ds.overlay.identity(item); folder = ds.overlay.work / 'cache/input_plan' / sid
    marker = read(folder / 'complete.json')
    if marker['identity'] != digest(expected) or marker['contract'] != expected or not (folder / 'payload.pt').stat().st_size:
        raise ValueError('Invalid SFT H: ' + sid)


def preflight(c, processes=16):
    global _context
    from p2_matrix16.config import validate
    if type(processes) is not int or processes < 1:raise ValueError('Positive CPU process count required')
    validate(c, formal=True); ds = Dataset(c); _context = ds
    ds.reader.part(None, 'text')
    done = read(ds.reader.work / 'COMPLETE.json')
    if not done['success'] or done['samples'] != len(ds.rows):raise ValueError('Original J cache incomplete')
    started = time.monotonic(); old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    def progress(n):
        value = dict(samples=n,total=len(ds.rows),elapsed_s=time.monotonic()-started,processes=processes)
        atomic_json(Path(c['output']) / 'preflight_progress.json', value); print(value, flush=True)
    try:
        progress(0)
        if processes == 1:
            for row in ds.rows:_row(row)
        else:
            with ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('fork'),initializer=_child) as pool:
                for offset in range(0,len(ds.rows),8192):
                    batch=ds.rows[offset:offset+8192]
                    for _ in pool.map(_row,batch,chunksize=32):pass
                    progress(offset+len(batch))
        progress(len(ds.rows))
    finally:_context=None;torch.set_num_threads(old_threads)
    stats = base.selected_statistics(c,ds.labels.manifest)
    if not any(v['enabled'] for v in stats.values()):raise ValueError('No usable Event supervision')
    value=dict(schema='p2-j-sft-event-preflight-v1',samples=len(ds.rows),data_identity=data_identity(c),
        source=source_identity(),experiment_mode=[c['text_condition'],c['loss'],c['stage']],
        statistics=stats,elapsed_s=time.monotonic()-started,processes=processes)
    atomic_json(Path(c['output']) / 'preflight.json',value);return value


def initialize_stage2(*args, **kwargs):
    raise ValueError('This backend supports frozen cached SFT H in stage1 only')
