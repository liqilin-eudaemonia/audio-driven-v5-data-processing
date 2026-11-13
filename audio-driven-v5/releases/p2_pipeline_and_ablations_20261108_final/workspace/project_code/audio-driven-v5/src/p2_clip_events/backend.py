"""J caches plus clip labels for the existing native matrix training loop."""
import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time
import torch
from p2_listen.contracts import ROOT, read, sha, digest, atomic_json
from p2_listen.cache import tree_to
from p2_matrix16.config import validate,feature_enabled,LOSSES
from .labels import LabelReader, tensors, select_classes
from .model import classes


def data_config(c):
    from p2_j_cache.data import open_manifest
    from p2_listen.config import activate
    if c['loss'] not in LOSSES:raise ValueError('Unsupported J loss recipe')
    if feature_enabled(c) and not c.get('teacher_cache'):
        raise ValueError('Feature supervision requires an audited teacher cache')
    if not feature_enabled(c) and c.get('teacher_cache'):
        raise ValueError('Disabled Feature branch must not load teacher')
    if 'qwen_hidden_source' in c:
        if c['qwen_hidden_source']!='base_pre_sft' or c['stage']!=1 or c['qwen_lora']['enabled']:
            raise ValueError('This comparison requires frozen pre-SFT Qwen H')
        if Path(c['plan_cache']).resolve()!=Path(c['j_cache']).resolve():
            raise ValueError('Pre-SFT guided H must come from the original J cache')
    d = open_manifest(c['j_cache'])['config']
    activate(d)
    return d


def source_identity():
    from p2_matrix16.config import source_identity as shared
    result = shared()
    result[str(ROOT/'scripts/p2_clip_events.py')] = sha(ROOT/'scripts/p2_clip_events.py')
    from p2_j_teacher.core import code_identity
    result.update(code_identity())
    return result


def teacher_identity(c):
    if not feature_enabled(c):return None
    folder=Path(c['teacher_cache']);protocol=read(folder/'protocol.json');complete=read(folder/'COMPLETE.json')
    if (Path(protocol['source']).resolve()!=Path(c['j_cache']).resolve()
            or not complete['success'] or complete['samples']!=protocol['samples']
            or complete['protocol_sha256']!=sha(folder/'protocol.json')
            or Path(complete['source']).resolve()!=Path(c['j_cache']).resolve()):
        raise ValueError('Teacher source or final audit mismatch')
    return dict(path=str(folder.resolve()),protocol_sha256=sha(folder/'protocol.json'),complete_sha256=sha(folder/'COMPLETE.json'))


def data_identity(c):
    m = read(Path(c['clip_labels'])/'manifest.json')
    spec = c['event_supervision']
    expected = select_classes(m, spec['min_positive']) if 'min_positive' in spec else m['classes']
    if (m['cache_manifest_sha256'] != sha(Path(c['j_cache'])/'manifest.json')
            or expected != spec['classes']):
        raise ValueError('Clip label/cache/class identity mismatch')
    result=dict(cache_manifest=m['cache_manifest_sha256'], clip_labels=sha(Path(c['clip_labels'])/'manifest.json'),
                database=m['database_sha256'], event_supervision=c['event_supervision'])
    if feature_enabled(c):result.update(teacher=teacher_identity(c),qwen_hidden_source='base_pre_sft' if c['stage']==1 else 'base_initialized_online_lora')
    return result


def selected_statistics(c, manifest):
    return {name: manifest['statistics'][name] for name in classes(c['event_supervision'])}


class Dataset:
    def __init__(self, c):
        from p2_j_cache.runtime import CacheReader
        from p2_listen.config import extractor_config
        self.spec = c;self.data = data_config(c);data_identity(c)
        self.labels = LabelReader(c['clip_labels'])
        self.reader = CacheReader(c['j_cache'])
        ids = self.labels.ids()
        if (digest(ids) != self.labels.manifest['sample_ids_sha256']
                or not set(ids) <= set(self.reader.ids('train'))):
            raise ValueError('Clip label membership differs')
        self.rows = [dict(sample_id=sid) for sid in ids]
        self.names = classes(c['event_supervision']);self.cfg = extractor_config(self.data)
        self.teacher_reader=None
        if feature_enabled(c):
            from p2_j_teacher.core import Reader as TeacherReader
            self.teacher_reader=TeacherReader(c['teacher_cache'])
            if not set(ids)<=self.teacher_reader.ids:raise ValueError('Teacher is missing selected training members')
        if c['stage'] == 2:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.cfg.omni_path, local_files_only=True)

    def sample(self, index, device='cpu'):
        from p2_j_cache.data import sample, read_input, observation
        sid = self.rows[index]['sample_id'];label = self.labels.row(sid)
        if label['source_digest'] != self.reader.manifest['entries'][sid]:
            raise ValueError('Clip label/source mismatch')
        guided = self.spec['text_condition'] == 'guided'
        if self.spec['stage'] == 1:
            cached = self.reader.training_sample(sid, 'guided' if guided else 'direct', device)
            value = dict(condition=cached['condition'], layout=cached['layout'],
                         text=cached['text'], av=dict(av=cached['av'], support=cached['support']))
        else:
            from p2_listen.features import image_grid
            from p2_listen.layout import make_layout
            item = sample(self.reader.work, self.reader.manifest, sid)
            obs = observation(item, read_input(item, self.reader.work))
            _, _, grid = image_grid(obs, self.processor, self.cfg)
            text = self.reader.part(sid, 'text', device)
            if guided:
                positive = self.reader.part(sid, 'text_guided', device)['positive']
                if positive['prompt'] != item['prompt']:
                    raise ValueError('Online/cached prompt mismatch')
                text = dict(text, positive=positive)
            value = dict(condition=None, layout=tree_to(make_layout(obs['requested_duration_s'], grid), device),
                         text=text, av=self.reader.part(sid, 'vae', device),
                         observation=obs, plan=item['prompt'] if guided else None)
        teacher=None
        if self.teacher_reader is not None:
            from p2_j_teacher.core import validate_geometry
            teacher=self.teacher_reader.load(sid,device)
            validate_geometry(teacher,value['layout'],self.cfg.target_dim)
            teacher={k:teacher[k] for k in ('video','audio','video_valid','audio_valid')}
        value.update(teacher=teacher, events=tensors(label, self.names, self.labels.manifest['absence_weight'], device))
        return value


_preflight_context = None


def _preflight_child():
    _, dataset = _preflight_context
    # Each forked reader owns its SQLite connection; the parent verified its SHA.
    dataset.labels.conn.close()
    dataset.labels = LabelReader(dataset.spec['clip_labels'], verify=False)


def _preflight_row(row):
    from p2_j_cache.runtime import cache_identity
    from p2_j_cache.data import sample
    c, dataset = _preflight_context
    m=dataset.reader.manifest;work=dataset.reader.work
    sid=row['sample_id'];item=sample(work,m,sid)
    if dataset.labels.row(sid)['source_digest']!=m['entries'][sid]:
        raise ValueError('Stale clip labels')
    parts=['vae']+(['text_guided'] if c['text_condition']=='guided' else [])
    if c['stage']==1:parts+=['input_plan' if c['text_condition']=='guided' else 'input']
    for part in parts:
        folder=work/'cache'/part/sid
        expected=cache_identity(m,item,part,dataset.reader.assets[part])
        marker=read(folder/'complete.json')
        if marker['identity']!=digest(expected) or marker['contract']!=expected or not (folder/'payload.pt').stat().st_size:
            raise ValueError('Invalid cache: '+str(folder))
    if dataset.teacher_reader is not None:
        tr=dataset.teacher_reader;folder=tr.work/'cache'/sid;marker=read(folder/'complete.json')
        expected=tr.identity(tr.item(sid))
        if marker['identity']!=digest(expected) or marker['contract']!=expected or not (folder/'payload.pt').stat().st_size:
            raise ValueError('Invalid teacher cache: '+sid)


def preflight(c, processes=64):
    global _preflight_context
    if type(processes) is not int or processes<1:raise ValueError('preflight processes must be positive')
    started=time.monotonic()
    validate(c);dataset = Dataset(c);m = dataset.reader.manifest;work = dataset.reader.work
    # Formal training must wait for the background extractor's full audit.
    if dataset.labels.manifest['full_train']:
        complete = read(work/'COMPLETE.json')
        if not complete['success'] or complete['samples'] != len(dataset.rows):
            raise ValueError('J cache final audit incomplete')
    dataset.reader.part(None, 'text')
    _preflight_context=(c,dataset)
    progress=Path(c['output'])/'preflight_progress.json'
    def report(count):
        value=dict(stage='clip_preflight',samples=count,total=len(dataset.rows),processes=processes,
                   elapsed_s=time.monotonic()-started,time=time.time())
        atomic_json(progress,value);print(value,flush=True)
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        report(0)
        if processes==1:
            for index,row in enumerate(dataset.rows):
                _preflight_row(row)
                if (index+1)%4096==0:report(index+1)
        else:
            with ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('fork'),
                                     initializer=_preflight_child) as pool:
                for start in range(0,len(dataset.rows),8192):
                    batch=dataset.rows[start:start+8192]
                    for _ in pool.map(_preflight_row,batch,chunksize=32):pass
                    report(start+len(batch))
        report(len(dataset.rows))
    finally:
        _preflight_context=None;torch.set_num_threads(old_threads)
    stats = selected_statistics(c, dataset.labels.manifest)
    if 'event' in c['loss'] and not any(v['enabled'] for v in stats.values()):
        raise ValueError('No classes with usable positive/negative contrast')
    value = dict(schema='p2-clip-preflight-v1', data_identity=data_identity(c), source=source_identity(),
                 experiment_mode=[c['text_condition'],c['loss'],c['stage']], statistics=stats, samples=len(dataset.rows),
                 processes=processes,elapsed_s=time.monotonic()-started)
    atomic_json(Path(c['output'])/'preflight.json', value)
    return value


def initialize_stage2(path, model, c, identity):
    from p2_matrix16.checkpoint import initialize_stage2 as shared
    return shared(path, model, c, identity, expected_source=source_identity())


def emit(labels, output, *, min_positive=5000):
    labels, output = Path(labels).resolve(), Path(output).resolve()
    m = read(labels/'manifest.json')
    selected = select_classes(m, min_positive)
    base = read(ROOT/'configs/p2_matrix16/guided_av_event_s1.json')
    paths = []
    for text in ('fixed','guided'):
        for loss in ('av','av_event'):
            for stage in (1,2):
                c = copy.deepcopy(base);ident = f'{text}_{loss}_s{stage}'
                c.update(id=ident, text_condition=text, loss=loss, stage=stage,
                         parent_experiment=f'{text}_{loss}_s1' if stage == 2 else None,
                         j_cache=m['cache_work'], clip_labels=str(labels),
                         data_config=str(Path(m['cache_work'])/'manifest.json'), plan_cache=m['cache_work'], teacher_cache=None,
                         output=str(output/'runs'/ident),
                         connector=dict(fusion_width=1024,fusion_depth=4,fusion_heads=16),
                         event_supervision=dict(mode='clip_multilabel', classes=selected, min_positive=min_positive))
                c['qwen_lora']['enabled'] = stage == 2
                validate(c);path=output/(ident+'.json');atomic_json(path,c,immutable=True);paths.append(str(path))
    return dict(configs=paths, training_started=False)
