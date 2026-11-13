"""Reuse all frozen caches; add only the derived interval-label database."""
import threading
from pathlib import Path
from p2_listen.contracts import read,sha,digest
from p2_whisper_event_feature.data import Dataset as OriginalDataset,to_device
from p2_whisper_event_feature.data import data_identity as original_identity
from .config import validate,temporal_enabled
from .labels import Reader
from .layout import make_layout,rasterize


class LegacyDataset(OriginalDataset):
    def __init__(self,c):
        validate(c);super().__init__(c)
        self.temporal=None
        if not temporal_enabled(c):return
        self.temporal=Reader(c['temporal_labels'])
        self._temporal_owner=threading.get_ident();self._temporal_threads=threading.local()
        m=self.temporal.manifest
        if (self.temporal.ids()!=[r['sample_id'] for r in self.rows] or not m['full_train']
                or m['sample_ids_sha256']!=digest([r['sample_id'] for r in self.rows])
                or m['cache_manifest_sha256']!=sha(Path(c['j_cache'])/'manifest.json')
                or Path(m['cache_work']).resolve()!=Path(c['j_cache']).resolve()
                or m['classes']!=c['hidden_event_supervision']['classes']
                or m['policy']!=c['hidden_event_supervision']['label_policy']):
            raise ValueError('Temporal labels / frozen training cache mismatch')

    def temporal_row(self,sid):
        if threading.get_ident()==self._temporal_owner:return self.temporal.row(sid)
        if not hasattr(self._temporal_threads,'reader'):
            self._temporal_threads.reader=Reader(self.spec['temporal_labels'],verify=False)
        return self._temporal_threads.reader.row(sid)

    def sample(self,index,device='cpu'):
        value=super().sample(index,device);sid=value['sample_id'];row=None
        if self.temporal is not None:
            row=self.temporal_row(sid)
            if row['source_digest']!=self.reader.manifest['entries'][sid]:raise ValueError('Stale temporal target: '+sid)
        if row is not None:duration=row['duration_s']
        else:
            from p2_j_cache.data import sample
            duration=sample(self.reader.work,self.reader.manifest,sid)['source']['duration_s']
        layout=make_layout(value['av']['av'],value['av']['support'],duration)
        events={'clip/'+k:v for k,v in value['events'].items()}
        if row is not None:
            events.update(rasterize(row,layout,self.spec['hidden_event_supervision']['classes'],
                                   self.spec['hidden_event_supervision']['background_weight']))
        if self.spec['hidden_event_supervision']['clip_weight']==0:events={k:v for k,v in events.items() if k.startswith('temporal/')}
        value.update(events=events,hidden_event_layout=layout)
        return value


def data_identity(c):
    validate(c)
    if c.get('current_cache_index'):
        from .current import checked_index
        index=checked_index(c['current_cache_index']);source=Path(c['current_source'])
        value=dict(current_cache_index=sha(c['current_cache_index']),current_cache_receipts=index['caches'],
            current_sources=sha(source/'manifest.json'),clip_manifest=sha(Path(c['clip_labels'])/'manifest.json'),
            source_split_counts=index['source_split_counts'],selection_sha256=index['selection_sha256'])
    else:value=original_identity(c)
    if c.get('clip_numeric_labels'):
        value['clip_numeric_manifest']=sha(Path(c['clip_numeric_labels'])/'manifest.json')
    if not temporal_enabled(c):
        value.update(objective='av+ltx_hidden_clip_event+feature+whisper')
        return value
    p=Path(c['temporal_labels']);m=read(p/'manifest.json')
    if m['database_sha256']!=sha(p/'labels.sqlite'):raise ValueError('Temporal database checksum mismatch')
    value.update(objective='av+ltx_hidden_clip_event+ltx_hidden_temporal_event+feature+whisper',
                 temporal_manifest=sha(p/'manifest.json'),temporal_database=m['database_sha256'])
    return value


def statistics(c,clip_manifest):
    from p2_clip_events.backend import selected_statistics
    if c.get('current_cache_index'):
        selected={}
        for mod in ('video','audio'):
            for k in c['event_supervision']['classes'][mod]:
                s=dict(clip_manifest['statistics'].get(k,dict(positive=0,weak_negative=0,enabled=False,pos_weight=1.,disabled_reason='no_positive_support')))
                if s['positive']<c['event_supervision'].get('min_positive',1):
                    s.update(enabled=False,disabled_reason='below_frozen_clip_min_positive')
                selected[k]=s
    else:selected=selected_statistics(c,clip_manifest)
    result={'clip/'+k:v for k,v in selected.items()} if c['hidden_event_supervision']['clip_weight']>0 else {}
    if not temporal_enabled(c):return result
    m=read(Path(c['temporal_labels'])/'manifest.json');background=c['hidden_event_supervision']['background_weight']
    for mod in ('video','audio'):
        for name in c['hidden_event_supervision']['classes'][mod]:
            s=m['statistics'][name];pos=s['positive_seconds'];neg=s['background_seconds']
            enabled=pos>0 and neg>0 and background>0
            result['temporal/'+name]=dict(s,enabled=enabled,pos_weight=max(1.,min(10.,background*neg/max(pos,1e-12))),
                disabled_reason=None if enabled else 'no_positive_background_contrast',
                weight_basis='physical_covered_seconds; runtime normalization uses actual valid time bins')
    return result


def Dataset(c):
    if c.get('current_cache_index'):
        from .current_data import CurrentDataset
        return CurrentDataset(c)
    return LegacyDataset(c)


def data_config(c):
    if c.get('current_cache_index'):
        index=read(c['current_cache_index'])
        return read(Path(index['caches']['av_vae']['work'])/'protocol.json')['config']
    from p2_clip_events.backend import data_config as original
    return original(c)


def sample_list(c):
    return Path(c.get('current_source',c['j_cache']))/'sample_list.csv'
