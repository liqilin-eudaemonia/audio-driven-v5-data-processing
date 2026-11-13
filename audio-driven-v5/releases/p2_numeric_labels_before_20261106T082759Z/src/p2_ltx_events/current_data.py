"""All five CURRENT mixed readers, preserving train+dev membership and splits."""
from pathlib import Path
from types import SimpleNamespace
import csv,json,sqlite3,threading
from p2_listen.contracts import read,sha,digest
from p2_clip_events.labels import LabelReader,tensors
from p2_clip_events.model import classes
from .current import checked_index
from .config import temporal_enabled
from .labels import Reader as TemporalReader
from .layout import make_layout,rasterize


class SourceReader:
    def __init__(self,work,verify=True):
        self.work=Path(work);self.manifest=read(self.work/'manifest.json');m=self.manifest
        if m['kind']!='current-label-sources':raise ValueError('Expected current label-source view')
        if verify and sha(self.work/'sources.sqlite')!=m['database_sha256']:raise ValueError('Current sources database changed')
        self.conn=sqlite3.connect((self.work/'sources.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    def row(self,sid):
        value=self.conn.execute('SELECT value FROM samples WHERE sid=?',(sid,)).fetchone()
        if value is None:raise KeyError(sid)
        row=json.loads(value[0])
        if digest(row)!=self.manifest['entries'][sid]:raise ValueError('Current label source changed: '+sid)
        return row


def open_readers(index,verify=True):
    from p2_j_current_av_cache import Reader as AVReader
    from p2_j_current_whisper_cache import Reader as WhisperReader
    from p2_j_current_gemma_cache import Reader as TextReader
    from p2_j_current_qwen_hidden import Reader as HReader
    import p2_j_current_teacher_cache as teacher
    from p2_whisper_event_feature.data import CompatibleTeacherReader
    from p2_whisper_av.compat import canonical_dependencies
    from p2_plan_cache.runtime import dependencies
    caches=index['caches']
    av=AVReader(caches['av_vae']['work'],verify=verify)
    text=TextReader(caches['gemma_text']['work'],verify=False)
    if verify:
        if canonical_dependencies(dependencies(text.config,'text'))!=canonical_dependencies(text.protocol['dependencies']):
            raise ValueError('Current Gemma assets changed beyond byte-identical private relocation')
        av.source.db.close();av.source.stream.close()
        av.source=HReader(caches['qwen_hidden']['work'],verify=True)
    av.text=text
    # Same mixed Reader methods; replace only the old source reader with the
    # existing audited three-file Gemma relocation adapter. No source hashes,
    # assets, payload checks, membership or geometry checks are relaxed.
    target=teacher.Reader.__new__(teacher.Reader);target.work=Path(caches['listener_teacher']['work'])
    target.new=teacher.IncrementReader(target.work,verify=verify);p=target.new.protocol
    if sha(teacher.OLD/'protocol.json')!=p['old_protocol_sha256'] or sha(teacher.OLD/'COMPLETE.json')!=p['old_complete_sha256']:
        raise ValueError('Accepted old teacher changed')
    target.old=CompatibleTeacherReader(teacher.OLD);target.h=HReader(teacher.SOURCE,verify=False)
    if read(target.work/'MIXED_COMPLETE.json')['samples']!=index['samples']:raise ValueError('Current teacher mixed map incomplete')
    return SimpleNamespace(av=av,whisper=WhisperReader(caches['speaker_whisper']['work']),teacher=target)


class CurrentDataset:
    def __init__(self,c,verify=True):
        from p2_listen.config import extractor_config
        self.spec=c;self.index=checked_index(c['current_cache_index']);self.source=SourceReader(c['current_source'],verify)
        self.labels=LabelReader(c['clip_labels'],verify=verify)
        self.temporal=TemporalReader(c['temporal_labels'],verify=verify) if temporal_enabled(c) else None
        self.names=classes(c['event_supervision']);self.owner=threading.get_ident();self.local=threading.local()
        self.main=open_readers(self.index,verify);self.reader=self.main.av;self.whisper=self.main.whisper;self.teacher_reader=self.main.teacher
        ids=self.reader.protocol['sample_ids'];m=self.source.manifest;lm=self.labels.manifest
        if (ids!=m['training_ids'] or ids!=self.labels.ids()
                or self.index['source_split_counts']!=m['source_split_counts']
                or sha(c['current_cache_index'])!=m['current_cache_index_sha256']
                or lm['cache_manifest_sha256']!=sha(Path(c['current_source'])/'manifest.json')):
            raise ValueError('Current all-cache / clip / temporal source membership differs')
        if self.temporal is not None:
            tm=self.temporal.manifest
            if (ids!=self.temporal.ids() or tm['cache_manifest_sha256']!=sha(Path(c['current_source'])/'manifest.json')
                    or tm['classes']!=c['hidden_event_supervision']['classes']):
                raise ValueError('Current temporal source membership differs')
        with (Path(c['current_source'])/'sample_list.csv').open() as stream:self.rows=list(csv.DictReader(stream))
        if [r['sample_id'] for r in self.rows]!=ids:raise ValueError('Current duration/order mismatch')
        self.cfg=extractor_config(self.reader.config)

    def _readers(self):
        if threading.get_ident()==self.owner:return self.main,self.source,self.labels,self.temporal
        if not hasattr(self.local,'readers'):
            self.local.readers=(open_readers(self.index,False),SourceReader(self.spec['current_source'],False),
                LabelReader(self.spec['clip_labels'],verify=False),
                TemporalReader(self.spec['temporal_labels'],verify=False) if temporal_enabled(self.spec) else None)
        return self.local.readers

    def reopen_after_fork(self):
        self.main=open_readers(self.index,False);self.reader=self.main.av
        self.source.conn.close();self.source=SourceReader(self.spec['current_source'],False)
        self.labels.conn.close();self.labels=LabelReader(self.spec['clip_labels'],verify=False)
        if self.temporal is not None:
            self.temporal.conn.close();self.temporal=TemporalReader(self.spec['temporal_labels'],verify=False)
        self.local=threading.local();self.owner=threading.get_ident()

    def sample(self,index,device='cpu'):
        from p2_j_teacher.core import validate_geometry
        sid=self.rows[index]['sample_id'];r,source,labels,temporal=self._readers()
        item=source.row(sid);current=r.av.source.row(sid)
        if item['selected_row_sha256']!=digest(current) or current['split']!=item['split']:
            raise ValueError('Current cache / selected label source mismatch')
        cached=r.av.training_sample(sid,'guided',device);clip=labels.row(sid)
        timed=temporal.row(sid) if temporal is not None else None
        if clip['source_digest']!=digest(item) or timed is not None and timed['source_digest']!=digest(item):raise ValueError('Stale current Event target')
        teacher=r.teacher.load(sid,device);validate_geometry(teacher,cached['layout'],self.cfg.target_dim)
        layout=make_layout(cached['av'],cached['support'],item['source']['duration_s'])
        events={'clip/'+k:v for k,v in tensors(clip,self.names,labels.manifest['absence_weight'],device).items()}
        if timed is not None:
            events.update(rasterize(timed,layout,self.spec['hidden_event_supervision']['classes'],
                self.spec['hidden_event_supervision']['background_weight']))
        if self.spec['hidden_event_supervision']['clip_weight']==0:events={k:v for k,v in events.items() if k.startswith('temporal/')}
        return dict(sample_id=sid,source_split=item['split'],condition=cached['condition'],layout=cached['layout'],
            teacher={k:teacher[k] for k in ('video','audio','video_valid','audio_valid')},events=events,
            hidden_event_layout=layout,text=cached['text'],whisper=r.whisper.load(sid,device),
            av=dict(av=cached['av'],support=cached['support']))

    def preflight_row(self,sid):
        """Verify contracts and payload presence without decoding cached tensors."""
        from p2_j_current_gemma_cache import Reader as TextReader
        r=self.main;item=self.source.row(sid);row=r.av.source.row(sid)
        if digest(row)!=item['selected_row_sha256']:raise ValueError('Current selected row changed')
        targets=[self.labels.row(sid)]
        if self.temporal is not None:targets.append(self.temporal.row(sid))
        if any(x['source_digest']!=digest(item) for x in targets):
            raise ValueError('Current Event source identity mismatch')
        def marker(folder,contract=None,identity=None,payload_sha=None):
            folder=Path(folder);m=read(folder/'complete.json')
            if (contract is not None and (m['contract']!=contract or m['identity']!=digest(contract))
                    or identity is not None and m['identity']!=identity
                    or payload_sha is not None and m['payload_sha256']!=payload_sha
                    or not (folder/'payload.pt').stat().st_size):raise ValueError('Invalid current cache marker: '+str(folder))
        marker(r.av.source.work/'cache/input_plan'/sid,r.av.source.identity(row,'input_plan'))
        if not hasattr(self,'_preflight_text'):self._preflight_text=TextReader(self.index['caches']['gemma_text']['work'],verify=False)
        text=self._preflight_text;marker(text.work/'cache/text_guided'/sid,text.identity(row))
        plan=r.av.plan(sid)
        if plan['disposition']=='reuse_both':marker(Path(plan['old_payload']).parent,identity=plan['old_identity'],payload_sha=plan['payload_sha256'])
        else:marker(r.av.work/'cache/vae'/sid,r.av.identity(row))
        w=r.whisper.new if sid in r.whisper.new.ids else r.whisper.old
        marker(w.folder(sid),w.identity(w.item(sid)))
        t=r.teacher.new if sid in r.teacher.new.ids else r.teacher.old
        if t is r.teacher.new:marker(t.work/'cache'/sid,t.identity(t.item(sid)))
        else:marker(t.work/'cache'/sid,t.identity(t.item(sid)))
