"""Join existing audited AV/H/text/Event/teacher/speaker caches by exact sample ID."""
from pathlib import Path
import threading
from p2_listen.contracts import read,sha
from p2_whisper_av.data import Dataset as WhisperDataset,data_identity as whisper_identity
from p2_whisper_av.compat import cache_reader_class
from p2_j_teacher.core import Reader as TeacherReader,code_identity,validate_geometry
from p2_clip_events.labels import LabelReader,tensors
from p2_clip_events.model import classes
from .config import validate,pure_av_view


def to_device(value,device):
    """Move both condition dataclasses as well as ordinary nested target tensors."""
    from p2_listen.cache import tree_to
    condition=value['condition'].to(device);speaker=value['whisper'].to(device)
    result=tree_to(dict(value,condition=None,whisper=None),device)
    result.update(condition=condition,whisper=speaker)
    return result


class CompatibleTeacherReader(TeacherReader):
    """Only the same byte-identical Gemma relocation as the existing Whisper arm."""
    def __init__(self,work):
        from p2_plan_cache.runtime import dependencies
        self.work=Path(work);p=self.protocol=read(self.work/'protocol.json')
        self.source=cache_reader_class()(p['source'])
        done=read(self.source.work/'COMPLETE.json')
        if not done['success'] or done['samples']!=len(self.source.ids('train')):
            raise ValueError('Teacher source cache is not fully accepted')
        if (p['schema']!='p2-j-feature-teacher-v1' or p['code']!=code_identity()
                or p['source_manifest_sha256']!=sha(self.source.work/'manifest.json')
                or p['source_complete_sha256']!=sha(self.source.work/'COMPLETE.json')
                or p['dependencies']!=dependencies(self.source.manifest['config'],'input_plan')):
            raise ValueError('Teacher source/code/assets differ')
        self.protocol_sha=sha(self.work/'protocol.json');self.ids=set(p['sample_ids'])


class Dataset(WhisperDataset):
    def __init__(self,c):
        validate(c)
        super().__init__(pure_av_view(c))
        self.spec=c
        self.labels=LabelReader(c['clip_labels'])
        self._label_owner=threading.get_ident();self._thread_labels=threading.local()
        self.names=classes(c['event_supervision'])
        from p2_clip_events.backend import data_config,data_identity
        from p2_listen.config import extractor_config
        data_identity(c);self.cfg=extractor_config(data_config(c))
        self.teacher_reader=CompatibleTeacherReader(c['teacher_cache'])
        if not {r['sample_id'] for r in self.rows}<=self.teacher_reader.ids:
            raise ValueError('Feature teacher is missing training members')

    def sample(self,index,device='cpu'):
        value=super().sample(index,device);sid=self.rows[index]['sample_id']
        if threading.get_ident()==self._label_owner:
            labels=self.labels
        else:
            # The main thread verified the immutable SQLite hash already. Each
            # CPU prefetch thread owns its own read-only connection.
            if not hasattr(self._thread_labels,'reader'):
                self._thread_labels.reader=LabelReader(self.spec['clip_labels'],verify=False)
            labels=self._thread_labels.reader
        label=labels.row(sid)
        if label['source_digest']!=self.reader.manifest['entries'][sid]:
            raise ValueError('Event/source identity mismatch')
        teacher=self.teacher_reader.load(sid,device)
        validate_geometry(teacher,value['layout'],self.cfg.target_dim)
        value.update(teacher={k:teacher[k] for k in ['video','audio','video_valid','audio_valid']},
            events=tensors(label,self.names,self.labels.manifest['absence_weight'],device))
        return value


def data_identity(c):
    validate(c)
    value=whisper_identity(c)
    value['objective']='av1+event.05+video_feature.025+audio_feature.025+speaker_whisper'
    return value
