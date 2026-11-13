import threading
import torch
from p2_whisper_av.data import Dataset as WhisperDataset,data_identity as whisper_identity
from p2_whisper_event_feature.data import CompatibleTeacherReader,to_device
from p2_clip_events.labels import LabelReader,tensors
from p2_clip_events.model import classes
from p2_matrix16.config import event_enabled,feature_enabled
from .config import validate,pure_av_view


class Dataset(WhisperDataset):
    def __init__(self,c):
        validate(c);super().__init__(pure_av_view(c));self.spec=c
        self.labels=LabelReader(c['clip_labels']);self.owner=threading.get_ident();self.local=threading.local()
        self.names=classes(c['event_supervision']);self.teacher_reader=None
        from p2_clip_events.backend import data_config,data_identity
        from p2_listen.config import extractor_config
        data_identity(c);self.cfg=extractor_config(data_config(c))
        if feature_enabled(c):
            self.teacher_reader=CompatibleTeacherReader(c['teacher_cache'])
            if not {r['sample_id'] for r in self.rows}<=self.teacher_reader.ids:raise ValueError('Teacher missing training members')

    def event_targets(self,sid,device):
        if not event_enabled(self.spec):
            return {k:dict(y=torch.zeros(1,device=device),known=torch.zeros(1,dtype=torch.bool,device=device),weight=torch.zeros(1,device=device)) for k in self.names}
        labels=self.labels
        if threading.get_ident()!=self.owner:
            if not hasattr(self.local,'reader'):self.local.reader=LabelReader(self.spec['clip_labels'],verify=False)
            labels=self.local.reader
        row=labels.row(sid)
        if row['source_digest']!=self.reader.manifest['entries'][sid]:raise ValueError('Event source differs')
        return tensors(row,self.names,self.labels.manifest['absence_weight'],device)

    def sample(self,index,device='cpu'):
        value=super().sample(index,device);sid=value['sample_id']
        value['events']=self.event_targets(sid,device)
        if self.teacher_reader is not None:
            from p2_j_teacher.core import validate_geometry
            teacher=self.teacher_reader.load(sid,device);validate_geometry(teacher,value['layout'],self.cfg.target_dim)
            value['teacher']={k:teacher[k] for k in ['video','audio','video_valid','audio_valid']}
        return value


def data_identity(c):
    validate(c);result=whisper_identity(c);result['objective']=c['loss']+'+speaker_whisper'
    return result
