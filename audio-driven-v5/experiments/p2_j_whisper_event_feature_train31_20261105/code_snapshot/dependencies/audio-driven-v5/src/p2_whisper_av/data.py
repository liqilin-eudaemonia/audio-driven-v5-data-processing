"""Reuse baseline train order, frozen H, Gemma and AV targets verbatim."""
from pathlib import Path

from p2_listen.contracts import digest, read, sha
from .cache import Reader


class Dataset:
    def __init__(self, c):
        from p2_clip_events.labels import LabelReader
        from .compat import cache_reader_class
        CacheReader = cache_reader_class()
        self.reader = CacheReader(c['j_cache'])
        labels = LabelReader(c['clip_labels'])
        try:
            ids = labels.ids()
            if (digest(ids) != labels.manifest['sample_ids_sha256'] or
                    ids != self.reader.ids('train')):
                raise ValueError('Whisper dataset must match the full baseline train order')
        finally:
            labels.conn.close()
        self.whisper = Reader(c['whisper_cache'],require_complete=True,source=c['j_cache'])
        encoder=self.whisper.protocol['encoder']
        if (Path(encoder['path']).resolve() != Path(c['whisper']['model_path']).resolve()
                or encoder['input_dim'] != c['whisper']['input_dim']):
            raise ValueError('Whisper cache encoder differs from the experiment configuration')
        if ids != self.whisper.protocol['sample_ids']:
            raise ValueError('Whisper full cache membership / train order differs')
        self.rows = [dict(sample_id=sid) for sid in ids]

    def sample(self, index, device='cpu'):
        sid = self.rows[index]['sample_id']
        cached = self.reader.training_sample(sid,'guided',device)
        # No event labels, Feature teachers, listener PCM, or raw video are read.
        return dict(sample_id=sid,condition=cached['condition'],layout=cached['layout'],
            text=cached['text'],av=dict(av=cached['av'],support=cached['support']),
            teacher=None,whisper=self.whisper.load(sid,device))


def data_identity(c):
    from p2_clip_events.backend import data_identity as baseline_identity
    folder = Path(c['whisper_cache']);p = read(folder/'protocol.json');complete = read(folder/'COMPLETE.json')
    if (complete.get('success') is not True or complete.get('samples') != p['samples'] or
            complete.get('protocol_sha256') != sha(folder/'protocol.json') or
            p['source_manifest_sha256'] != sha(Path(c['j_cache'])/'manifest.json') or
            Path(p['source']).resolve() != Path(c['j_cache']).resolve()):
        raise ValueError('Whisper cache completion / source identity mismatch')
    return dict(baseline=baseline_identity(c),whisper_protocol=sha(folder/'protocol.json'),
                whisper_complete=sha(folder/'COMPLETE.json'),whisper_encoder=p['encoder'])
