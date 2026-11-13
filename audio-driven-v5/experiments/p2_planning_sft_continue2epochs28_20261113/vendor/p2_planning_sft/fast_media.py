"""Equivalent first-frame reads: verified legacy PNGs, then bounded CPU decoding.

Never changes labels, audio, RGB conversion, or nearest-source-PTS selection.
The original reference reader stays unchanged for independent comparison.
"""
import hashlib
import math
import os
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from p2_listen.contracts import read, digest, sha
from upright_lora_media import resolve_example
from listen_response_sft.data import require, first_frame
from listen_benchmark.j_omni import check_file


def decode_first_frame(src, threads=4):
    import av
    from listen_visual_quality.measurement import SourceBytes
    from listen_benchmark.frame_crop import crop_original_frame_reused
    start=src['source_start_s']
    with SourceBytes(src['source']) as member, av.open(member) as container:
        stream=container.streams.video[0]
        stream.thread_type='AUTO';stream.codec_context.thread_count=threads
        require([stream.width,stream.height]==src['video_transform']['source_size'],'Source size mismatch')
        rate=float(stream.average_rate)
        require(math.isfinite(rate) and rate>0,'Invalid source FPS')
        # Seek to a preceding keyframe while retaining the frame before start.
        seek=max(0,start-1/rate-1e-6)
        container.seek(int(seek/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
        previous=None;previous_t=None
        for frame in container.decode(video=0):
            require(frame.pts is not None,'Missing source PTS')
            t=float(frame.pts*frame.time_base)
            if t+1e-8>=start:
                # A malformed seek must not omit a potentially nearer predecessor.
                if previous is None and seek>0 and t>start+1e-8:
                    return first_frame(src)
                chosen=previous if previous is not None and abs(previous_t-start)<=abs(t-start) else frame
                actual=float(chosen.pts*chosen.time_base)
                if abs(actual-start)>1/rate+1e-5:return first_frame(src)
                return crop_original_frame_reused(chosen,src['video_transform']),actual
            previous=frame;previous_t=t
    return first_frame(src)


class FastInputs:
    def __init__(self, config):
        require(config.get('version') in ('verified-png-auto4-v1','verified-png-isolated-auto4-v2'),'Unknown input acceleration policy')
        self.isolated=config['version']=='verified-png-isolated-auto4-v2'
        self.timeout=config.get('decode_timeout_s',30)
        require(5<=self.timeout<=60,'Invalid decoder timeout')
        self.log_dir=config.get('decode_log_dir')
        self._decoder=None
        self.threads=config.get('decode_threads',4)
        require(type(self.threads) is int and 1<=self.threads<=4,'Decode thread budget exceeds checked policy')
        seal=Path(config['legacy_seal']);require(sha(seal)==config['legacy_seal_sha256'],'Legacy seal changed')
        self.seal=read(seal);self.root=Path(self.seal['root']);self.index=Path(self.seal['index'])
        st=self.index.stat()
        require((st.st_size,st.st_mtime_ns)==(self.seal['index_bytes'],self.seal['index_mtime_ns']),'Legacy index changed')
        self._pid=None;self._conn=None
    def __getstate__(self):
        return dict(self.__dict__,_pid=None,_conn=None,_decoder=None)
    def legacy_frame(self, row, resolved):
        if self._pid!=os.getpid():
            if self._conn is not None:self._conn.close()
            self._conn=sqlite3.connect(self.index.resolve().as_uri()+'?mode=ro',uri=True);self._pid=os.getpid()
        found=self._conn.execute('select digest from frames where sid=?',(row['sample_id'],)).fetchone()
        if found is None:return None
        sid=row['sample_id'];item=read(self.root/'samples'/(sid+'.json'))
        require(digest(item)==found[0],'Legacy frozen source changed')
        # A known ID alone is insufficient: match the complete effective source.
        if item['source']!=resolved['source'] or item['split']!=row['split']:return None
        audit=read(self.root/'media_audit'/(sid+'.json'))
        require(audit['sample_id']==sid and audit['split']==row['split'],'Legacy audit identity mismatch')
        require(audit['final_crop']==resolved['source']['video_transform']['crop'],'Legacy crop mismatch')
        require(audit['reference_from_first_frame'] and audit['source_video_used'],'Invalid legacy reference origin')
        require(audit['crop_revision']==resolved['crop_provenance']['revision'],'Legacy crop revision mismatch')
        with Image.open(self.root/'references'/(sid+'.png')) as image:image=image.convert('RGB')
        require(image.size==(512,512),'Invalid legacy image dimensions')
        require(hashlib.sha256(np.asarray(image).tobytes()).hexdigest()==audit['first_rgb_sha256'],'Legacy RGB changed')
        return image,audit['source_frame_pts_s'][0]
    def __call__(self,row,crop_config):
        import soundfile as sf
        r=resolve_example(row,crop_config);src=r['source'];a=r['speaker_audio']
        check_file(a['path'],a['sha256'])
        with sf.SoundFile(a['path']) as f:
            sr=f.samplerate
            require(sr==a['sample_rate'] and f.channels==1,'Wrong speaker audio format')
            first,last=round(src['source_start_s']*sr),round(src['source_end_s']*sr)
            require(0<=first<last<=len(f),'Speaker audio bounds invalid')
            f.seek(first);wave=f.read(last-first,dtype='float32')
        require(len(wave)==last-first and np.isfinite(wave).all(),'Incomplete speaker audio')
        require(abs(len(wave)/sr-row['duration_s'])<=1/sr,'Speaker duration mismatch')
        cached=self.legacy_frame(row,r)
        if cached is None:
            if self.isolated:
                from .isolated_decode import IsolatedDecoder
                if self._decoder is None:self._decoder=IsolatedDecoder(self.threads,self.timeout,self.log_dir)
                image,pts=self._decoder(src);origin='isolated_decode'
            else:image,pts=decode_first_frame(src,self.threads);origin='auto_decode'
        else:image,pts=cached;origin='verified_legacy_png'
        return dict(sample_id=row['sample_id'],split=row['split'],
            inputs=dict(speaker_audio=wave,speaker_sample_rate=sr,listener_first_frame=image),target_text=r['target_text'],
            audit=dict(speaker_audio_path=a['path'],speaker_audio_sha256=a['sha256'],speaker_first_sample=first,speaker_last_sample=last,
                source_frame_time_s=pts,crop_provenance=r['crop_provenance'],annotation_crop_differs=r['annotation_crop_differs'],
                future_listener_media_in_inputs=False,first_frame_read_path=origin))
