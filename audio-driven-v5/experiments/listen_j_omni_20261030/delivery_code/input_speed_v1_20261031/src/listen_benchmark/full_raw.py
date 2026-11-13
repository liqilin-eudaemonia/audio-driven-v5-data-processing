"""Raw full-corpus annotations: explicit listener inputs, no event consolidation."""
import hashlib
import io
import math
from contextlib import contextmanager
from pathlib import Path

VERSION = 'listen-full-raw-v1'


class SourceBytes(io.RawIOBase):
    """Seekable original TAR member bytes, including members larger than 2 GiB."""
    def __init__(self, source):
        self.file=open(source['archive'],'rb')
        self.offset=source['offset'];self.size=source['byte_size'];self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self, offset, whence=0):
        if whence not in (0,1,2):raise ValueError('Invalid whence')
        position=(0 if whence==0 else self.position if whence==1 else self.size)+offset
        if position<0:raise ValueError('Negative member seek')
        self.position=position;return position
    def read(self, size=-1):
        remaining=max(0,self.size-self.position)
        count=remaining if size<0 else min(size,remaining)
        self.file.seek(self.offset+self.position);data=self.file.read(count)
        self.position+=len(data);return data
    def close(self):
        self.file.close();super().close()


@contextmanager
def open_source(source):
    import av
    from listen_labeling.full_source import source_uri
    try:container=av.open(source_uri(source),timeout=(30,60))
    except av.error.InvalidDataError:
        # FFmpeg subfile opening fails on some large members. Same bytes,
        # seek support and decoder; no re-encoding or replacement media.
        with SourceBytes(source) as member, av.open(member) as container:
            yield container,'bounded_original_bytes'
    else:
        with container:yield container,'ffmpeg_subfile'


def views(duration):
    result = [('full', 0., duration)]
    for i in range(math.ceil(max(0., duration - 3.) / 1.) + 1):
        start, end = float(i), min(duration, i + 3.)
        if (start, end) != (0., duration):
            result.append((f'w{i:03d}', start, end))
    return result


def compact(row, manifest, line_number, line_hash):
    from listen_labeling.seamless import integer_crop
    from p2_listen.contracts import safe_id
    safe_id(row['sample_id'])
    if row['split'] != 'train' or row['admission_pool'] not in ('primary', 'onset_reserve') or row['audio_final']['pool'] != 'keep_candidate':
        raise ValueError('Outside explicitly authorized train audio-passed pools')
    if row['listener_id'] == row['speaker_id'] or row['listener_index'] == row['speaker_index']:
        raise ValueError('Conflicting listener identity')
    source, audio = row['sources']['listener'], row['normalized_audio']['listener']
    if Path(source['member']).stem != row['listener_id'] or audio['contract']['source'] != source:
        raise ValueError('Listener audio/video identity mismatch')
    if any(type(source[k]) is not int or source[k] < 0 for k in ('offset', 'byte_size')) or not source['byte_size']:
        raise ValueError('Invalid source byte range')
    start, end = row['start'], row['end']
    if not all(type(t) in (int, float) and math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
        raise ValueError('Invalid source time range')
    if abs(end - start - row['duration_s']) > 2e-6:
        raise ValueError('Duration mismatch')
    crop, crop_error = None, None
    try:
        geometry = row['face_geometry']['listener']
        if geometry.get('rotation_degrees', 0) != 0:
            raise ValueError('Nonzero source rotation requires explicit review')
        crop = integer_crop(row.get('crop_proposals', {}).get('listener'), geometry['width'], geometry['height'])
    except (ValueError, KeyError, TypeError) as exc:
        crop_error = str(exc)
    return dict(sample_id=row['sample_id'], pair=row['pair'], split=row['split'],
        admission_pool=row['admission_pool'], listener_id=row['listener_id'], speaker_id=row['speaker_id'],
        listener_index=row['listener_index'], source=source,
        normalized_audio={k: audio[k] for k in ('path', 'sha256', 'sample_rate')},
        source_start_s=start, source_end_s=end, duration_s=end-start,
        video_transform=crop, crop_error=crop_error, crop_proposal=row.get('crop_proposals', {}).get('listener'),
        boundary_activity=row.get('boundary_activity', {}), boundary_review_flags=row.get('boundary_review_flags', []),
        source_manifest=str(manifest), source_line_1based=line_number, source_line_sha256=line_hash,
        training_admitted=False)


def validated_tail_endpoint(duration, fps, reported_end, last_inside, following_pts):
    """Resolve oversized packet duration only with adjacent decoded frame PTS.

    This never supplies missing frames or changes the selected input images.
    Leave every already-valid endpoint unchanged for provenance stability.
    """
    tolerance = 1 / fps + 1e-5
    if (reported_end is not None and reported_end - duration > tolerance
            and last_inside is not None and following_pts is not None
            and 0 <= last_inside < duration <= following_pts
            and 0 < following_pts - last_inside <= tolerance
            and following_pts - duration <= tolerance):
        return following_pts, dict(reported_end_s=reported_end,
                                   last_inside_frame_s=last_inside,
                                   following_frame_pts_s=following_pts,
                                   reason='oversized_duration_metadata; adjacent decoded PTS establishes complete support')
    return reported_end, None


def prepare(row, branch, *, fast_crop=False, native_dump=None):
    """Decode once per parent; use the benchmark's 8 FPS / bicubic / PCM path."""
    import io
    import av
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from listen_labeling.media import crop_resize
    from listen_labeling.full_source import source_uri
    rate = row['normalized_audio']['sample_rate']
    first, last = round(row['source_start_s'] * rate), round(row['source_end_s'] * rate)
    origin, duration = first / rate, (last-first) / rate
    if duration > 16 or duration <= 0:
        raise ValueError('Duration outside established 16-second budget; no truncation')
    result = dict(images=[], audio=None, duration_s=duration, meta={})
    if branch in ('audio', 'asr'):
        with sf.SoundFile(row['normalized_audio']['path']) as wav:
            if wav.channels != 1 or wav.samplerate != rate or not 0 <= first < last <= len(wav):
                raise ValueError('Independent listener waveform support/format mismatch')
            wav.seek(first); audio = wav.read(last-first, dtype='float32')
        if not np.isfinite(audio).all() or np.max(np.abs(audio), initial=0) > 1:
            raise ValueError('Invalid listener waveform')
        # Match the accepted benchmark's PCM16 transport before resampling.
        buf=io.BytesIO(); sf.write(buf, audio, rate, format='WAV', subtype='PCM_16')
        data=buf.getvalue(); audio,_=sf.read(io.BytesIO(data), dtype='float32')
        factor=math.gcd(rate,16000)
        if rate != 16000: audio=resample_poly(audio,16000//factor,rate//factor).astype(np.float32)
        result['audio']=audio
        result['meta']['audio']=dict(role='listener', isolated=True, path=row['normalized_audio']['path'],
            source_start_sample=first, source_end_sample=last, source_rate=rate,
            source_waveform_sha256_declared=row['normalized_audio']['sha256'], wav_sha256=hashlib.sha256(data).hexdigest(),
            encoder_rate=16000, encoder_samples=len(audio), resampling='scipy.resample_poly', pcm_transport='PCM_16')
    else:
        if row['video_transform'] is None: raise ValueError('Missing valid crop: '+str(row['crop_error']))
        from PIL import Image
        records=[]; next_grid=0; last_end=None; previous=None; last_inside=None; following_pts=None
        # Nod uses the declared clip duration, while Omni uses sample-rounded
        # audio duration. Share only identical end boundaries, including clips
        # whose last decoded frame happens to land on the boundary.
        native_images=[];native_times=[]
        native_enabled=native_dump is not None and row.get('duration_s',duration)==duration
        with open_source(row['source']) as (container,reader):
            stream=container.streams.video[0];stream.codec_context.thread_count=1
            if [stream.width,stream.height] != row['video_transform']['source_size']:
                raise ValueError('Decoded dimensions differ from crop source')
            fps=float(stream.average_rate)
            container.seek(int(origin/stream.time_base), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None: raise ValueError('Missing video PTS')
                source_t=float(frame.pts*frame.time_base); t=source_t-origin
                if previous is not None and source_t <= previous: raise ValueError('Nonmonotonic source PTS')
                previous=source_t
                if t >= duration:
                    following_pts=t
                    break
                last_inside=t
                last_end=t+(float(frame.duration*frame.time_base) if frame.duration else 1/fps)
                if t < 0:continue
                # The production nod runtime uses PyAV16; Omni uses PyAV17.
                # Their 10-bit RGB conversions differ. Reuse only the verified
                # 8-bit path; other formats retain independent legacy decoding.
                if native_enabled and frame.format.name!='yuv420p':
                    native_enabled=False;native_images.clear();native_times.clear()
                selected=t+1e-8 >= next_grid/8
                if not selected and not native_enabled:continue
                if fast_crop:
                    from .frame_crop import crop_original_frame
                    im=crop_original_frame(frame,row['video_transform'])
                else:im=crop_resize(frame.to_image(),row['video_transform'])
                if native_enabled:
                    native_images.append(np.asarray(im));native_times.append(t)
                if not selected:continue
                buf=io.BytesIO();im.save(buf,format='JPEG',quality=85);data=buf.getvalue()
                # Same JPEG pixels presented to the benchmark models.
                result['images'].append(Image.open(io.BytesIO(data)).convert('RGB'))
                records.append(dict(time_s=t, source_time_s=source_t, source_pts=frame.pts,
                                    size=[512,512], jpeg_sha256=hashlib.sha256(data).hexdigest()))
                next_grid=math.floor((t+1e-8)*8)+1
        last_end, tail_correction=validated_tail_endpoint(duration,fps,last_end,last_inside,following_pts)
        if not records or len(records)>128 or last_end is None or abs(last_end-duration)>1/fps+1e-5:
            raise ValueError('Incomplete visual interval; no silent frame reduction')
        result['meta']['video']=dict(frames=records, last_frame_s=records[-1]['time_s'], decoded_end_s=last_end,
            source_time_origin_s=origin, requested_fps=8, video_transform=row['video_transform'], source=row['source'], reader=reader)
        if tail_correction is not None:
            result['meta']['video']['tail_duration_metadata_correction']=tail_correction
        if native_enabled:
            from .shared_visual_decode import save_native
            save_native(native_dump,row,native_images,native_times,reader)
    return result
