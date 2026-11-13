"""Production exporter: two FFmpeg processes per window; lossless WAV slicing."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from p3_listen import media
from p3_listen.common import digest,write


def export(root,row):
    import av
    import soundfile as sf
    p=media.paths(root,row)
    if p['marker'].exists():
        media.verify_media(root,row);return
    if not row.get('normalized_audio'):return media.export(root,row)
    p['marker'].parent.mkdir(parents=True,exist_ok=True)
    start_us=int((Decimal(str(row['start']))*1000000).to_integral_value(rounding=ROUND_HALF_UP))
    start_sample=(start_us*48000+500000)//1000000
    samples=round(row['duration_s']*48000)
    for role,key in (('speaker','speaker_audio'),('listener','target_audio')):
        source=row['normalized_audio'][role]
        if media.sha(source['path'])!=source['sha256']:raise ValueError('Normalized source audio changed')
        with sf.SoundFile(source['path']) as audio:
            if audio.samplerate!=48000 or audio.channels!=1 or audio.subtype!='PCM_24':
                raise ValueError('Expected normalized mono 48 kHz PCM24 input')
            audio.seek(start_sample);wave=audio.read(samples,dtype='int32')
        if len(wave)!=samples:raise ValueError('Original audio too short or duration mismatch')
        tmp=p[key].with_name(p[key].stem+'.pending.wav')
        sf.write(tmp,wave,48000,subtype='PCM_24');tmp.replace(p[key])
    b=row['listener_crop']
    vf=(f"setpts=PTS-STARTPTS,crop={b['size']}:{b['size']}:{b['x']}:{b['y']},"
        f"scale={row['width']}:{row['height']},fps={row['fps']}:eof_action=pass,setsar=1")
    tmp=p['target_video'].with_name('listener.pending.mp4')
    media.command(['ffmpeg','-v','error','-nostdin','-y','-threads','2','-ss',str(row['start']),
        '-i',media.uri(row['sources']['listener'],'.mp4'),'-vf',vf,'-frames:v',str(row['frames']),
        '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','16','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(tmp)])
    tmp.replace(p['target_video'])
    with av.open(str(p['target_video'])) as video:
        stream=video.streams.video[0];stream.thread_count=1
        if (stream.frames,stream.height,stream.width)!=(row['frames'],row['height'],row['width']):
            raise ValueError('Extracted video geometry mismatch')
    # Keep FFmpeg's original color conversion for the reference PNG. PyAV's
    # default conversion is not pixel-identical for every source color space.
    media.command(['ffmpeg','-v','error','-nostdin','-y','-threads','2','-i',
                   str(p['target_video']),'-frames:v','1',str(p['listener_image'])])
    write(p['marker'],dict(row_sha256=digest(row),files={k:media.sha(v) for k,v in p.items() if k!='marker'},
        source='original archive MP4 and separate WAV',frames=row['frames'],duration_s=row['duration_s']))
