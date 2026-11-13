"""Frozen 24-row train/dev review with verified final crop and paired source audio.

This replays the current J source contract; it does not create training admission,
VAE caches, or pretend the historical caption was generated using a changed crop.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sqlite3
import subprocess
import sys

import av
import numpy as np
from PIL import Image
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_omni import read_audio, check_file, official_metadata
from listen_benchmark.j_omni_crop import effective_row, caption_crop_info, REVISION
from listen_benchmark.frame_crop import crop_original_frame_reused
from listen_visual_quality.measurement import SourceBytes
from listen_audio_postprocess.recipes import waveform_digest
from listen_audio_postprocess.bulk import official
from listen_visual_quality.io import file_hash

WORK=ROOT/'experiments/listen_j_face224_training_review_20261101'
OUT=ROOT/'data_previews/listen_benchmark_review/j_face224_training_20261101'
DB=ROOT/'experiments/listen_j_final_quality_selection_20261101/selection.sqlite'


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def draw():
    c=sqlite3.connect('file:'+str(DB)+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    rows=[dict(r) for r in c.execute("SELECT * FROM samples WHERE split IN ('train','dev') AND face_p10>=224 ORDER BY sid")]
    c.close()
    assert len(rows)==267070 and abs(sum(r['duration'] for r in rows)/3600-468.17825039930557)<1e-6
    picks=[]
    for split,n in [('train',20),('dev',4)]:
        picks.extend(random.Random(f'20261101:{split}:face224').sample([r for r in rows if r['split']==split],n))
    manifest=dict(seed='20261101:<split>:face224',population_count=len(rows),
        population_hours=sum(r['duration'] for r in rows)/3600,
        sampling='Uniform without replacement within each split, sorted sample IDs; 20 train + 4 dev. Dev oversampled for inspection.',
        subset_is_not_representative_of_split_proportions=True,selected=picks)
    frozen=WORK/'sampling.json'
    if frozen.exists():assert json.loads(frozen.read_text())==manifest
    else:write(frozen,manifest)
    return picks,manifest


def render_video(src,dest,fps=25):
    """Nearest source PTS to each 25 Hz grid; crop before bicubic resize."""
    start,end=src['source_start_s'],src['source_end_s'];duration=src['duration_s']
    count=math.ceil(duration*fps-1e-7);desired=[start+i/fps for i in range(count)]
    times=[];rgb_hash=hashlib.sha256();first_hash=None
    with SourceBytes(src['source']) as member, av.open(member) as container, av.open(str(dest/'video_only.mp4'),'w') as output:
        stream=container.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=1
        assert [stream.width,stream.height]==src['video_transform']['source_size']
        native_fps=float(stream.average_rate)
        container.seek(int(max(0,start-1)/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
        encoder=output.add_stream('libx264',rate=fps)
        encoder.width=encoder.height=512;encoder.pix_fmt='yuv420p'
        encoder.options={'crf':'15','preset':'fast','threads':'1'}
        previous=None;previous_t=None;last_cropped_pts=None;cropped=None;idx=0
        for frame in container.decode(video=0):
            if frame.pts is None:raise ValueError('Missing source PTS')
            t=float(frame.pts*frame.time_base)
            while idx<count and t+1e-8>=desired[idx]:
                chosen=previous if previous is not None and abs(previous_t-desired[idx])<=abs(t-desired[idx]) else frame
                ts=float(chosen.pts*chosen.time_base)
                if abs(ts-desired[idx])>1/native_fps+1e-5:raise ValueError('Source video gap')
                if chosen.pts!=last_cropped_pts:
                    cropped=crop_original_frame_reused(chosen,src['video_transform'])
                    last_cropped_pts=chosen.pts
                pixels=np.asarray(cropped);rgb_hash.update(pixels.tobytes())
                if idx==0:
                    cropped.save(dest/'reference.png');first_hash=hashlib.sha256(pixels.tobytes()).hexdigest()
                    Image.fromarray(chosen.to_ndarray(format='rgb24')).save(dest/'source_frame.jpg',quality=92)
                encoded=av.VideoFrame.from_ndarray(pixels,format='rgb24');encoded.pts=idx;encoded.time_base=Fraction(1,fps)
                for packet in encoder.encode(encoded):output.mux(packet)
                times.append(ts);idx+=1
            if idx==count:break
            previous,previous_t=frame,t
        if idx!=count:raise ValueError(f'Incomplete video: {idx}/{count}')
        for packet in encoder.encode():output.mux(packet)
    padded=8*math.ceil((count-1)/8)+1
    return dict(width=512,height=512,fps=fps,frames=count,native_fps=native_fps,
        source_frame_pts_s=times,desired_source_times_s=desired,
        maximum_grid_error_s=max(abs(a-b) for a,b in zip(times,desired)),
        preencode_rgb_sha256=rgb_hash.hexdigest(),first_frame_rgb_sha256=first_hash,
        preview_encoding='H264 yuv420p CRF15; lossy review copy, not VAE cache',
        existing_P2_layout_reference=dict(frames=padded,padding_s=padded/fps-duration,
            note='Existing 8n+1 padding formula shown only; no J training layout/cache committed'))


def save_audio(dest,name,wave,sr):
    wave=np.asarray(wave,dtype='<f4',order='C');assert wave.ndim==2 and wave.shape[1]==1 and np.isfinite(wave).all()
    raw=dest/f'{name}.f32';raw.write_bytes(wave.tobytes())
    buf=io.BytesIO();sf.write(buf,wave,sr,format='WAV',subtype='FLOAT')
    subprocess.run(['ffmpeg','-y','-v','error','-threads','1','-f','wav','-i','pipe:0',
        '-c:a','aac','-b:a','192k',str(dest/f'{name}.m4a')],input=buf.getvalue(),check=True,capture_output=True)
    return dict(raw=f'{name}.f32',download=f'{name}.m4a',sample_rate=sr,samples=len(wave),channels=1,
        full_silence=not bool(np.any(wave)),waveform_sha256=waveform_digest(wave,sr),
        raw_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),dtype='little-endian float32')


def build_one(arg):
    number,row=arg;sid=row['sid'];dest=OUT/'media'/sid;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'sample.json').exists():
        old=json.loads((dest/'sample.json').read_text());assert old['id']==sid
        return old
    with open(row['file'],'rb') as f:f.seek(row['offset']);raw=f.readline();rec=json.loads(raw)
    assert rec['sample_id']==sid and rec['split']==row['split'] and row['face_p10']>=224
    label_bytes=Path(rec['caption_file']).read_bytes();cap=json.loads(label_bytes)
    assert cap['sample_id']==sid and cap['caption']==rec['caption']
    if rec.get('caption_file_sha256'):assert hashlib.sha256(label_bytes).hexdigest()==rec['caption_file_sha256']
    decision=rec['quality'];measurement=json.loads(Path(decision['measurement_file']).read_text())
    original=cap.get('original_source',cap['source'])
    assert original==measurement['source']
    cfg=json.loads((ROOT/'experiments/listen_j_omni_20261030/config.json').read_text())
    effective,provenance=effective_row({'sample_id':sid,'source':original},cfg,decision=decision)
    src=effective['source'];assert provenance['revision']==REVISION and src['video_transform']['crop']==measurement['crop']['new']
    check_file(original['source_result'],original['source_result_sha256'])
    pair=json.loads(Path(original['source_result']).read_text())
    clip=pair['clips'][original['source_clip_index']]
    assert clip['sample_id']==sid and clip['listener_id']==src['listener_id'] and clip['speaker_id']==src['speaker_id']
    assert clip['listener_id']!=clip['speaker_id'] and clip['split']==row['split']
    assert abs(clip['start']-src['source_start_s'])<1e-7 and abs(clip['end']-src['source_end_s'])<1e-7
    assert clip['sources']['listener']==src['source']
    wave,sr,identity=read_audio(cap['audio_recipe'])
    assert cap['audio_recipe']['listener_id']==src['listener_id']
    assert identity['processed_waveform_sha256']==cap['input_evidence']['audio_identity']['processed_waveform_sha256']
    assert abs(len(wave)/sr-src['duration_s'])<=1/sr
    listener=save_audio(dest,'listener',wave,sr)
    a=clip['normalized_audio']['speaker'];assert a['contract']['source']==clip['sources']['speaker']
    assert Path(clip['sources']['speaker']['member']).stem==src['speaker_id']
    check_file(a['path'],a['sha256'])
    with sf.SoundFile(a['path']) as f:
        assert f.channels==1 and f.samplerate==sr==a['sample_rate']
        first,last=round(src['source_start_s']*sr),round(src['source_end_s']*sr)
        assert 0<=first<last<=len(f);f.seek(first);speaker_wave=f.read(last-first,dtype='float32',always_2d=True)
    assert len(speaker_wave)==len(wave)
    speaker=save_audio(dest,'speaker',speaker_wave,sr)
    video=render_video(src,dest)
    subprocess.run(['ffmpeg','-y','-v','error','-threads','1','-i',str(dest/'video_only.mp4'),
        '-i',str(dest/'listener.m4a'),'-map','0:v:0','-map','1:a:0','-c','copy',
        '-t',str(src['duration_s']),'-movflags','+faststart',str(dest/'listener.mp4')],check=True,capture_output=True)
    (dest/'video_only.mp4').unlink()
    metadata=official_metadata(src['source'],WORK/'review_metadata')
    assert metadata['provenance']['sha256']==cap['audio_recipe']['provenance']['sha256']
    off=official(metadata['raw'],cap['audio_recipe']['source']['first_sample']/sr,src['duration_s'])
    cleaned={k:[{q:v for q,v in x.items() if q in ('id','start','end')} for x in off[k]] for k in ('words','sentences','vad')}
    assert cleaned==cap['audio_recipe']['official_listener']
    origin=caption_crop_info(cap);caption_crop=cap['source']['video_transform']['crop']
    (dest/'caption_original.json').write_bytes(label_bytes)
    (dest/'statistics_original.json').write_bytes(raw)
    write(dest/'audio_recipe.json',cap['audio_recipe'])
    write(dest/'crop_provenance.json',dict(final=provenance,annotation=origin))
    item=dict(id=sid,number=number,split=row['split'],pair=src['pair'],group_id=src['group_id'],
        duration=src['duration_s'],source_start_s=src['source_start_s'],source_end_s=src['source_end_s'],
        listener_id=src['listener_id'],speaker_id=src['speaker_id'],video=video,
        media=f'media/{sid}/',listener_audio=listener,speaker_audio=speaker,
        face_p10=row['face_p10'],face_p50=row['face_p50'],face_valid_fraction=rec['face_sample_detection_fraction'],
        source_size=src['video_transform']['source_size'],final_crop=src['video_transform']['crop'],
        final_crop_revision=REVISION,annotation_crop=caption_crop,annotation_origin=origin['label_origin'],
        annotation_crop_differs_from_final=caption_crop!=src['video_transform']['crop'],
        original_crop=original['video_transform']['crop'],caption=cap['caption'],
        types=rec['counted_types'],behavior_quantities=rec['behavior_quantities'],nod=rec['nod'],
        official_listener=off,audio_support=rec['audio'],speaker_text=clip.get('text',''),
        caption_review_flags=cap.get('review_flags',[]),stat_status=rec.get('record_status'),
        training_admitted=bool(rec.get('training_admitted',False)),human_ground_truth=False,
        source_paths=dict(caption=rec['caption_file'],statistics=row['file'],statistics_offset=row['offset'],
            pair=src['source_result'],measurement=decision['measurement_file'],
            normalized_listener=src['normalized_audio']['path'],normalized_speaker=a['path']),
        audit=dict(final_crop_verified=True,caption_sha256=hashlib.sha256(label_bytes).hexdigest(),
            statistic_line_sha256=hashlib.sha256(raw).hexdigest(),speaker_source_sha256=a['sha256'],
            source_result_sha256=src['source_result_sha256'],listener_audio_identity=identity,
            official_metadata_and_recipe_equal=True,paired_audio_length_equal=True,
            reference_same_first_video_frame=True,preview_sha256=file_hash(dest/'listener.mp4')))
    write(dest/'sample.json',item)
    print(number,sid,'final_crop',item['final_crop'],'label_crop_differs',item['annotation_crop_differs_from_final'],flush=True)
    return item


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    if not 1<=args.workers<=2:raise ValueError('Review decoding is limited to 2 CPU workers')
    WORK.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    selected,manifest=draw()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        items=list(pool.map(build_one,enumerate(selected,1)))
    assert len(items)==24 and len({x['id'] for x in items})==24
    data=dict(schema='j-face224-final-crop-training-material-review-v1',samples=items,
        population_count=manifest['population_count'],population_hours=manifest['population_hours'],
        sampling=manifest['sampling'],seed=manifest['seed'],
        status='source_material_replay; J training manifest and latent cache not yet committed',
        policy=dict(crop=REVISION,video_fps=25,video_size=[512,512],
            specification_reference='experiments/p2_train50h_postprocessed_reselected_20261027/config.json:data',
            caption='Original Omni paragraph, unchanged; source crop explicitly retained',
            listener_audio='Existing official words/sentences/VAD union recipe, bit-exact float32 replay',
            speaker_audio='Paired normalized participant track, identical source time range, no listener gate',
            browser_audio='Raw float32 WebAudio; downloadable MP4/M4A are AAC preview copies',
            new_training_manifest_created=False,new_training_latents_created=False,test_included=False))
    write(OUT/'data.json',data)
    (OUT/'index.html').write_text((ROOT/'scripts/j_face224_training_review.html').read_text())
    fontpath=Path('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if fontpath.exists():
        from fontTools.ttLib import TTFont
        from fontTools.subset import Subsetter
        font=TTFont(str(fontpath),fontNumber=2);subset=Subsetter()
        subset.populate(text=(OUT/'index.html').read_text()+json.dumps(data,ensure_ascii=False))
        subset.subset(font);font.flavor='woff';font.save(OUT/'review.woff')
    write(WORK/'build_audit.json',dict(samples=len(items),train=sum(x['split']=='train' for x in items),
        dev=sum(x['split']=='dev' for x in items),all_final_crops_verified=True,
        legacy_annotation_count=sum(x['annotation_origin']=='legacy_original_crop_preserved' for x in items),
        annotation_crop_differs=sum(x['annotation_crop_differs_from_final'] for x in items),
        listener_silent=sum(x['listener_audio']['full_silence'] for x in items),
        all_audio_sha_and_metadata_verified=True,raw_labels_preserved=True,
        persistent_wav_exports=0,new_training_admission=False,new_training_latents=False,
        data_sha256=file_hash(OUT/'data.json')))
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
