import sys,time,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,'/private/liqi_workspace/project_code/audio-driven-v5/src')
from p2_j_teacher.media import frame_indices,selected_frames as original
from p2_listen.contracts import read,digest
import torch
def selected_frames(item, override, size, reference=False, threads=4):
    from p2_j_cache.data import decode_frames, SourceVideoGap
    source = item['source']; indices = frame_indices(source['duration_s'])
    policy = None if override is None else override['frames']
    if override is not None and override['source_digest'] != digest(source):
        raise ValueError('Reviewed frame override source mismatch')
    if reference:
        frames, times = decode_frames(source, gap_overrides=policy)
        return np.stack([np.asarray(Image.fromarray(frames[i]).resize(size)) for i in indices]), [times[i] for i in indices]
    # Decode the same source frames as the native 25 fps reader, crop only frames
    # selected by the 8 fps teacher. No compressed review MP4 is used.
    import av
    from listen_visual_quality.measurement import SourceBytes
    from listen_benchmark.frame_crop import crop_original_frame_roi
    desired = [source['source_start_s'] + i / 25 for i in indices]
    result = []; times = []; previous = None; previous_t = None; last_pts = None; rgb = None
    with SourceBytes(source['source']) as member, av.open(member) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'; stream.codec_context.thread_count = threads
        if [stream.width, stream.height] != source['video_transform']['source_size']:
            raise ValueError('Source video dimensions changed')
        fps = float(stream.average_rate)
        container.seek(int(max(0, source['source_start_s'] - 1/fps - 1e-6) / float(stream.time_base)),
                       stream=stream, backward=True, any_frame=False)
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError('Missing source PTS')
            t = float(frame.pts * frame.time_base)
            while len(result) < len(indices) and t + 1e-8 >= desired[len(result)]:
                wanted = desired[len(result)]
                if previous is None and t > wanted + 1e-8:
                    return original(item,override,size,reference=False)
                chosen = previous if previous is not None and abs(previous_t-wanted) <= abs(t-wanted) else frame
                actual = float(chosen.pts * chosen.time_base)
                if abs(actual-wanted) > 1/fps + 1e-5:
                    approved = (policy or {}).get(str(indices[len(result)]))
                    record = dict(target_pts_s=wanted, previous_pts_s=previous_t, next_pts_s=t, selected_pts_s=actual)
                    if (approved is None or previous_t is None or set(approved) != set(record)
                            or any(abs(approved[k]-v)>1e-8 for k,v in record.items())):
                        raise SourceVideoGap(item['sample_id'] + ': teacher source frame gap')
                if chosen.pts != last_pts:
                    rgb = np.asarray(crop_original_frame_roi(chosen, source['video_transform']).resize(size))
                    last_pts = chosen.pts
                result.append(rgb); times.append(actual)
            if len(result) == len(indices):
                break
            previous, previous_t = frame, t
        while len(result) < len(indices) and previous is not None and abs(previous_t-desired[len(result)]) <= 1/fps+1e-5:
            if previous.pts != last_pts:
                rgb = np.asarray(crop_original_frame_roi(previous, source['video_transform']).resize(size))
                last_pts = previous.pts
            result.append(rgb); times.append(previous_t)
    if len(result) != len(indices):
        raise ValueError('Incomplete teacher source video: '+item['sample_id'])
    return np.stack(result), times

if __name__=='__main__':
 root=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_j_train500h_cache_20261101')
 out=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_j_teacher500h_node8_20261102')
 ids=read(out/'pilot_ids.json')
 m=read(root/'manifest.json');torch.set_num_threads(1);rows=[]
 for sid in ids:
  item=read(root/'samples'/(sid+'.json'));expected=torch.load(out/'pilot_v3/cache'/sid/'payload.pt',weights_only=True)['audit']
  threads=2
  t=time.monotonic();f,pts=selected_frames(item,m.get('video_sampling_overrides',{}).get(sid),(224,224),threads=threads);dt=time.monotonic()-t
  assert hashlib.sha256(f.tobytes()).hexdigest()==expected['teacher_rgb_sha256'],sid
  assert pts==expected['source_pts'],sid
  row=dict(sample_id=sid,seconds=dt,threads=threads,pixels_pts_exact=True);rows.append(row);print(json.dumps(row),flush=True)
 (out/'decode_auto2_all32.json').write_text(json.dumps(rows,indent=2)+'\n')
