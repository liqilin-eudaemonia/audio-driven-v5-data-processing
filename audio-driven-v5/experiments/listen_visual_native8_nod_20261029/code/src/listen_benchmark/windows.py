"""Overlapping windows with exact clip offsets; no inferred event boundaries."""
import copy
import math


def window_rows(rows, size=3.0, stride=1.0):
    if not 0 < stride <= size:
        raise ValueError("Window stride must be positive and no larger than size")
    output=[]
    for source in rows:
        duration=source["duration_s"]
        count=math.ceil(max(0.0,duration-size)/stride)+1
        for index in range(count):
            start=index*stride;end=min(duration,start+size)
            row=copy.deepcopy(source)
            row.update(sample_id=source["sample_id"]+f"__w{index:03d}",parent_id=source["sample_id"],
                       source_duration_s=duration,duration_s=end-start,
                       window=dict(start_s=start,end_s=end,parent_duration_s=duration))
            output.append(row)
    return output


def slice_media(media, start, end, fps):
    if not 0 <= start < end <= media["duration_s"]+1e-8:
        raise ValueError("Window outside source clip")
    if abs(start*fps-round(start*fps))>1e-8:
        raise ValueError("Window start must align to the native sampling grid")
    result=dict(media,duration_s=end-start,content=[])
    meta=copy.deepcopy(media["meta"])
    meta["window"]=dict(start_s=start,end_s=end,parent_duration_s=media["duration_s"],
                         local_time_origin="window_start",audio_rate=16000)
    if media["images"]:
        first=round(start*fps)
        last=min(len(media["images"]),math.ceil(end*fps-1e-8))
        result["images"]=media["images"][first:last]
        frames=copy.deepcopy(meta["video"]["frames"][first:last])
        if not frames:raise ValueError("Window has no sampled frame")
        for frame in frames:
            frame["parent_time_s"]=frame["time_s"]
            frame["time_s"]-=start
        meta["video"].update(frames=frames,last_frame_s=frames[-1]["time_s"],decoded_end_s=end-start)
    if media["audio"] is not None:
        first=round(start*16000);last=min(len(media["audio"]),round(end*16000))
        result["audio"]=media["audio"][first:last]
        meta["audio"].update(encoder_samples=last-first,window_sample_bounds=[first,last])
    result["meta"]=meta
    return result
