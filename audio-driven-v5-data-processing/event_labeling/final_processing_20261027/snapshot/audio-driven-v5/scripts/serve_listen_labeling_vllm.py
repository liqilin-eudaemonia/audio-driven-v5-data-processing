#!/usr/bin/env python3
"""vLLM frontend adapter preserving exact PTS for already sampled JPEG videos.

Only repository-local runtime adaptation: no shared dependency files are changed.
"""
import math,runpy

def exact_video_metadata(metadata,times,duration):
    if not isinstance(times,list) or len(times)!=metadata['total_num_frames'] or not times:
        raise ValueError('One exact source timestamp required per JPEG frame')
    if not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
        raise ValueError('Invalid source clip duration')
    if any(type(t) not in (int,float) or not math.isfinite(t) or not 0<=t<duration for t in times):
        raise ValueError('Video timestamp outside source clip')
    indices=[round(t*1000000) for t in times]
    if any(b<=a for a,b in zip(indices,indices[1:])):raise ValueError('Non-increasing exact frame timestamps')
    # vLLM Qwen3-VL consumes frames_indices/fps to construct temporal tokens.
    # Use a microsecond time base, while keeping the actual sampled JPEG count.
    return dict(metadata,fps=1000000.0,frames_indices=indices,total_num_frames=math.ceil(duration*1000000),duration=duration,do_sample_frames=False)

def main():
    from vllm.multimodal.media.video import VideoMediaIO
    original=VideoMediaIO.load_base64
    def load(self,media_type,data):
        frames,metadata=original(self,media_type,data)
        times=self.kwargs.get('frame_times_s')
        if times is not None:
            if media_type.lower()!='video/jpeg':raise ValueError('Exact PTS adapter requires JPEG sequence')
            if frames.shape[1:3]!=(512,512):raise ValueError('Expected listener ROI at exactly 512x512')
            metadata=exact_video_metadata(metadata,times,self.kwargs['clip_duration_s'])
        return frames,metadata
    VideoMediaIO.load_base64=load
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
if __name__=='__main__':main()
