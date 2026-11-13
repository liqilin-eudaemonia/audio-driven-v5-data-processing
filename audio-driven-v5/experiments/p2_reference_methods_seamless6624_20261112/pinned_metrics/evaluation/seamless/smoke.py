"""Real lazy GT, lossless roundtrip and audio/first-frame checks; not model scores."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .bind import load_references
from .reader import decode_reference,load_conditioning,load_listener_audio
from ..video.paired_metrics.media import decode
from ..responsenet_pairs.prepare import write_json,write_jsonl


def lossless(frames,path):
    import av
    with av.open(str(path),'w') as container:
        stream=container.add_stream('ffv1',rate=25)
        stream.width=stream.height=512;stream.pix_fmt='bgr0'
        stream.options={'threads':'1'}
        for i,frame in enumerate(frames):
            video=av.VideoFrame.from_ndarray(frame,format='rgb24');video.pts=i;video.time_base=Fraction(1,25)
            for packet in stream.encode(video):container.mux(packet)
        for packet in stream.encode():container.mux(packet)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    references=load_references(a.prepared)
    selected=[]
    for fmt in ('original_caption_verbatim','original_caption_strip_chat_prefix_v1'):
        candidates=[r for r in references if r['caption_format']==fmt]
        if not candidates:raise ValueError(f'No {fmt} samples')
        selected.append(min(candidates,key=lambda r:(r['duration_s'],r['id'])))
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=[];audits=[]
    for r in selected:
        print('Decoding real source:',r['id'],flush=True)
        gt,fps,audit=decode_reference(r['gt_reference'])
        inputs,input_audit=load_conditioning(r['gt_reference'])
        np.testing.assert_array_equal(gt[0],np.asarray(inputs['listener_first_frame']))
        wave,sr,audio_audit=load_listener_audio(r['gt_reference'])
        path=a.output/(r['id']+'.self_check_only.mkv');lossless(gt,path)
        roundtrip,rate,_=decode(path)
        assert rate==fps
        np.testing.assert_array_equal(roundtrip,gt)
        manifest.append(dict(id=r['id'],dataset='seamless',gt_reference=r['gt_reference'],
            generated=str(path.resolve()),identity='listener',listening_intervals=r['listening_intervals'],self_check_only=True))
        audits.append(dict(id=r['id'],caption_format=r['caption_format'],frames=len(gt),fps=fps,
            source_decode=audit,first_frame_matches_existing_reader=True,lossless_roundtrip_exact=True,
            speaker_samples=len(inputs['speaker_audio']),speaker_sample_rate=inputs['speaker_sample_rate'],
            listener_samples=len(wave),listener_sample_rate=sr,listener_audio_audit=audio_audit,
            input_audit=input_audit))
    write_jsonl(a.output/'self_check_pairs.jsonl',manifest)
    write_json(a.output/'media_audit.json',dict(dataset='seamless',self_check_only=True,
        generated_model_results=False,whole_dataset_media_decoded=False,records=audits))
    status=subprocess.run([sys.executable,'-m','evaluation.video.paired_metrics.run',
        '--manifest',str(a.output/'self_check_pairs.jsonl'),'--assets',str(a.assets),
        '--output',str(a.output/'metrics'),'--device','cpu','--threads','2','--batch-size','2','--face-padding','0']).returncode
    if status==0:
        summary=json.loads((a.output/'metrics/summary.json').read_text());m=summary['metrics']
        assert summary['datasets']==['seamless']
        assert m['psnr']=='inf' and m['ssim']==1 and m['lpips']<1e-7
        assert m['fid']<1e-6 and m['fvd']<1e-6 and m['emo_acc']==1 and m['delta_resp']<1e-7
        assert m['emo_div_generated']==m['emo_div_gt']
        write_json(a.output/'assertions.json',dict(passed=True,self_check_only=True,dataset='seamless',metric_count=8))
    raise SystemExit(status)


if __name__=='__main__':main()
