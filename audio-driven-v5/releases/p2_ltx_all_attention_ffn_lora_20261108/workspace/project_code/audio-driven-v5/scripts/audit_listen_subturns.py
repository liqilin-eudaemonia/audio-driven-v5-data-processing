"""Check partition coverage, normalized levels, sample counts, and audio time alignment."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from p3_listen.common import read, write
from p3_listen.media import command, uri
from p3_listen.subturns import OUT


def main():
    manifest=read(OUT/"manifest.json")
    audit=read(OUT/"subturn_audit.json")
    for p in audit:
        assert abs(p["original_seconds"]-p["retained_seconds"])<1e-6
        assert all(0<c["end"]-c["start"]<=10+1e-7 for c in p["cuts"])
        assert all(a["end"]==b["start"] for a,b in zip(p["cuts"],p["cuts"][1:]))
    audio={}
    for pair in manifest["pairs"]:
        for c in pair["clips"]:
            for role in ("speaker","listener"):
                audio[c[role]]=c["normalized_audio"][role]
    counts=[]
    for key,value in audio.items():
        before=uri(value["contract"]["source"],".wav")
        original=json.loads(command(["ffprobe","-v","error","-select_streams","a:0","-show_streams","-of","json",before]))["streams"][0]
        normalized=sf.info(value["path"])
        assert abs(float(original["duration"])-normalized.frames/normalized.samplerate)<2/48000
        assert abs(value["after"]["input_i"]+18)<=.31
        assert value["after"]["input_tp"]<=-.8
        counts.append(dict(source=key,samples=normalized.frames,rate=normalized.samplerate))
    # Compare three original/normalized waveforms spanning the source loudness range.
    order=sorted(audio,key=lambda k:audio[k]["before"]["input_i"])
    sync=[]
    for key in (order[0],order[len(order)//2],order[-1]):
        value=audio[key]
        result=subprocess.run(["ffmpeg","-v","error","-nostdin","-i",uri(value["contract"]["source"],".wav"),
            "-t","30","-ar","48000","-ac","1","-f","f32le","pipe:1"],capture_output=True,check=True)
        x=np.frombuffer(result.stdout,dtype=np.float32)
        y,_=sf.read(value["path"],frames=len(x),dtype="float32")
        assert len(x)==len(y)
        corr=correlate(y,x,mode="full",method="fft")
        center=len(x)-1
        local=corr[center-480:center+481]
        lag=int(np.argmax(local))-480
        coefficient=float(np.corrcoef(x,y)[0,1])
        assert abs(lag)<=1,(key,lag)
        sync.append(dict(source=key,lag_samples=lag,lag_ms=lag/48,correlation=coefficient))
    result=dict(pairs=len(manifest["pairs"]),parents=len(audit),subturns=manifest["counts"]["clips"],
                continuous_nonoverlapping=True,full_parent_time_preserved=True,max_seconds=10,
                normalized_sources=len(audio),lufs_range=[min(v["after"]["input_i"] for v in audio.values()),max(v["after"]["input_i"] for v in audio.values())],
                peak_max_dbtp=max(v["after"]["input_tp"] for v in audio.values()),sample_counts=counts,synchronization=sync,
                manual_semantic_and_listening_review="pending")
    write(OUT/"audio_boundary_validation.json",result)
    print({k:v for k,v in result.items() if k!="sample_counts"})


if __name__=="__main__":main()
