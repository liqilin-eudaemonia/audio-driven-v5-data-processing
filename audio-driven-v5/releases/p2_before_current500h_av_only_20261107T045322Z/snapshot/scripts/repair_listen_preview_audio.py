"""Mux complete independent audio after frame-bounded video encoding.

FFmpeg 4.2's -frames:v stops muxing audio early. Preserve encoded video and mux
the complete independent audio in a second pass without -frames:v.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from p3_listen.common import read,write
from p3_listen.media import command
from p3_listen.subturns import OUT


def repair(workers=32):
    manifest=read(OUT/"manifest.json")
    def one(c):
        folder=OUT/c["path"]
        marker=read(folder/"complete.json")
        if marker.get("audio_mux_revision")==2:return
        for role in ("speaker","listener"):
            video=folder/(role+"_512.mp4")
            tmp=folder/(role+"_512.remux.mp4")
            command(["ffmpeg","-v","error","-nostdin","-y","-threads","1","-i",str(video),
                     "-i",str(folder/(role+".m4a")),"-map","0:v:0","-map","1:a:0","-map_metadata","-1",
                     "-c","copy","-t",str(c["duration"]),"-movflags","+faststart",str(tmp)])
            tmp.replace(video)
        video=folder/"faces.mp4";tmp=folder/"faces.remux.mp4"
        command(["ffmpeg","-v","error","-nostdin","-y","-threads","1","-i",str(video),
                 "-i",str(folder/"speaker.m4a"),"-i",str(folder/"listener.m4a"),"-filter_complex_threads","1",
                 "-filter_complex","[1:a][2:a]amix=inputs=2:normalize=0,aresample=48000,apad=pad_len=239,alimiter=limit=0.891250938:level=false:attack=5:release=50,atrim=start_sample=239,asetpts=PTS-STARTPTS[a]",
                 "-map","0:v:0","-map","[a]","-map_metadata","-1","-c:v","copy","-c:a","aac","-b:a","160k",
                 "-t",str(c["duration"]),"-movflags","+faststart",str(tmp)])
        tmp.replace(video)
        marker["audio_mux_revision"]=2
        write(folder/"complete.json",marker)
    rows=[c for p in manifest["pairs"] for c in p["clips"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,c) for c in rows]
        for i,f in enumerate(as_completed(futures)):
            f.result()
            if i%100==0 or i==len(rows)-1:print(dict(audio_remuxed=i+1,total=len(rows)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=32)
    repair(p.parse_args().workers)
