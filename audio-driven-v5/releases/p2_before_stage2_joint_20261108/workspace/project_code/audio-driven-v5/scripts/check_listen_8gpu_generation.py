"""Eight independent real AV generation checks, one on each physical GPU."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from p3_listen.common import read,write,sha


def main():
    root=ROOT/"experiments/p3_fixed_text_v2_subturns"
    out=root/"validation_8gpu/generation"
    out.mkdir(parents=True,exist_ok=True)
    dev=read(root/"s0_selection.json")["dev"][:2]
    rows={r["sample_id"]:r for r in read(root/"windows.json")["windows"]}
    c0=root/"runs/s0_C0_checked/step_000002.pt"
    a=root/"runs/s0_A_checked/step_000002.pt"
    swap=out/"swap_speaker.wav"
    donor=root/"media/dev"/dev[1]/"speaker.wav"
    if rows[dev[1]]["duration_s"]<rows[dev[0]]["duration_s"]:
        raise ValueError("Swap donor is too short")
    subprocess.run(["ffmpeg","-v","error","-nostdin","-y","-i",str(donor),"-t",str(rows[dev[0]]["duration_s"]),
                    "-ar","48000","-ac","1","-c:a","pcm_s24le",str(swap)],check=True)
    jobs=[dict(gpu=0,mode="text",sample=dev[0],seed=137,checkpoint=None),
          dict(gpu=1,mode="pred",sample=dev[0],seed=251,checkpoint=c0),
          dict(gpu=2,mode="oracle",sample=dev[0],seed=137,checkpoint=c0),
          dict(gpu=3,mode="prefix",sample=dev[0],seed=137,checkpoint=c0),
          dict(gpu=4,mode="audio-swap",sample=dev[0],seed=137,checkpoint=c0),
          dict(gpu=5,mode="pred",sample=dev[1],seed=137,checkpoint=c0),
          dict(gpu=6,mode="pred",sample=dev[0],seed=137,checkpoint=a),
          dict(gpu=7,mode="oracle",sample=dev[0],seed=137,checkpoint=a)]
    def execute(job):
        name=f"gpu{job['gpu']}_{job['mode']}"
        destination=out/name
        cmd=[sys.executable,str(ROOT/"scripts/p3_listen.py"),"generate","--split","dev", "--sample-id",job["sample"],
             "--mode",job["mode"],"--seed",str(job["seed"]),"--gpu",str(job["gpu"]),"--output",str(destination)]
        if job["checkpoint"]:cmd += ["--checkpoint",str(job["checkpoint"])]
        if job["mode"]=="audio-swap":cmd += ["--swap-audio",str(swap)]
        begin=time.monotonic()
        with (out/(name+".log")).open("w") as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        record=dict(gpu=job["gpu"],mode=job["mode"],sample_id=job["sample"],seed=job["seed"],
                    checkpoint=str(job["checkpoint"]),returncode=result.returncode,seconds=time.monotonic()-begin,output=str(destination))
        if result.returncode==0:
            completed=read(destination/"completed.json")
            if completed["video_sha256"]!=sha(destination/"generated.mp4"):
                raise ValueError("Generated file hash changed")
            record["decoded"]=completed["decoded"]
        return record
    begin=time.monotonic()
    results=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(execute,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            write(out/"progress.json",dict(completed=len(results),total=8,results=results))
            print(json.dumps(result),flush=True)
    report=dict(gpus=list(range(8)),jobs=8,successes=sum(r["returncode"]==0 for r in results),
                wall_seconds=time.monotonic()-begin,results=results,engineering_only=True,quality_evaluated=False)
    write(out/"completed.json",report)
    if report["successes"]!=8:raise SystemExit(1)


if __name__=="__main__":main()
