"""Deterministic development jobs and blind-review sheets; no automatic quality claims."""
import csv
import random
from pathlib import Path

from .common import digest, read, resolve, sha, write_once


def matrix(cfg, args):
    if args.split != "dev":
        raise ValueError("Diagnostic matrix is dev only")
    if not args.checkpoint:
        raise ValueError("Matrix requires a selected C0/A checkpoint")
    root = resolve(cfg["output"])
    ids = read(root / "s0_selection.json")["dev"]
    rows = {r["sample_id"]:r for r in read(root / "windows.json")["windows"]}
    jobs = []
    for mode, subset in (("text",ids),("pred",ids),("oracle",ids),("prefix",ids[:8]),("audio-swap",ids[:8])):
        for i,sid in enumerate(subset):
            for seed in cfg["generation"]["seeds"]:
                donor = None
                if mode == "audio-swap":
                    # Donor must cover the duration; the execution step must trim to exact samples.
                    choices=[x for x in ids if rows[x]["pair"] != rows[sid]["pair"] and rows[x]["duration_s"]>=rows[sid]["duration_s"]]
                    donor=choices[0] if choices else None
                jobs.append(dict(job_id=f"J{len(jobs)+1:04d}",sample_id=sid,split="dev",mode=mode,seed=seed,
                                 checkpoint=str(args.checkpoint.resolve()) if mode!="text" else None,
                                 swap_donor=donor, swap_duration_s=rows[sid]["duration_s"] if mode=="audio-swap" else None,
                                 status="needs_equal_duration_donor" if mode=="audio-swap" and donor is None else "pending"))
    out=args.output or root/"evaluation"
    write_once(Path(out)/"jobs.json",dict(config_sha256=digest(cfg),checkpoint_sha256=sha(args.checkpoint),jobs=jobs))
    random.Random(cfg["seed"]).shuffle(jobs)
    blind=[]
    mapping={}
    for i,job in enumerate(jobs):
        review_id=f"R{i+1:04d}"
        mapping[review_id]=job["job_id"]
        blind.append(dict(review_id=review_id,appropriateness="",timing="",visual_naturalness="",audio_naturalness="",
                          silence_false_positive="",face_deformation="",speaker_copy="",notes=""))
    Path(out).mkdir(parents=True,exist_ok=True)
    sheet=Path(out)/"blind_review.csv"
    if not sheet.exists():
        with sheet.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(blind[0]))
            writer.writeheader();writer.writerows(blind)
    write_once(Path(out)/"review_key.json",mapping)
    print(dict(jobs=len(jobs),output=str(out)),flush=True)
