#!/usr/bin/env python3
"""Prepare test-only annotation candidates; never claim unmeasured class coverage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from listen_benchmark.train_subset import audit_catalog,identities,Selection
from listen_labeling.seamless import integer_crop
from p2_listen.contracts import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument("--catalog",required=True);p.add_argument("--train",required=True);p.add_argument("--output",required=True);p.add_argument("--candidates",type=int,default=512)
    a=p.parse_args();catalog=Path(a.catalog).resolve();train=Path(a.train).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    pairs,_=audit_catalog(json.loads(catalog.read_text()))
    train_rows=json.loads((train/"selection.json").read_text());train_keys=set().union(*(identities(pairs[r["pair"]]) for r in train_rows))
    pool=[];source_files={};counts=Counter()
    for path in sorted((catalog.parent/"manifests").glob("test*keep_candidate.jsonl")):
        source_files[str(path)]=sha(path)
        with path.open("rb") as f:
            for n,line in enumerate(f,1):
                row=json.loads(line);counts["source_rows"]+=1;pair=pairs[row["pair"]]
                if row["split"]!="test" or pair["split"]!="test" or pair["group_id"]!=row["group_id"] or train_keys & identities(pair):
                    raise ValueError("Test/train split identity mismatch")
                g=row.get("face_geometry",{}).get("listener",{});face=g.get("face_short_side_px",{}).get("p10",0)
                crop=row.get("crop_proposals",{}).get("listener")
                if face<160 or g.get("valid_fraction",0)<.98 or not crop or crop.get("size",0)<384:continue
                if not 2<=row["duration_s"]<=11.95:continue
                try:transform=integer_crop(crop,g["width"],g["height"])
                except (KeyError,ValueError):continue
                r=dict(sample_id=row["sample_id"],pair=row["pair"],group_id=row["group_id"],split="test",
                    source_start_s=row["start"],source_end_s=row["end"],duration_s=row["duration_s"],
                    face_short_side_p10=face,face_short_side_p50=g["face_short_side_px"]["p50"],face_valid_fraction=g["valid_fraction"],
                    official_listener_feedback_count=len(row.get("listener_events",[])),
                    source_manifest=str(path),source_line_1based=n,source_line_sha256=hashlib.sha256(line).hexdigest(),
                    video_transform=transform,source=row,seven_class_coverage_verified=False,training_admitted=False)
                pool.append(r)
    pool.sort(key=lambda r:(-r["face_short_side_p10"],-min(r["official_listener_feedback_count"],3),r["sample_id"]))
    # This is an annotation shortlist, not the final event-ranked test selection.
    chosen=Selection(8)
    for r in pool:
        if len(chosen.rows)>=a.candidates:break
        # Selection only needs category counts to track its own counters.
        from listen_benchmark.backchannel_face_stats import CATEGORIES
        temp=dict(r,**{c+"_count":0 for c in CATEGORIES})
        chosen.add(temp,"annotation_candidate_face_rank_only")
    with (out/"eligible_candidates.jsonl").open("w") as f:
        for r in pool:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    # Category zeros above are private Selection bookkeeping, not observations.
    shortlist=[{k:v for k,v in r.items() if k not in {c+"_count" for c in CATEGORIES}} for r in chosen.rows]
    atomic_json(out/"annotation_shortlist.json",shortlist)
    atomic_json(out/"status.json",dict(status="awaiting_test_annotation",requested_final_samples=24,eligible_candidates=len(pool),
        annotation_shortlist=len(chosen.rows),selection_not_final=True,seven_class_coverage_verified=False,
        train_identity_overlap=0,catalog_sha256=sha(catalog),train_selection_sha256=sha(train/"selection.json"),source_files=source_files,
        note="No existing seven-class test annotations. Class counts are not measured here. No canonical labels or training admission generated."))
    print(json.dumps(dict(eligible=len(pool),shortlist=len(chosen.rows),output=str(out))))


if __name__=="__main__":main()
