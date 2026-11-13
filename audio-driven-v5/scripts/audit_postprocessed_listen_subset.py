#!/usr/bin/env python3
"""Verify subset identities and actual CPU training reads, then publish COMPLETE."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import random
import multiprocessing as mp
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from p2_listen.contracts import read,sha,digest,atomic_json,input_identity
from p2_listen.config import load,activate
from listen_benchmark.postprocessed_subset import event_statistics,CATEGORIES

_AUDIT_ONE=None
def _audit_item(values):
    return _AUDIT_ONE(values)


def main():
    p=argparse.ArgumentParser();p.add_argument("root");p.add_argument("--workers",type=int,default=24);p.add_argument("--read-samples",type=int,default=32)
    args=p.parse_args();root=Path(args.root).resolve()
    if not (root/"export_complete.json").exists():raise ValueError("Export not complete")
    cfg=load(root/"config.json");activate(cfg)
    from p2_listen.annotations import validate,validate_contract
    from listen_audio_postprocess.overlay import sample_contract,load_training_sample
    from p2_listen.features import identity
    from listen_benchmark.audit_identity_snapshot import IdentitySnapshot
    snapshot=IdentitySnapshot(cfg)
    obs=read(root/"manifests/observations.json");targets=read(root/"manifests/targets.json");rows=read(root/"selection.json")
    refs=[json.loads(line) for line in (root/"dataset.jsonl").open()]
    ids=[o["sample_id"] for o in obs]
    if len(set(ids))!=len(ids) or any(ids!=[v["sample_id"] for v in x] for x in (targets,rows,refs)):
        raise ValueError("ID/order mismatch")
    if ids!=(root/"sample_ids.txt").read_text().splitlines():raise ValueError("ID text mismatch")
    policy=read(root/"selection_policy.json")
    floors=policy["floors"];max_per_pair=policy["max_per_pair"]
    if floors["core_face_p10"]<160 or max_per_pair<1:raise ValueError("Invalid selection policy")
    spans=defaultdict(list);coverage=Counter()
    for r in rows:
        floor=floors["core_face_p10"] if r["selection_reason"].startswith("core") else floors["head_shake_face_p10"] if r["selection_reason"]=="supplement:head_shake" else floors["supplement_face_p10"]
        if r["split"]!="train" or r["label_issue_flag"] or r["face_valid_fraction"]<.98 or r["face_short_side_p10"]<floor or r["non_smile_event_count"]<1:
            raise ValueError("Selected row fails quality criteria")
        spans[r["pair"]].append((r["source_start_s"],r["source_end_s"]))
        coverage.update(c for c in CATEGORIES if r[c+"_count"])
    for values in spans.values():
        values.sort()
        if len(values)>max_per_pair or any(b[0]<a[1]-1e-6 for a,b in zip(values,values[1:])):raise ValueError("Duplicate source support")
    if set(coverage)!=set(CATEGORIES):raise ValueError("Missing seven-class coverage")
    def one(values):
        o,t,r,ref=values;sid=o["sample_id"]
        if o["experiment_split"]!="train" or o["official_split"]!="train":raise ValueError("Wrong split")
        _,_,_,recipe,label,overlay_identity=sample_contract(cfg,o,t)
        if sha(root/"labels/canonical"/(sid+".json"))!=ref["canonical_sha256"] or sha(root/"audio_postprocess/recipes"/(sid+".json"))!=ref["recipe_sha256"]:
            raise ValueError("Exported label/recipe changed")
        stats=event_statistics(label,r["duration_s"])
        if any(r[k]!=v for k,v in stats.items()):raise ValueError("Postprocessed counts differ")
        for part,info in ref["cache"].items():
            marker=read(Path(info["path"])/"complete.json")
            expected=overlay_identity if part=="audio_postprocess" else identity(cfg,part,o,t if part=="vae" else None,label=label if part=="events" else None)
            if marker["identity"]!=digest(expected) or marker["identity"]!=info["identity"] or marker["payload_sha256"]!=info["payload_sha256"]:
                raise ValueError(f"Stale cache {sid}/{part}")
        return sid
    global _AUDIT_ONE
    _AUDIT_ONE=one
    # CPU-only contract checking; fork avoids serial JSON work under one GIL.
    with mp.get_context('fork').Pool(args.workers) as workers:
        for n,_ in enumerate(workers.imap_unordered(_audit_item,zip(obs,targets,rows,refs),chunksize=16),1):
            if n%2000==0:
                atomic_json(root/'audit_progress.json',dict(audited=n,total=len(ids)))
                print(json.dumps(dict(audited=n,total=len(ids))),flush=True)
    chosen=set(random.Random(20261027).sample(range(len(obs)),min(args.read_samples,len(obs))))
    for c in CATEGORIES:chosen.add(next(i for i,r in enumerate(rows) if r[c+"_count"]))
    for i in sorted(chosen):
        load_training_sample(cfg,obs[i],targets[i],"cpu")
    snapshot_check=snapshot.verify_and_restore()
    report=dict(status="passed",samples=len(ids),actual_duration_s=sum(o["requested_duration_s"] for o in obs),
        per_class_samples=dict(coverage),split="train",cross_split_identity_overlap=0,within_pair_overlap=0,
        all_cache_contracts_verified=True,all_label_recipe_hashes_verified=True,training_reads=len(chosen),training_started=False,
        payload_hash_verification="Full upstream audit inherited; additional selected CPU loads verify payloads",immutable_identity_snapshot=snapshot_check)
    atomic_json(root/"validation.json",report)
    names=["config.json","selection_policy.json","summary.json","split_audit.json","samples.csv","sample_ids.txt","dataset.jsonl","integrated_labels.jsonl","integrated_sources.json","manifests/observations.json","manifests/targets.json","dataset_contract.json","validation.json"]
    atomic_json(root/"COMPLETE.json",dict(schema="postprocessed-large-face-subset-v1",status="passed",samples=len(ids),hours=report["actual_duration_s"]/3600,files={n:sha(root/n) for n in names},training_started=False))
    atomic_json(root/"progress.json",dict(phase="complete",exported=len(ids),total=len(ids),validated=True))
    atomic_json(root/"audit_progress.json",dict(audited=len(ids),total=len(ids),validated=True))
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
