#!/usr/bin/env python3
"""Audit a packaged subset without executing models or modifying shared caches."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from listen_benchmark.backchannel_face_stats import CATEGORIES, atomic_json, read, sha
from listen_benchmark.train_subset import audit_catalog, audit_eval_manifests
from p2_listen.contracts import audit_splits, input_identity, observation, target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset");parser.add_argument("--workers",type=int,default=16)
    args=parser.parse_args();root=Path(args.dataset).resolve()
    complete=read(root/"COMPLETE.json")
    for name,digest in complete["files"].items():
        if sha(root/name)!=digest:raise ValueError(f"Output hash mismatch: {name}")
    contract=read(root/"dataset_contract.json")
    if sha(Path(__file__).resolve().parents[1]/"src/listen_benchmark/train_subset.py")!=contract["selector_code_sha256"]:
        raise ValueError("Selector code changed")
    obs=read(root/"manifests/observations.json");targets=read(root/"manifests/targets.json")
    rows=list(csv.DictReader((root/"samples.csv").open()))
    from listen_benchmark.backchannel_face_stats import loads
    refs=[loads(line) for line in (root/"dataset.jsonl").open()]
    ids=[o["sample_id"] for o in obs]
    if len(set(ids))!=len(ids) or any(ids!=[r["sample_id"] for r in items] for items in [targets,rows,refs]):
        raise ValueError("Manifest identity/order mismatch")
    if ids!=(root/"sample_ids.txt").read_text().splitlines():raise ValueError("ID text differs")
    splits=audit_splits(obs)
    if splits!={"train":len(ids),"dev":0,"test":0}:raise ValueError("Non-train sample")
    spans=collections.defaultdict(list)
    for r in rows:
        if r['split']!='train' or int(r['label_issue_flag']) or float(r['face_valid_fraction'])<.98:
            raise ValueError("Selected row violates quality/split constraints")
        floor=224 if r['selection_reason'].startswith('core') else 160 if r['selection_reason']=='supplement:head_shake' else 192
        if float(r['face_short_side_p10'])<floor:raise ValueError("Face floor violated")
        spans[r['pair']].append((float(r['source_start_s']),float(r['source_end_s'])))
    max_per_pair=read(root/'selection_policy.json')['max_per_pair']
    for intervals in spans.values():
        intervals.sort()
        if len(intervals)>max_per_pair or any(b[0]<a[1]-1e-6 for a,b in zip(intervals,intervals[1:])):
            raise ValueError("Within-train source overlap or pair cap violation")
    def one(values):
        o,t,r,ref=values; sid=o['sample_id']
        observation(o,verify=False);target(t,o,verify=False)
        label_path=root/'labels/canonical'/(sid+'.json')
        if sha(label_path)!=ref['canonical_label_sha256']:raise ValueError("Canonical hash mismatch")
        label=read(label_path)
        counts=collections.Counter(e['type'] for e in label['events'])
        if any(counts[c]!=int(r[c+'_count']) for c in CATEGORIES):raise ValueError("Category count mismatch")
        for part,info in ref['cache'].items():
            marker=read(Path(info['path'])/'complete.json')
            if marker['identity']!=info['identity'] or marker['payload_sha256']!=info['payload_sha256']:
                raise ValueError("Cache receipt mismatch")
            if marker['contract']['input']!=input_identity(o):raise ValueError("Cache input mismatch")
            if part in ('teacher','vae') and marker['contract']['target']!=t:raise ValueError("Cache target mismatch")
            if part=='events' and marker['contract']['label']!=label:raise ValueError("Event cache/label mismatch")
            if not (Path(info['path'])/'payload.pt').is_file():raise ValueError("Missing payload")
        return counts
    counts=collections.Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n,c in enumerate(pool.map(one,zip(obs,targets,rows,refs)),1):
            counts.update(c)
            if n%5000==0:print(f"audited {n}/{len(ids)}",flush=True)
    catalog=read(contract['catalog'])
    if sha(contract['catalog'])!=contract['catalog_sha256']:raise ValueError("Catalog changed")
    pairs,_=audit_catalog(catalog)
    eval_files=audit_eval_manifests(Path(contract['catalog']).parent/'manifests',pairs,rows)
    if eval_files!=read(root/'split_audit.json')['eval_manifests']:raise ValueError("Eval manifest identities changed")
    report=dict(samples=len(ids),split_counts=splits,duplicate_ids=0,within_pair_overlap=0,
        cross_split_identity_overlap=0,canonical_category_counts=dict(counts),
        actual_hours=sum(o['requested_duration_s'] for o in obs)/3600,
        source_unique_hours=sum(b-a for values in spans.values() for a,b in values)/3600,
        all_output_hashes_verified=True,all_selected_canonical_hashes_verified=True,
        all_selected_cache_input_target_label_identities_verified=True,
        payload_hash_verification="Inherited from completed upstream full audit; no expensive duplicate payload scan",
        no_gpu_used=True,training_started=False)
    atomic_json(root/'validation.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
