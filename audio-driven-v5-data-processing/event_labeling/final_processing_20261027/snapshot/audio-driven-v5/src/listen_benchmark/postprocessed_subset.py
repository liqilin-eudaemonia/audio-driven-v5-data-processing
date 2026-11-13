"""Select a train subset from completed audio overlays without recaching media.

Reuse native face measurements, but recompute ranking from current admitted
canonical events. Disjoint support pieces of one event never inflate its count.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

from .backchannel_face_stats import CATEGORIES, read, sha, loads, union_seconds
from .train_subset import audit_catalog, audit_eval_manifests, select_rows, summarize
from p2_listen.contracts import atomic_json, digest

VERSION = "postprocessed-large-face-subset-v1"


def event_statistics(label, duration):
    counts = Counter()
    units = defaultdict(set)
    intervals = defaultdict(list)
    types = defaultdict(set)
    for event in label["events"]:
        category = event["type"]
        if category not in CATEGORIES:
            raise ValueError("Unknown canonical category")
        source = event["source"]
        key = source.get("event_id") or event["event_id"]
        units[category].add(key)
        types[key].add(category)
        a, b = event["start"], event["end"]
        if not 0 <= a < b <= duration + .0011:
            raise ValueError("Invalid canonical interval")
        intervals["all"].append((a,b))
        intervals["audio" if category == "listener_vocalization" else "visual"].append((a,b))
    counts.update({c:len(units[c]) for c in CATEGORIES})
    total = len(types)
    return dict(**{c+"_count":counts[c] for c in CATEGORIES},
        category_counts_json=json.dumps(dict(counts),sort_keys=True),
        category_count=sum(v>0 for v in counts.values()), event_count=total,
        canonical_interval_count=len(label["events"]),
        visual_event_count=len(set().union(*(units[c] for c in CATEGORIES if c != "listener_vocalization"))),
        audio_event_count=counts["listener_vocalization"], other_modality_event_count=0,
        non_smile_event_count=sum(bool(v-{"smile"}) for v in types.values()),
        valid_timed_event_count=total, invalid_timed_event_count=0,
        events_per_10s=total*10/duration, timed_events_per_10s=total*10/duration,
        event_coverage_s=union_seconds(intervals["all"]),
        visual_coverage_s=union_seconds(intervals["visual"]), audio_coverage_s=union_seconds(intervals["audio"]),
        event_coverage_fraction=union_seconds(intervals["all"])/duration,
        uncategorized_event_count=0)


def refresh_one(args):
    row, full = args
    path = full/"view/labels/canonical"/(row["sample_id"]+".json")
    body = path.read_bytes(); label = loads(body)
    if label["sample_id"] != row["sample_id"] or label["taxonomy"] != list(CATEGORIES):
        raise ValueError("Canonical identity mismatch")
    if not label["training_admitted"]:
        return None
    result = dict(row, pre_postprocess_event_count=row["event_count"],
                  pre_postprocess_audio_event_count=row["audio_event_count"],
                  current_canonical_sha256=hashlib.sha256(body).hexdigest())
    result.update(event_statistics(label, row["duration_s"]))
    return result if result["non_smile_event_count"] else None


def refresh(args):
    output=Path(args.output).resolve(); full=Path(args.full).resolve(); stats=Path(args.stats).resolve()
    output.mkdir(parents=True,exist_ok=True)
    complete=read(full/"COMPLETE.json")
    if complete["status"]!="passed" or sha(full/"audit.json")!=complete["audit_sha256"]:
        raise ValueError("Full postprocess audit is not complete")
    signature=dict(full_completion_sha256=sha(full/"COMPLETE.json"),
        stats_completion_sha256=sha(stats/"COMPLETE.json"), selector_sha256=sha(__file__))
    marker=output/"pool_complete.json"
    if marker.exists():
        saved=read(marker)
        if saved["signature"]!=signature or sha(output/"eligible_pool.jsonl")!=saved["sha256"]:
            raise ValueError("Selection pool provenance changed")
        return [loads(line) for line in (output/"eligible_pool.jsonl").open("rb")]
    if getattr(args,"pool_source",None):
        previous=Path(args.pool_source).resolve();saved=read(previous/"pool_complete.json")
        for key in ("full_completion_sha256","stats_completion_sha256"):
            if saved["signature"][key]!=signature[key]:raise ValueError("Imported pool refers to different completed data")
        source=previous/"eligible_pool.jsonl"
        if sha(source)!=saved["sha256"]:raise ValueError("Imported pool changed")
        (output/"eligible_pool.jsonl").symlink_to(source)
        atomic_json(marker,dict(signature=signature,sha256=saved["sha256"],samples=saved["samples"],
            imported_from=str(previous),source_marker_sha256=sha(previous/"pool_complete.json"),
            note="Reuse unchanged postprocessed counts and face measurements; selected labels revalidated on export"))
        return [loads(line) for line in source.open("rb")]
    if sha(stats/"stats.sqlite")!=read(stats/"COMPLETE.json")["files"]["stats.sqlite"]:
        raise ValueError("Face statistics changed")
    with sqlite3.connect(f"file:{stats}/stats.sqlite?mode=ro",uri=True) as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute("SELECT * FROM samples WHERE split='train' AND label_issue_flag=0 AND face_valid_fraction>=.98 AND face_short_side_p10>=160 AND non_smile_event_count>=1")]
    result=[]
    with ThreadPoolExecutor(max_workers=args.workers) as workers, (output/"eligible_pool.jsonl").open("w") as stream:
        for n,row in enumerate(workers.map(refresh_one,((r,full) for r in rows)),1):
            if row is not None:
                result.append(row);stream.write(json.dumps(row,ensure_ascii=False)+"\n")
            if n%10000==0:
                progress=dict(phase="refresh_postprocessed_counts",read=n,total=len(rows),eligible=len(result))
                atomic_json(output/"progress.json",progress);print(json.dumps(progress),flush=True)
    atomic_json(marker,dict(signature=signature,sha256=sha(output/"eligible_pool.jsonl"),samples=len(result),upstream_candidates=len(rows)))
    return result


def export_one(args):
    row,obs,target,full,output,base=args
    sid=row["sample_id"];view=full/"view"
    if obs["sample_id"]!=sid or target["sample_id"]!=sid or obs["official_split"]!="train" or obs["experiment_split"]!="train":
        raise ValueError("Selected sample split mismatch")
    if abs(obs["requested_duration_s"]-row["duration_s"])>.0011:
        raise ValueError("Selected duration mismatch")
    hashes={}
    for folder in ("labels/canonical","audio_postprocess/recipes"):
        source=view/folder/(sid+".json");body=source.read_bytes()
        hashes[folder]=hashlib.sha256(body).hexdigest()
        if folder=="labels/canonical" and hashes[folder]!=row["current_canonical_sha256"]:
            raise ValueError("Selected label changed during export")
        (output/folder/(sid+".json")).write_bytes(body)
    label=loads((output/"labels/canonical"/(sid+".json")).read_bytes())
    recipe=read(output/"audio_postprocess/recipes"/(sid+".json"))
    if label["provenance"]["audio_recipe_sha256"]!=digest(recipe) or recipe["source"]["sha256"]!=target["media_hashes"]["target_audio"]:
        raise ValueError("Label/audio recipe/target hash mismatch")
    refs={}
    for part,root in (("input",view),("vae",base),("audio_postprocess",view),("events",view)):
        folder=root/"cache"/part/sid;marker=read(folder/"complete.json")
        if digest(marker["contract"])!=marker["identity"] or marker["contract"]["part"]!=part:
            raise ValueError("Invalid cache marker")
        if part=="audio_postprocess" and marker["contract"]["recipe"]!=recipe:
            raise ValueError("Stale audio cache")
        payload=folder/"payload.pt"
        if not payload.is_file() or payload.stat().st_size==0:
            raise ValueError("Cache payload missing")
        refs[part]=dict(path=str(folder),identity=marker["identity"],payload_sha256=marker["payload_sha256"])
    for name in (obs["speaker_audio"],obs["listener_image"],target["target_audio"],target["target_video"]):
        if not Path(name).is_file():raise FileNotFoundError(name)
    return dict(sample_id=sid,canonical_sha256=hashes["labels/canonical"],recipe_sha256=hashes["audio_postprocess/recipes"],
                source_line_sha256=row["source_line_sha256"],cache=refs)


def export_integrated(full,output,selected,archives):
    wanted={r["sample_id"] for r in selected};found=set();sources={}
    prefixes={k[:24] for k in archives}
    paths=sorted(p for p in (full/"integrated").glob("*.jsonl") if p.name[:24] in prefixes)
    def extract(path):
        ready=read(full/"ready"/path.name.replace(".jsonl",".json"))
        body=path.read_bytes();checksum=hashlib.sha256(body).hexdigest()
        if checksum!=ready["integrated_sha256"]:raise ValueError("Integrated shard changed")
        kept=[]
        for line in body.splitlines(keepends=True):
            value=loads(line);sid=value["sample_id"]
            if sid not in wanted:continue
            if not value.get("audio_postprocess",{}).get("visual_events_unchanged"):
                raise ValueError("Not postprocessed integrated labels")
            kept.append((sid,line))
        return path,checksum,kept
    with (output/"integrated_labels.jsonl").open("wb") as stream:
        with ThreadPoolExecutor(max_workers=16) as workers:
            # Bound outstanding results so large original-description records do not accumulate.
            for begin in range(0,len(paths),64):
                for path,checksum,kept in workers.map(extract,paths[begin:begin+64]):
                    sources[str(path)]=checksum
                    for sid,line in kept:
                        if sid in found:raise ValueError("Duplicate integrated sample")
                        stream.write(line);found.add(sid)
                if begin%1024==0:print(json.dumps(dict(phase="export_integrated",shards=begin,total_shards=len(paths),samples=len(found))),flush=True)
    if found!=wanted:raise ValueError("Missing postprocessed integrated labels")
    atomic_json(output/"integrated_sources.json",sources)


def build(args):
    output=Path(args.output).resolve();full=Path(args.full).resolve();catalog=Path(args.catalog).resolve()
    if (output/"COMPLETE.json").exists():raise ValueError("Completed output already exists")
    pool=refresh(args)
    pairs,split_counts=audit_catalog(read(catalog))
    for row in pool:
        pair=pairs[row["pair"]]
        if pair["split"]!="train" or pair["group_id"]!=row["group_id"] or row["listener_id"] not in (pair["a"]["id"],pair["b"]["id"]):
            raise ValueError("Catalog/statistics mismatch")
    selected,policy=select_rows(pool,hours=args.hours,max_per_pair=getattr(args,"max_per_pair",8),
                               core_face_floor=getattr(args,"core_face_floor",224))
    del pool
    policy.update(event_counts="Distinct admitted source events in CURRENT canonical labels; support fragments count once",source_full=str(full))
    summary=summarize(selected)
    summary['below_224_face_p10']=summary['below_core_face_floor']
    summary['core_face_floor']=policy['floors']['core_face_p10']
    summary['below_core_face_floor']=sum(r['face_short_side_p10']<summary['core_face_floor'] for r in selected)
    atomic_json(output/"selection_policy.json",policy);atomic_json(output/"summary.json",summary)
    atomic_json(output/"selection.json",selected)
    print(json.dumps(dict(phase="selected",summary=summary)),flush=True)
    audit=dict(catalog_sha256=sha(catalog),retained_catalog_pairs=split_counts,identity_intersections=0,
        identity_keys=["pair","connected_group","session","version+participant_without_AB_suffix"],
        eval_manifests=audit_eval_manifests(catalog.parent/"manifests",pairs,selected))
    atomic_json(output/"split_audit.json",audit)
    for name in ("manifests","labels/canonical","audio_postprocess/recipes","cache"):
        (output/name).mkdir(parents=True,exist_ok=True)
    view=full/"view";cfg=read(view/"config.json");base=Path(cfg["data"]["audio_postprocess"]["base_output"])
    # Preserve the upstream policy and recipe bytes; subset membership is a separate contract.
    (output/"audio_postprocess/policy_manifest.json").write_bytes((view/"audio_postprocess/policy_manifest.json").read_bytes())
    cfg=copy.deepcopy(cfg);cfg["output"]=str(output);atomic_json(output/"config.json",cfg)
    for name in ("input","text","audio_postprocess","events"):
        dest=output/"cache"/name;source=view/"cache"/name
        if dest.is_symlink():
            if dest.resolve()!=source.resolve():raise ValueError("Unexpected cache symlink")
        else:dest.symlink_to(source,target_is_directory=True)
    db=sqlite3.connect(f"file:{full}/scope.sqlite?mode=ro",uri=True);db.row_factory=sqlite3.Row
    archive_keys={r["path"]:r["key"] for r in db.execute("SELECT * FROM archives")}
    observations=[];targets=[];jobs=[];archives=set()
    for row in selected:
        record=db.execute("SELECT obs,target,archive FROM samples WHERE sid=?",(row["sample_id"],)).fetchone()
        if record is None:raise ValueError("Selected sample missing from full scope")
        obs=loads(record["obs"]);target=loads(record["target"])
        observations.append(obs);targets.append(target);archives.add(archive_keys[record["archive"]])
        jobs.append((row,obs,target,full,output,base))
    db.close()
    actual=sum(o["requested_duration_s"] for o in observations)
    summary["exact_audio_duration_s"]=actual;summary["exact_audio_hours"]=actual/3600
    atomic_json(output/"summary.json",summary)
    atomic_json(output/"manifests/observations.json",observations);atomic_json(output/"manifests/targets.json",targets)
    with ThreadPoolExecutor(max_workers=args.workers) as workers,(output/"dataset.jsonl").open("w") as stream:
        for n,reference in enumerate(workers.map(export_one,jobs),1):
            stream.write(json.dumps(reference,ensure_ascii=False)+"\n")
            if n%1000==0:
                progress=dict(phase="export_cache_references",exported=n,total=len(selected))
                atomic_json(output/"progress.json",progress);print(json.dumps(progress),flush=True)
    atomic_json(output/"references_complete.json",dict(samples=len(selected),dataset_sha256=sha(output/"dataset.jsonl")))
    finish(args,selected,archives,actual)


def finish(args,selected=None,archives=None,actual=None):
    output=Path(args.output).resolve();full=Path(args.full).resolve()
    if selected is None:
        selected=read(output/"selection.json")
        refs=[loads(line) for line in (output/"dataset.jsonl").open("rb")]
        if [r["sample_id"] for r in refs]!=[r["sample_id"] for r in selected]:raise ValueError("References export is incomplete")
        with sqlite3.connect(f"file:{full}/scope.sqlite?mode=ro",uri=True) as db:
            keys=dict(db.execute("SELECT path,key FROM archives"));archives=set()
            for row in selected:
                path=db.execute("SELECT archive FROM samples WHERE sid=?",(row["sample_id"],)).fetchone()[0];archives.add(keys[path])
        actual=sum(o["requested_duration_s"] for o in read(output/"manifests/observations.json"))
    export_integrated(full,output,selected,archives)
    with (output/"samples.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    (output/"sample_ids.txt").write_text("".join(r["sample_id"]+"\n" for r in selected))
    atomic_json(output/"dataset_contract.json",dict(schema=VERSION,samples=len(selected),duration_s=actual,
        official_split="train",experiment_split="train",full_postprocess=str(full),
        full_completion_sha256=sha(full/"COMPLETE.json"),full_audit_sha256=sha(full/"audit.json"),
        selector_sha256=sha(__file__),labels_are_weak=True,training_started=False,
        waveform_storage="Original listener audio plus recipes; no new WAV",cache_storage="Reuse completed full overlay and original video caches",
        feature_loss_enabled=False,upstream_policy_scope="Policy manifest covers the full upstream pool; subset membership is solely these manifests",
        payload_validation="Existing full audit inherited; selected markers, recipes, labels and file presence rechecked"))
    atomic_json(output/"export_complete.json",dict(samples=len(selected),hours=actual/3600,training_started=False))
    print(json.dumps(dict(phase="export_complete",output=str(output),samples=len(selected),hours=actual/3600)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("full","stats","catalog","output"):p.add_argument("--"+name,required=True)
    p.add_argument("--workers",type=int,default=32);p.add_argument("--hours",type=float,default=50)
    p.add_argument("--max-per-pair",type=int,default=8);p.add_argument("--core-face-floor",type=float,default=224)
    p.add_argument("--pool-source",help="Reuse a checksum-verified pool for the exact same completed upstream data")
    p.add_argument("--finish-export",action="store_true")
    args=p.parse_args()
    (finish if args.finish_export else build)(args)
