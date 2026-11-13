#!/usr/bin/env python3
"""Build a review from an explicit frozen run, without rewriting descriptions."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/listen_semantic_prompt_pilot_20261027"
RUN = PILOT / "batch_v2"
INPUTS = PILOT / "inputs.jsonl"
TARGETS = ROOT / "experiments/p2_train50h_postprocessed_reselected_20261027/manifests/targets.json"
REVIEW = ROOT / "data_previews/listen_benchmark_review"
OUT = REVIEW / "semantic_prompt_pilot_20261027"
TEMPLATE = Path(__file__).with_name("listen_semantic_prompt_review.html")
FOCUS = "naturalistic_V03_S0220_I00000135_local_6339c7c6a045"
FOCUS_NOTE = (
    "视觉条目从 6 条缩到 4 条，描述字符从 1098 减到 185。手部动作仍带着 while smiling，"
    "和相邻微笑描述重复。末尾选用了较短的 neutral 来源，与较长的 slightly smiling 来源之间的差异没有充分写入冲突。"
    "模型自检没有标出这些问题。这条可以通过程序检查，不能当作全量自动发布已经可靠的证据。"
)


def load_json(path):
    return json.loads(path.read_text())


def explain(error):
    if isinstance(error, list):
        reason = ""
        if error and isinstance(error[0], dict):
            reason = error[0].get("reason") or ""
        return "模型自检拒绝", reason or json.dumps(error, ensure_ascii=False)
    text = error if isinstance(error, str) else json.dumps(error, ensure_ascii=False)
    if text.startswith("[{") or text.startswith("["):
        try:
            return explain(json.loads(text))
        except json.JSONDecodeError:
            pass
    rules = (
        ("Evidence does not overlap", "证据与所选区间不重叠"),
        ("Support must reference a kept source", "证据指向另一条 support，而不是保留条目"),
        ("Plan quote not in source", "原文引用不在来源中"),
        ("Truncated model output", "输出截断"),
        ("Would merge disjoint", "用宽记录把分开的动作连在一起"),
    )
    for needle, label in rules:
        if needle in text:
            return label, text
    return "检查未通过", text


def slim_original(event):
    sources = event.get("description_sources") or []
    descriptions = list(event.get("original_descriptions") or [])
    if not descriptions and event.get("description"):
        descriptions = [event.get("description")]
    for item in sources:
        text = item.get("description")
        if text and text not in descriptions:
            descriptions.append(text)
    return {
        "event_id": event["event_id"],
        "modality": event["modality"],
        "description": event.get("description") or "",
        "text": event.get("text"),
        "start": event.get("start"),
        "end": event.get("end"),
        "intervals": event.get("intervals") or [],
        "categories": event.get("categories") or [],
        "original_descriptions": descriptions,
        "source_count": len(sources),
    }


def slim_derived(event):
    return {
        "event_id": event["event_id"],
        "modality": event["modality"],
        "description": event.get("description") or "",
        "text": event.get("text"),
        "start": event.get("start"),
        "end": event.get("end"),
        "intervals": event.get("intervals") or [],
        "timing_source_id": event.get("timing_source_id"),
        "source_event_ids": event.get("source_event_ids") or [],
        "evidence": event.get("evidence") or [],
    }


def proposal_for(job_id):
    files = list((RUN / "attempts").glob(job_id + ".proposal.*.json"))
    if not files:
        return 0, None
    files.sort(key=lambda path: int(path.name.rsplit(".", 2)[1]))
    last = load_json(files[-1])
    raw = last.get("raw") or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        parsed = {
            "decisions": [],
            "conflicts": [],
            "parse_error": f"第 {last.get('attempt')} 次输出无法解析：{exc}",
            "raw_excerpt": raw[:1200],
        }
    else:
        parsed = {
            "attempt": last.get("attempt"),
            "decisions": parsed.get("decisions") or [],
            "conflicts": parsed.get("conflicts") or [],
            "parse_error": None,
            "raw_excerpt": "",
        }
    return len(files), parsed


def link_media(sample_id, group, targets):
    if group == "historical":
        video = REVIEW / "media" / sample_id / "video.mp4"
        audio = REVIEW / "media" / sample_id / "audio.wav"
        if not video.is_file() or not audio.is_file():
            raise SystemExit(f"missing historical preview: {sample_id}")
        return f"/listen_benchmark_review/media/{sample_id}/video.mp4", f"/listen_benchmark_review/media/{sample_id}/audio.wav"
    row = targets[sample_id]
    dest = OUT / "media" / sample_id
    dest.mkdir(parents=True, exist_ok=True)
    for name, source in (("video.mp4", row["target_video"]), ("audio.wav", row["target_audio"])):
        source_path = Path(source)
        if not source_path.is_file():
            raise SystemExit(f"missing current media: {source}")
        link = dest / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(source_path, dest))
    return f"media/{sample_id}/video.mp4", f"media/{sample_id}/audio.wav"


def visual_stats(events):
    visual = [event for event in events if event.get("modality") == "visual"]
    return len(visual), sum(len(event.get("description") or "") for event in visual)


def main():
    global RUN, INPUTS, OUT, FOCUS
    parser=argparse.ArgumentParser(description="Build a review of a specified completed semantic run; never use implicit old results")
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--focus',default='improvised_V01_S1400_I00000135_local_ee550232f257')
    args=parser.parse_args();RUN=args.run.resolve();INPUTS=args.inputs.resolve();OUT=args.output.resolve();FOCUS=args.focus
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit('Use a new preview directory; existing pages are preserved')
    protocol=load_json(RUN/'protocol.json')
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from listen_benchmark.semantic_prompt import digest
    jobs=[json.loads(line) for line in INPUTS.open() if line.strip()]
    if digest(jobs)!=protocol['input_sha256']:raise SystemExit('Preview input identity mismatch')
    model=Path(protocol['model']).name
    summary = load_json(RUN / "summary.json")
    rejected = {item["job_id"]: item for item in summary["rejected"]}
    targets = {row["sample_id"]: row for row in load_json(TARGETS)}
    results = {}
    for path in (RUN / "results").glob("*.json"):
        record = load_json(path)
        results[record["sample_id"]] = record
    samples = []
    groups = {
        "current": {"samples": 0, "accepted": 0, "visual_in": 0, "visual_out": 0, "chars_in": 0, "chars_out": 0, "audio": 0},
        "historical": {"samples": 0, "accepted": 0, "visual_in": 0, "visual_out": 0, "chars_in": 0, "chars_out": 0, "audio": 0},
    }
    labels = {}
    with (INPUTS).open() as handle:
        for line in handle:
            job = json.loads(line)
            record = job["record"]
            sample_id = record["sample_id"]
            role=job['provenance']['role']
            if role not in ('current_postprocessed','historical_regression_only'):raise SystemExit('Unknown input role')
            group = 'current' if role=='current_postprocessed' else 'historical' 
            originals = [slim_original(event) for event in record["events"]]
            visual_in, chars_in = visual_stats(originals)
            audio_count = sum(event["modality"] == "audio" for event in originals)
            result = results.get(sample_id)
            status = "accepted" if result else "rejected"
            if status == "accepted" and job["job_id"] in rejected:
                raise SystemExit(f"accepted sample also listed as rejected: {sample_id}")
            if status == "rejected" and job["job_id"] not in rejected:
                raise SystemExit(f"missing result without a rejection record: {sample_id}")
            attempts, proposal = proposal_for(job["job_id"])
            derived, decisions, conflicts, review, prompt = [], [], [], None, ""
            visual_out = chars_out = None
            if result:
                if result['source_record']!=record or result['provenance']['protocol_sha256']!=digest(protocol):
                    raise SystemExit('Result/source protocol mismatch')
                original_audio=sorted([e for e in record['events'] if e['modality']=='audio'],key=lambda e:e['event_id'])
                output_audio=sorted([e for e in result['events'] if e['modality']=='audio'],key=lambda e:e['event_id'])
                if original_audio!=output_audio:raise SystemExit(f'audio annotation changed: {sample_id}')
                derived = [slim_derived(event) for event in result["events"]]
                decisions = result.get("source_decisions") or []
                conflicts = result.get("conflicts") or []
                review = result.get("semantic_review")
                prompt_path = RUN / "prompts" / f"{job['job_id']}.txt"
                prompt = prompt_path.read_text() if prompt_path.is_file() else ""
                visual_out, chars_out = visual_stats(derived)
                bucket = groups[group]
                bucket["accepted"] += 1
                bucket["visual_in"] += visual_in
                bucket["visual_out"] += visual_out
                bucket["chars_in"] += chars_in
                bucket["chars_out"] += chars_out
                bucket["audio"] += audio_count
            else:
                label, detail = explain(rejected[job["job_id"]]["error"])
                labels[label] = labels.get(label, 0) + 1
            groups[group]["samples"] += 1
            video, audio = link_media(sample_id, group, targets)
            error_label, error_detail, stage = "", "", ""
            if status == "rejected":
                error_label, error_detail = explain(rejected[job["job_id"]]["error"])
                stage = rejected[job["job_id"]].get("stage") or ""
            samples.append({
                "id": sample_id,
                "job_id": job["job_id"],
                "group": group,
                "group_label": "当前后处理 50h" if group == "current" else "历史回归，只检查规则",
                "status": status,
                "error_label": error_label,
                "error_detail": error_detail,
                "stage": stage,
                "attempts": attempts,
                "duration_s": record["duration_s"],
                "video": video,
                "audio": audio,
                "focus": sample_id == FOCUS,
                "review_note": FOCUS_NOTE if sample_id == FOCUS and RUN.name=="batch_v2" else "",
                "visual_in": visual_in,
                "visual_out": visual_out,
                "chars_in": chars_in,
                "chars_out": chars_out,
                "audio_count": audio_count,
                "events": derived,
                "proposal": proposal if status == "rejected" else None,
                "originals": originals,
                "decisions": decisions,
                "conflicts": conflicts,
                "semantic_review": review,
                "prompt": prompt,
            })
    if len(samples)!=summary['samples'] or len(results)+len(rejected)!=len(samples):
        raise SystemExit('Incomplete or mismatched result accounting')
    samples.sort(key=lambda item: (not item["focus"], item["group"] != "current", item["status"] != "accepted", item["id"]))
    payload = {
        "focus_id": FOCUS,
        "model": model,
        "run": str(RUN),
        "summary": {
            "samples": len(samples),
            "accepted": summary["accepted"],
            "rejected": len(rejected),
            "current": groups["current"],
            "historical": groups["historical"],
            "reject_labels": [{"label": label, "count": count} for label, count in sorted(labels.items(), key=lambda item: (-item[1], item[0]))],
            "raw_annotations_modified": False,
            "canonical_cache_updated": False,
        },
        "samples": samples,
    }
    if sum(g["accepted"] for g in groups.values())!=summary["accepted"]:
        raise SystemExit("acceptance counts disagree")
    OUT.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    (OUT / "index.html").write_text(html)
    (OUT / "summary.json").write_text(json.dumps(payload["summary"], ensure_ascii=False, indent=2) + "\n")
    (OUT / "README.md").write_text(
        f"# 语义去冗余复核页\n\n来源：{RUN}\n\n模型：{model}；共{len(samples)}条，"
        f"通过{summary['accepted']}条，拒绝{len(rejected)}条。\n\n"
        "描述直接展示结果文件；音频记录逐字段核对。不改原标签、媒体和训练缓存。通过程序及模型自检不等于人工真值。\n"
    )
    print(json.dumps({"page": str(OUT / "index.html"), "bytes": (OUT / "index.html").stat().st_size, "summary": payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
