#!/usr/bin/env python3
"""Build the semantic-dedup review page from the frozen 8B pilot. No media re-inference."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/listen_semantic_prompt_pilot_20261027"
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
    files = list((PILOT / "batch_v2/attempts").glob(job_id + ".proposal.*.json"))
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
        return f"../media/{sample_id}/video.mp4", f"../media/{sample_id}/audio.wav"
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
    selection = load_json(PILOT / "selection.json")
    current_ids = set(selection["current_postprocessed_ids"])
    historical_ids = set(selection["historical_ids"])
    summary = load_json(PILOT / "batch_v2/summary.json")
    rejected = {item["job_id"]: item for item in summary["rejected"]}
    targets = {row["sample_id"]: row for row in load_json(TARGETS)}
    results = {}
    for path in (PILOT / "batch_v2/results").glob("*.json"):
        record = load_json(path)
        results[record["sample_id"]] = record
    samples = []
    groups = {
        "current": {"samples": 0, "accepted": 0, "visual_in": 0, "visual_out": 0, "chars_in": 0, "chars_out": 0, "audio": 0},
        "historical": {"samples": 0, "accepted": 0, "visual_in": 0, "visual_out": 0, "chars_in": 0, "chars_out": 0, "audio": 0},
    }
    labels = {}
    with (PILOT / "inputs.jsonl").open() as handle:
        for line in handle:
            job = json.loads(line)
            record = job["record"]
            sample_id = record["sample_id"]
            group = "current" if sample_id in current_ids else "historical"
            if group == "historical" and sample_id not in historical_ids:
                raise SystemExit(f"sample outside the two pilot groups: {sample_id}")
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
                for left, right in zip(
                    [event for event in record["events"] if event["modality"] == "audio"],
                    [event for event in result["events"] if event["modality"] == "audio"],
                ):
                    same = all(left.get(key) == right.get(key) for key in ("event_id", "text", "description", "start", "end", "intervals"))
                    if not same:
                        raise SystemExit(f"audio annotation changed: {sample_id}")
                derived = [slim_derived(event) for event in result["events"]]
                decisions = result.get("source_decisions") or []
                conflicts = result.get("conflicts") or []
                review = result.get("semantic_review")
                prompt_path = PILOT / "batch_v2/prompts" / f"{job['job_id']}.txt"
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
                "review_note": FOCUS_NOTE if sample_id == FOCUS else "",
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
    if sum(groups[name]["samples"] for name in groups) != 100:
        raise SystemExit("expected 100 pilot samples")
    expected = {
        "current": {"accepted": 23, "visual_in": 120, "visual_out": 90, "chars_in": 12238, "chars_out": 2410, "audio": 32},
        "historical": {"accepted": 43, "visual_in": 66, "visual_out": 53, "chars_in": 5864, "chars_out": 1613, "audio": 31},
    }
    for name, want in expected.items():
        got = {key: groups[name][key] for key in want}
        if got != want:
            raise SystemExit(f"{name} stats {got} != {want}")
    samples.sort(key=lambda item: (not item["focus"], item["group"] != "current", item["status"] != "accepted", item["id"]))
    payload = {
        "focus_id": FOCUS,
        "summary": {
            "samples": 100,
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
    if payload["summary"]["accepted"] != 66 or payload["summary"]["rejected"] != 34:
        raise SystemExit("acceptance counts drifted")
    OUT.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    (OUT / "index.html").write_text(html)
    (OUT / "summary.json").write_text(json.dumps(payload["summary"], ensure_ascii=False, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# 语义去冗余复核页\n\n"
        "页面：http://127.0.0.1:8876/listen_benchmark_review/semantic_prompt_pilot_20261027/\n\n"
        "展示 2026-10-27 本地 Qwen3-VL-8B-Instruct 纯文本试验的 100 条结果。"
        "66 条为通过程序与模型自检的候选指导，34 条保留拒绝原因和最后一次提案，没有候选指导。"
        "原始标注全文留在右侧。音频文字、次数和有效区间原样保留。没有改训练标签，也没有改缓存。\n\n"
        "当前 50 小时样本的画面和 listener 音频链接到已有 512 裁剪；历史回归样本复用原文整合页的预览媒体。没有重新识别视频。\n\n"
        "问题样本 `naturalistic_V03_S0220_I00000135_local_6339c7c6a045` 排在列表首位。"
        "程序通过只说明引用、时间和来源约束成立，页面上的人工阅读备注说明它仍有附带重复和冲突披露不足。\n"
    )
    print(json.dumps({"page": str(OUT / "index.html"), "bytes": (OUT / "index.html").stat().st_size, "summary": payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
