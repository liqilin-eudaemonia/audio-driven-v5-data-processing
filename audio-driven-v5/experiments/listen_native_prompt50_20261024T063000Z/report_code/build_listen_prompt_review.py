#!/usr/bin/env python3
"""Export fresh native-AV prompt A/B results alongside existing Qwen3-VL reference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--review", default=str(ROOT / "data_previews/listen_benchmark_review"))
    a = p.parse_args()
    run, review = Path(a.run).resolve(), Path(a.review).resolve()
    out = review / "native_prompt"
    out.mkdir(exist_ok=True)
    old = json.loads((review / "review.json").read_text())
    feedback = json.loads((run / "human_feedback.json").read_text())
    notes = {r["sample_id"]: r["note"] for r in feedback["records"]}
    manifest = json.loads((run / "manifest.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    samples = []
    for row in manifest:
        sid = row["sample_id"]
        previous = next(s for s in old["samples"] if s["id"] == sid)
        assert previous["source_hashes"] == row["media_hashes"]
        sample = dict(id=sid, duration=row["duration_s"], video="../"+previous["video"],
                      audio="../"+previous["audio"], note=notes.get(sid, ""), results={})
        waveform, rate = sf.read(row["target_audio"])
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        step = max(1, round(rate * .02))
        sample["energy"] = [[round(i/rate, 5), float(np.sqrt(np.mean(waveform[i:i+step]**2)))]
                            for i in range(0, len(waveform), step)]
        sample["energy_window_s"] = step / rate
        for system in cfg["systems"]:
            name = system["name"]
            path = run / "results/items" / sid / name / "summary.json"
            result = dict(status="pending", raw="", events=[])
            if path.exists():
                record = json.loads(path.read_text())["results"][0]
                raw = record.get("raw_response", {})
                request = json.loads(Path(record["path"]).with_name("request.json").read_text())
                result = dict(status=record["status"], parsed=record.get("parsed"),
                              raw=raw.get("text", ""), error=record.get("error"), prompt=request["prompt"],
                              finish=raw.get("finish_reason"), tokens=raw.get("completion_tokens"),
                              seconds=record.get("inference_wall_seconds"),
                              events=record.get("parsed", {}).get("events", []) if record["status"]=="schema_pass" else [])
                dest = out / "raw" / sid / (name+".json")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record["path"], dest)
                result["raw_url"] = str(dest.relative_to(out))
            sample["results"][name] = result
        reference = old["results"]["qwen3_vl_visual"][sid]["discovery"]
        sample["results"]["qwen3_vl_reference"] = dict(
            reference, events=(reference.get("parsed") or {}).get("events", []) if reference["status"]=="schema_pass" else [],
            raw_url="../"+reference["raw_url"], historical_reference=True)
        samples.append(sample)
    names = {"native_baseline":"Omni 原生视听 · 原 prompt", "native_occurrences_v1":"Omni 原生视听 · 逐次事件 prompt",
             "qwen3_vl_reference":"Qwen3-VL 视觉 · 历史参考"}
    stats = {}
    for name in names:
        counts = Counter()
        for sample in samples:
            result = sample["results"][name]
            counts[result["status"]] += 1
            if result["status"] != "schema_pass":
                continue
            events = result["events"]
            counts["positive_clips"] += bool(events)
            for event in events:
                counts[event["modality"]] += 1
                counts["long_events"] += event["end"]-event["start"] >= .8 * sample["duration"]
                counts["uncertain_events"] += bool(event.get("uncertainty"))
        stats[name] = dict(counts)
    resources = [json.loads(line) for line in (run / "resources.jsonl").read_text().splitlines()] if (run / "resources.jsonl").exists() else []
    data = dict(run_id=run.name, built_utc=datetime.now(timezone.utc).isoformat(), samples=samples,
                names=names, stats=stats, resources=resources, human_ground_truth=False,
                status=json.loads((run / "status.json").read_text()),
                analysis=(run / "analysis.txt").read_text() if (run / "analysis.txt").exists() else "新旧提示词对照正在生成。事件更多、区间更短均不代表更准确；需结合媒体复核。")
    (out / "comparison.json").write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")
    template = (ROOT / "scripts/listen_prompt_review.html").read_text()
    html = template.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font_path = Path('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font_path.exists():
        font = TTFont(font_path, fontNumber=2)
        selected = subset.Subsetter()
        selected.populate(text=html)
        selected.subset(font)
        font.flavor = 'woff'
        font.save(out / 'review.woff')
    temp = out / "index.tmp"
    temp.write_text(html)
    temp.replace(out / "index.html")
    (run / "comparison_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(output=str(out), status=data["status"], stats=stats), ensure_ascii=False))


if __name__ == "__main__":
    main()
