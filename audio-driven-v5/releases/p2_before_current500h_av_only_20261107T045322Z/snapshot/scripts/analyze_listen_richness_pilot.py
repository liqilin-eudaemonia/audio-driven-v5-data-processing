#!/usr/bin/env python3
"""CPU-only, auditable selection statistics from an existing caption extraction pilot.

This is a selection sidecar, not an annotation editor or a new ground-truth dataset.
Rules are deliberately conservative; unrecognized behavior names remain open candidates.
No per-sample overrides, model calls, media decoding, or source DB mutations.
"""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from listen_richness.common import digest, file_hash, percentile, rows, write, write_jsonl
from listen_richness.extraction import sentences

VERSION = "richness-type-quantity-tables-v2"
# Statistics keys only. Original names and descriptions are never replaced.
ALIASES = {
    "visual": {"nod": "head nodding", "nodding": "head nodding", "head nod": "head nodding",
               "smile": "smiling", "blink": "blinking", "eye closure": "eye closing",
               "lip movement": "mouth movement", "lip parting": "mouth opening"},
    "audio": {"laughing": "audible laughter", "laughter": "audible laughter",
              "spoken response": "speech"},
}
BASELINE_NAMES = {
    "posture", "seated", "facing forward", "gaze holding", "gaze", "head stillness",
    "head posture", "mouth position", "brow position", "eye openness", "lips closed",
    "neutral expression", "facial expression neutral",
}
VAGUE_NAMES = {"expression", "facial expression", "lips", "eyebrows"}
NUMBERS = {"one": 1, "single": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3,
           "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
NUM = r"(?:\d+|" + "|".join(NUMBERS) + r")"
TARGETS = {
    "head nodding": r"\bnod(?:s|ding)?\b", "head tilting": r"\btilt(?:s|ing|ed)?\b",
    "eye closing": r"\bclos(?:e|es|ing|ed)\b.{0,22}\beyes\b|\beyes\b.{0,22}\bclos",
    "blinking": r"\bblink(?:s|ing)?\b", "speech": r"\butterance\b|\bspoken word\b|\bsay(?:s)?\b|\bspeak",
    "humming": r"\bhum(?:s|ming)?\b", "vocalization": r"\bvocalization\b|\butterance\b|\bsound\b",
    "audible laughter": r"\blaugh|\bchuckle", "gaze shifting": r"\bgaze\b|\blooks?\b",
}


def has(pattern, text):
    return bool(re.search(pattern, text, flags=re.I))


def number_value(token):
    return int(token) if token.isdigit() else NUMBERS[token.lower()]


def brief_close_reopen(text):
    return has(r"(?:briefly clos\w*.{0,25}eyes|eyes close briefly).{0,140}(?:reopen|opens?.{0,20}eyes)", text)


def classify(b, evidence, caption, snapshot):
    """Return a statistics disposition; never overwrite the model's record."""
    name, modality = b["canonical"], b["raw"]["modality"]
    form = b["raw"]["form"]
    text = " ".join(evidence)
    if b["raw"]["polarity"] != "positive":
        return "not_positive", "model_negative_or_uncertain"
    if modality == "audio" and snapshot["audio"]["waveform_full_silence"]:
        return "pending", "audio_claim_on_fully_muted_waveform"
    if form == "baseline":
        return "baseline", "model_baseline"
    if name in BASELINE_NAMES and form != "change":
        return "baseline", "ordinary_resting_context"
    if name in VAGUE_NAMES:
        if has(r"neutral|relaxed|unchanged|lips closed", text):
            return "baseline", "neutral_resting_context"
        return "pending", "unspecified_behavior_name"
    if name == "gaze shifting":
        # Require a gaze/eye/looking transition, not a nearby mouth/head movement.
        motion = (r"(?:shift\w*|lift\w*|lower\w*|rais\w*|return\w*|turn\w*)\b.{0,24}\b(?:gaze|eyes|look\w*|focus)"
                  r"|\b(?:gaze|eyes)\b.{0,22}\b(?:shift\w*|lift\w*|lower\w*|return\w*)"
                  r"|\blook\w*\b.{0,30}\b(?:before|then)\b.{0,25}(?:lift|look|return)"
                  r"|\beyes look\b.{0,65}\bbefore shifting\b"
                  r"|shifts? in (?:his |her |their )?gaze")
        if not has(motion, text):
            if has(r"steady|consistent|fixed|remain|holding|maintain|initially|begins by looking", text):
                return "baseline", "gaze_direction_without_explicit_transition"
            return "pending", "gaze_transition_not_supported"
    if name == "head tilting":
        if has(r"(?:performs?|gives?|distinct|brief|single)\b.{0,25}\bnod", text) and not any(not has(r"\bnod", s) and has(r"\btilt|returning to a neutral", s) for s in evidence):
            return "pending", "possible_nod_mechanics_not_additional_tilt"
        if form == "state" and not has(r"\btilts\b|\btilting\b|returning|before settling", text):
            return "baseline", "held_head_position_not_movement"
        if form == "change" and not has(r"\btilt|lower.{0,18}head|head.{0,18}downward|returning to a neutral|before settling", text):
            return "pending", "generic_head_shift_does_not_prove_tilt"
    if name in {"mouth closing", "mouth opening"}:
        noun = r"(?:mouth|lips)"
        verb = r"(?:opens|opening|parts|parting)" if name == "mouth opening" else r"(?:closes|closing)"
        if not has(noun + r"\b.{0,25}\b" + verb + r"\b|\b" + verb + r"\b.{0,25}\b" + noun, text):
            if name == "mouth closing" and has(r"closed", text):
                return "baseline", "closed_mouth_position_not_transition"
            if form == "change":
                return "pending", "mouth_position_not_opening_transition"
    if name == "adjusting glasses" and not has(r"adjust\w*\b.{0,30}\bglasses", text):
        return "pending", "glasses_adjustment_not_in_source"
    if name == "adjusting fingers" and not has(r"adjust\w*\b.{0,30}\bfinger", text):
        return "pending", "finger_adjustment_not_in_source"
    if name == "blinking" and not has(r"blink", text):
        if not brief_close_reopen(caption):
            return "pending", "eye_closure_without_reopening_not_blink"
    if name == "speech" and modality == "visual":
        return "pending", "visible_speech_motion_is_not_audio_or_separate_visual_type"
    if modality == "audio" and has(r"resembling|as if|consistent with|suggesting", text):
        if not has(r"audibl\w*|\bsays\b|producing|produces|emits|emitting|audio contains", text):
            return "pending", "sound_inferred_from_appearance"
    return "counted", "caption_positive_candidate"


def repetition(name, evidence, raw_repeats):
    """Separate literal, inferred and unknown counts. Never accept clock numbers."""
    text = " ".join(evidence).lower()
    target = TARGETS.get(name)
    literal = []
    if target:
        for m in re.finditer(r"\b(" + NUM + r")\b", text):
            token = m.group(1)
            before, after = text[max(0, m.start()-45):m.start()], text[m.end():m.end()+55]
            # Clock expressions and times inside quoted content are not repetitions.
            if has(r"^\s*(?:[-–]|\.\d|and a half|seconds?\b|minutes?\b)", after):
                continue
            lead = re.match(r"^\s*,?\s*(?:(?:distinct|brief|subtle|clear|short|downward|head)\s*,?\s*)*", after)
            tail = after[lead.end():] if lead else after
            tied_before = has(target, before) and token in {"once", "twice", "thrice"}
            tied_after = bool(re.match(target, tail, flags=re.I))
            if tied_before or tied_after:
                literal.append(number_value(token))
    # Numbered composite sequence is evidence only for named constituents, not all nearby actions.
    if name in {"eye closing", "head tilting", "blinking"} and has(r"sequence of closing.{0,25}eyes and lowering.{0,20}head occurs once", text):
        literal.append(1)
    clauses=re.split(r"[,;.!?]|\b(?:and|while|before|after|though)\b",text)
    # Never borrow repetition words from a different action in the same sentence.
    repeat_marker=r"\brepeats?\b|\brepeatedly\b|\boccasionally\b|\bseveral (?:times|nods|blinks)\b|series of|intermittently"
    repeated=bool(target and any(has(target,c) and has(repeat_marker,c) for c in clauses))
    if name=="head nodding" and has(r"\bnods?\b.{0,15}\bagain\b",text):
        repeated=True
    # A single occurrence followed by one explicitly described recurrence permits a derived count,
    # not a literal total. Require the action to be present in this grouped evidence.
    derived_two = bool(target and has(target, text) and (
        (has(r"\bsingle\b", text) and has(r"(?:this nod repeats|she repeats this nod|he repeats this nod|this nod is followed by another)", text) and name == "head nodding")
        or (name == "humming" and has(r"\bhum\b", text) and has(r"another similar sound", text))
        or (name == "speech" and has(r"repeats the word", text))))
    unique = sorted(set(literal))
    if len(unique) == 1 and unique[0] > 1:
        return dict(kind="literal", count=unique[0], evidence=evidence, raw=raw_repeats)
    if derived_two:
        return dict(kind="derived_from_distinct_mentions", count=2, evidence=evidence, raw=raw_repeats)
    if unique == [1] and not repeated:
        # A short utterance followed by further speech is not the total speech count.
        if name == "speech" and has(r"continues speaking|continues to speak|further", text):
            return dict(kind="partial_literal_not_total", count=None, evidence=evidence, raw=raw_repeats)
        return dict(kind="literal", count=1, evidence=evidence, raw=raw_repeats)
    if repeated:
        return dict(kind="multiple_unspecified", count=None, evidence=evidence, raw=raw_repeats)
    return dict(kind="unknown", count=None, evidence=evidence, raw=raw_repeats)


def aggregate(snapshot, payload):
    spans = {s["id"]: s["text"] for s in sentences(snapshot["caption"])}
    ledger, groups = [], {}
    for raw in payload["behaviors"]:
        b = dict(raw=copy.deepcopy(raw), canonical=ALIASES.get(raw["modality"], {}).get(raw["behavior"], raw["behavior"]))
        ids = raw.get("source_ids", [])
        count_ids = raw.get("repeat", {}).get("source_ids", [])
        all_ids = sorted(set(ids + count_ids))
        b["source_ids"] = all_ids
        b["evidence"] = [spans[i] for i in all_ids if i in spans]
        b["reference_repaired"] = not set(count_ids).issubset(ids)
        if b["canonical"] == "eye closing" and brief_close_reopen(" ".join(b["evidence"])):
            b["canonical"] = "blinking"
            b["contextual_alias_reason"] = "explicit_brief_close_reopen_cycle"
        if not ids or any(i not in spans for i in all_ids):
            b.update(disposition="pending", reason="invalid_source_ids")
        else:
            b["disposition"], b["reason"] = classify(b, b["evidence"], snapshot["caption"], snapshot)
        ledger.append(b)
        if b["disposition"] != "counted":
            continue
        key = raw["modality"] + ":" + b["canonical"]
        g = groups.setdefault(key, dict(key=key, behavior=b["canonical"], modality=raw["modality"],
                                        evidence_ids=[], source_ids=set(), forms=set(), raw_repeats=[], sources=["caption"]))
        g["evidence_ids"].append(raw["id"])
        g["source_ids"].update(all_ids)
        g["forms"].add(raw["form"])
        g["raw_repeats"].append(raw["repeat"])
    for g in groups.values():
        g["source_ids"] = sorted(g["source_ids"])
        g["forms"] = sorted(g["forms"])
        g["evidence"] = [spans[i] for i in g["source_ids"]]
        g["repeat"] = repetition(g["behavior"], g["evidence"], g.pop("raw_repeats"))
        if g["key"] == "visual:head nodding" and snapshot["nod"]["candidate_interval_count"]:
            g["sources"].append("nod_detector")
        if g["modality"] == "audio" and snapshot["audio"]["official_bout_count"]:
            g["sources"].append("official_audio_union")
        # State groups are not measured physical event counts.
        g["state_group_count"] = int(g["modality"]=="visual" and "state" in g["forms"])
        pure_state = g["modality"]=="visual" and g["forms"] == ["state"]
        g["quantity"] = dict(unit="state_group" if pure_state else "occurrence",
            count=1 if pure_state else g["repeat"]["count"],
            basis="deduplicated_state_group" if pure_state else g["repeat"]["kind"],
            physical_event_count_verified=False)
    src, face, audio, nod = (snapshot[k] for k in ("source", "face", "audio", "nod"))
    grouped = sorted(groups.values(), key=lambda g: g["key"])
    pending = [b for b in ledger if b["disposition"] == "pending"]
    names = {g["key"] for g in grouped}
    # Broad evidence coverage: one nod type and one audio presence flag, never event-count addition.
    nod_candidate = bool(nod["candidate_interval_count"])
    audio_candidate = bool(audio["official_bout_count"])
    nod_negative = any(b["canonical"] == "head nodding" and b["raw"]["polarity"] == "negative" for b in ledger)
    flags = []
    if nod_candidate and nod_negative: flags.append("nod_caption_detector_disagreement")
    if audio_candidate and not any(g["modality"] == "audio" for g in grouped): flags.append("official_audio_without_text_audio_type")
    if any(b["reason"] == "audio_claim_on_fully_muted_waveform" for b in ledger): flags.append("text_audio_on_muted_waveform")
    p10 = face["face_short_side_p10"]
    row = dict(sample_id=snapshot["sample_id"], pair=src["pair"], split=src["split"], duration_s=src["duration_s"],
               listener_id=src["listener_id"],source_start_s=src["source_start_s"],source_end_s=src["source_end_s"],
               caption=snapshot["caption"], caption_file=snapshot["caption_file"], crop_revision=snapshot["crop_revision"],
               **face, retained=True, training_admitted=False, human_ground_truth=False,
               raw_positive_type_count=len({(b["modality"], b["behavior"]) for b in payload["behaviors"] if b["polarity"] == "positive"}),
               counted_type_count=len(names), counted_types=sorted(names),
               change_type_count=sum("change" in g["forms"] for g in grouped),
               state_type_count=sum(g["state_group_count"] for g in grouped),
               behavior_quantities=[dict(behavior=g["key"],**g["quantity"]) for g in grouped],
               known_occurrences_by_type={g["key"]:g["quantity"]["count"] for g in grouped
                                         if g["quantity"]["unit"]=="occurrence" and g["quantity"]["count"] is not None},
               unknown_occurrence_types=[g["key"] for g in grouped if g["quantity"]["unit"]=="occurrence" and g["quantity"]["count"] is None],
               state_group_count=sum(g["state_group_count"] for g in grouped),
               pending_behavior_count=len(pending), baseline_record_count=sum(b["disposition"] == "baseline" for b in ledger),
               negative_or_uncertain_record_count=sum(b["disposition"] == "not_positive" for b in ledger),
               text_stat_status="partial_with_pending" if pending else "computed_not_ground_truth",
               groups=grouped, evidence_ledger=ledger, audio=audio, nod=nod, consistency_flags=flags,
               nod_counted_once=int(nod_candidate or "visual:head nodding" in names),
               audio_presence_counted_once=int(audio_candidate or any(g["modality"] == "audio" for g in grouped)),
               face_core_candidate=p10 is not None and p10 >= 224,
               face_supplement_candidate=p10 is not None and p10 >= 160,
               zero_counted_types_is_verified_absence=False)
    row["duration_bin_s"] = int(src["duration_s"] // 2) * 2
    return row


def report(records):
    catalog=defaultdict(lambda: dict(sample_ids=[], pairs=set(), state_samples=0, change_samples=0,
        literal_counts=[], derived_counts=[], unknown_occurrence_samples=0, pending_sample_ids=[],
        covered_clip_seconds=0., known_occurrence_samples=0))
    reasons=Counter()
    for r in records:
        reasons.update(b["reason"] for b in r["evidence_ledger"] if b["disposition"] in {"pending","baseline"})
        for key in {b["raw"]["modality"]+":"+b["canonical"] for b in r["evidence_ledger"] if b["disposition"]=="pending"}:
            catalog[key]["pending_sample_ids"].append(r["sample_id"])
        for g in r["groups"]:
            c=catalog[g["key"]];c["sample_ids"].append(r["sample_id"]);c["pairs"].add(r["pair"])
            c["state_samples"]+=g["state_group_count"];c["change_samples"]+=int("change" in g["forms"])
            c["covered_clip_seconds"]+=r["duration_s"]
            if g["quantity"]["unit"]=="occurrence":
                if g["quantity"]["count"] is None:c["unknown_occurrence_samples"]+=1
                else:c["known_occurrence_samples"]+=1
                if g["repeat"]["kind"]=="literal":c["literal_counts"].append(g["repeat"]["count"])
                if g["repeat"]["kind"]=="derived_from_distinct_mentions":c["derived_counts"].append(g["repeat"]["count"])
    for c in catalog.values():
        c["sample_count"]=len(c["sample_ids"]);c["pair_count"]=len(c.pop("pairs"))
        c["literal_occurrences_sum"]=sum(c["literal_counts"]) if c["literal_counts"] else None
        c["derived_occurrences_sum"]=sum(c["derived_counts"]) if c["derived_counts"] else None
        c["supported_occurrences_sum"]=sum(c["literal_counts"]+c["derived_counts"]) if c["known_occurrence_samples"] else None
        c["pending_sample_count"]=len(c["pending_sample_ids"])
    overlaps=[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if (a["pair"],a["listener_id"])==(b["pair"],b["listener_id"]) and max(a["source_start_s"],b["source_start_s"])<min(a["source_end_s"],b["source_end_s"]):
                overlaps.append([a["sample_id"],b["sample_id"]])
    return dict(schema=VERSION, sample_count=len(records), retained_count=sum(r["retained"] for r in records),
                duration_s=sum(r["duration_s"] for r in records),
                counted_sample_type_total=sum(r["counted_type_count"] for r in records),
                raw_positive_sample_type_total=sum(r["raw_positive_type_count"] for r in records),
                distinct_candidate_types=sum(c["sample_count"]>0 for c in catalog.values()),
                pending_only_types=[k for k,c in catalog.items() if not c["sample_count"]],
                pending_samples=sum(bool(r["pending_behavior_count"]) for r in records),
                pending_records=sum(r["pending_behavior_count"] for r in records),
                zero_counted_type_samples=sum(r["counted_type_count"]==0 for r in records),
                audio_union_samples=sum(r["audio"]["official_bout_count"]>0 for r in records),
                official_audio_union_seconds=sum(r["audio"]["official_core_s"] for r in records),
                nod_candidate_samples=sum(r["nod"]["candidate_interval_count"]>0 for r in records),
                nod_candidate_intervals=sum(r["nod"]["candidate_interval_count"] for r in records),
                nod_support_seconds=sum(r["nod"]["support_s"] for r in records),
                face_p10_ge224=sum(r["face_core_candidate"] for r in records),
                face_p10_ge160=sum(r["face_supplement_candidate"] for r in records),
                face_p50_median=percentile([r["face_short_side_p50"] for r in records if r["face_short_side_p50"] is not None],.5),
                reasons=dict(reasons), catalog=dict(sorted(catalog.items())),
                source_overlap_pairs=overlaps,count_scope="sum_across_clips_not_unique_source_events",
                no_new_visual_timestamps=True, exact_event_total=None, video_ground_truth_audited=False,
                selection_applied=False, note="Types/count proxies from existing text, not verified events. No record removed.")


ZH = {"head nodding":"点头","smiling":"微笑","blinking":"眨眼","eye closing":"闭眼",
      "speech":"口头回应","vocalization":"发声（未细分）","humming":"轻哼","audible laughter":"可听笑声",
      "gaze shifting":"视线转移","head tilting":"头部倾斜","head turning":"转头","head movement":"头部运动",
      "hand raising":"抬手","hand lowering":"放下手","hand covering":"手部遮挡","adjusting glasses":"调整眼镜",
      "adjusting fingers":"调整手指","eyebrow raising":"抬眉","eyebrow movement":"眉部运动","brow furrowing":"眉部皱起",
      "forehead wrinkling":"额头起皱","frowning":"皱眉表情","eye movement":"眼部运动","mouth movement":"嘴唇／口部运动",
      "mouth opening":"张嘴／嘴唇张开","mouth closing":"闭嘴","lip pursing":"抿嘴／噘嘴"}
BASIS_ZH={"literal":"原文明示","derived_from_distinct_mentions":"明确分次推得","unknown":"次数未知",
          "multiple_unspecified":"多次，次数未知","partial_literal_not_total":"局部有次数，总数未知","deduplicated_state_group":"持续状态组"}


def display_name(key):
    modality,name=key.split(":",1)
    if key=="visual:speech":return "视觉中的speech（模态／含义待核）"
    return ZH.get(name,name)+("（音频）" if modality=="audio" else "")


def quantity_text(g):
    q=g["quantity"]
    if q["unit"]=="state_group":return "1个状态组（不是动作次数）"
    if q["count"] is None:return BASIS_ZH[q["basis"]]
    return f"{q['count']}次（{BASIS_ZH[q['basis']]}）"


def export_tables(out,records,summary):
    columns=["样本ID","时长秒","采用类型数","待核条目数","各类数量","次数未知类型","原始人脸p10","原始人脸p50","官方发声支持秒","点头候选区间数","原提取错误"]
    with (out/"sample_stats.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(columns)
        for r in records:w.writerow([r["sample_id"],r["duration_s"],r["counted_type_count"],r["pending_behavior_count"],
            "; ".join(display_name(g["key"])+": "+quantity_text(g) for g in r["groups"]),
            "; ".join(display_name(k) for k in r["unknown_occurrence_types"]),r["face_short_side_p10"],r["face_short_side_p50"],
            r["audio"]["official_core_s"],r["nod"]["candidate_interval_count"],r["original_validation_error"]])
    with (out/"behavior_catalog.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["类型键","显示名","含该类片段数","原文明示次数合计","分次推得次数合计","有次数依据片段数","次数未知片段数","持续状态组数","待核片段数","覆盖片段时长秒"])
        for k,c in summary["catalog"].items():w.writerow([k,display_name(k),c["sample_count"],c["literal_occurrences_sum"],c["derived_occurrences_sum"],c["known_occurrence_samples"],c["unknown_occurrence_samples"],c["state_samples"],c["pending_sample_count"],round(c["covered_clip_seconds"],3)])
    event_rows=[]
    for r in records:
        for g in r["groups"]:
            event_rows.append(dict(sample_id=r["sample_id"],behavior=g["key"],display_name=display_name(g["key"]),
                quantity=g["quantity"],evidence=g["evidence"],source_ids=g["source_ids"],sources=g["sources"]))
    write_jsonl(out/"behavior_quantities.jsonl",event_rows)
    lines=["# 50段：行为类型与数量统计","","全部50段保留。类型为开放行为候选，不等于已确认的交际性backchannel分类。仅使用现有v3提取和CPU统计；没有重新识别视频。","",
        "次数空白／未知不等于零。状态组不与动作次数相加；点头检测区间、官方发声区间只作独立支持。覆盖时长是含该类片段的整段时长，不是该动作持续时长。","",
        "## 按类别汇总","","| 类型 | 含该类片段 | 明示次数合计 | 分次推得次数合计 | 次数未知片段 | 状态组 | 待核片段 |","|---|---:|---:|---:|---:|---:|---:|"]
    for k,c in summary["catalog"].items():
        lines.append(f"| {display_name(k)} | {c['sample_count']} | {c['literal_occurrences_sum'] if c['literal_occurrences_sum'] is not None else '—'} | {c['derived_occurrences_sum'] if c['derived_occurrences_sum'] is not None else '—'} | {c['unknown_occurrence_samples']} | {c['state_samples']} | {c['pending_sample_count']} |")
    lines += ["","## 逐片段汇总","","| 样本 | 类型数 | 各类数量／未知 | 人脸p10 | 待核条目 |","|---|---:|---|---:|---:|"]
    for i,r in enumerate(records,1):
        values="；".join(display_name(g["key"])+"："+quantity_text(g) for g in r["groups"]) or "无采用项；不代表没有行为"
        lines.append(f"| [{r['sample_id']}](review50.md#sample-{i:02}) | {r['counted_type_count']} | {values} | {r['face_short_side_p10']:.1f} | {r['pending_behavior_count']} |")
    (out/"tables.md").write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();p=args.pilot.resolve();out=args.out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError("Use a new output directory")
    requests={r["sample_id"]:r for r in rows(p/"requests50_v3.jsonl")}
    review={r["sample_id"]:r for r in rows(p/"review50.jsonl")}
    db=p/"stats50_v4.sqlite";dbhash=file_hash(db)
    with sqlite3.connect(f"file:{db}?mode=ro",uri=True) as c:
        snapshots={sid:(sha,json.loads(s)) for sid,sha,s in c.execute("select sid,snapshot_sha,snapshot from samples")}
    if set(requests)!=set(snapshots) or set(review)!=set(requests):raise ValueError("Sample identity mismatch")
    records=[];input_hashes={}
    for sid in sorted(requests):
        sha,snapshot=snapshots[sid];r=requests[sid]
        if digest(snapshot)!=sha or sha!=r["snapshot_sha256"]:raise ValueError("Snapshot hash mismatch")
        if file_hash(snapshot["caption_file"])!=snapshot["caption_file_sha256"]:raise ValueError("Source caption changed")
        if json.loads(r["messages"][-1]["content"])["caption"]!=snapshot["caption"]:raise ValueError("Caption mismatch")
        v=review[sid]["versions"]["v3"]
        attempt_path=p/"pilot50_v3/attempts"/f"{sid}.{v['attempt']}.json"
        attempt=json.loads(attempt_path.read_text());payload=json.loads(attempt["raw"])
        if payload!=v["payload"] or attempt["snapshot_sha256"]!=sha:raise ValueError("Attempt mismatch")
        row=aggregate(snapshot,payload)
        row.update(original_validation_error=v["validation_error"], extraction_attempt=v["attempt"],
                   extraction_file=str(attempt_path),snapshot_sha256=sha)
        records.append(row);input_hashes[str(attempt_path)]=file_hash(attempt_path)
        input_hashes[snapshot["caption_file"]]=snapshot["caption_file_sha256"]
    summary=report(records)
    out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/"sample_stats.jsonl",records);write(out/"summary.json",summary)
    export_tables(out,records,summary)
    lines=["# 50段统计逐条复核", "", "只使用v3已有响应；全部数据保留。以下是文本统计候选，不是视频真值。", "",
           "类型数不等于事件次数；pending保留原文但暂不计入已采用统计；普通状态另存；不输出评分。", ""]
    for i,r in enumerate(records,1):
        lines += [f'<a id="sample-{i:02}"></a>',"",f"## {r['sample_id']}","",r["caption"],"",
                  f"时长 {r['duration_s']:.3f}s；原始人脸p10/p50 {r['face_short_side_p10']:.1f}/{r['face_short_side_p50']:.1f}px；原始正向类型 {r['raw_positive_type_count']} → 采用类型 {r['counted_type_count']}；待核 {r['pending_behavior_count']}。",
                  f"点头候选 {r['nod']['candidate_interval_count']} 段／{r['nod']['support_s']:.3f}s；官方发声并集 {r['audio']['official_bout_count']} 段／{r['audio']['official_core_s']:.3f}s。", "",
                  "| 类型 | 状态／变化 | 次数依据 |", "|---|---|---|"]
        for g in r["groups"]:lines.append(f"| {display_name(g['key'])} | {','.join(g['forms'])} | {quantity_text(g)} |")
        lines += ["", "原始提取逐项处置（保留原句依据）：", ""]
        for b in r["evidence_ledger"]:
            lines += [f"- {b['raw']['id']} · {b['raw']['behavior']} · {b['disposition']} · {b['reason']}",
                      "  > " + " ".join(b["evidence"])]
        lines.append("")
    (out/"review50.md").write_text("\n".join(lines))
    if file_hash(db)!=dbhash:raise ValueError("Read-only DB changed")
    for path,sha in input_hashes.items():
        if file_hash(path)!=sha:raise ValueError("Input changed during statistics run")
    write(out/"provenance.json",dict(version=VERSION,script_sha256=file_hash(__file__),
          source_database=str(db),database_sha256=dbhash,inputs=input_hashes,aliases=ALIASES,
          preserved_samples=len(records),cpu_only=True,training_data_changed=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in {"catalog","reasons"}},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
