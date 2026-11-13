#!/usr/bin/env python3
"""Read-only ranking audit of an existing small pilot; no media/model processing."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repeat_bonus(row):
    # Evidence reward only. Unknown counts are never converted to zero occurrences.
    ignored={"visual:blinking","visual:eye closing","visual:smiling"}
    return sum(min(max(n-1,0),2) for k,n in row["known_occurrences_by_type"].items() if k not in ignored)


def order(records, counts=True):
    return sorted(records,key=lambda r:(-r["counted_type_count"],-repeat_bonus(r) if counts else 0,
                                        -(r["face_short_side_p10"] or 0),r["sample_id"]))


def stats(records):
    return dict(n=len(records),type_count_median=median(r["counted_type_count"] for r in records),
        duration_median_s=median(r["duration_s"] for r in records),
        duration_range_s=[min(r["duration_s"] for r in records),max(r["duration_s"] for r in records)],
        face_p10_median=median(r["face_short_side_p10"] for r in records),
        official_audio_supported_samples=sum(r["audio"]["official_core_s"]>0 for r in records),
        face_ge160_samples=sum(r["face_short_side_p10"]>=160 for r in records),
        pending_samples=sum(bool(r["pending_behavior_count"]) for r in records),
        splits=dict(Counter(r["split"] for r in records)))


def audit(records, n=10):
    if len(records)<2*n or len({r["sample_id"] for r in records})!=len(records):raise ValueError("Need distinct, disjoint extremes")
    ranked=order(records);control=order(records,False)
    top,bottom=ranked[:n],ranked[-n:]
    sid=lambda rs:{r["sample_id"] for r in rs}
    return ranked,dict(sample_count=len(records),top=stats(top),bottom=stats(bottom),
        top_ids=[r["sample_id"] for r in top],bottom_ids=[r["sample_id"] for r in bottom],
        no_count_top_overlap=len(sid(top)&sid(control[:n])),no_count_bottom_overlap=len(sid(bottom)&sid(control[-n:])),
        all_records_retained=True,source_labels_modified=False,
        preliminary_quality_eligible_count=sum(r["face_short_side_p10"]>=160 and r["face_sample_detection_fraction"]>=.98 and r["clarity_measured_fraction"]>=.8 for r in records),
        method="type_count DESC; capped known repeat reward DESC; native face p10 DESC; ID",
        duration_used_as_sort_key=False,duration_check="descriptive comparison only, not a causal control",
        unknown_count_policy="preserved null; no negative penalty; no-count sensitivity reported",
        face_filter_applied=False,training_selection_applied=False,
        independent_video_or_audio_review=False,semantic_accuracy_estimated=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path);p.add_argument("--notes",required=True,type=Path)
    a=p.parse_args();before=digest_file(a.input)
    records=[json.loads(line) for line in a.input.open() if line.strip()]
    notes=json.loads(a.notes.read_text());ranked,summary=audit(records)
    if a.out.exists() and any(a.out.iterdir()):raise ValueError("Output directory must be new")
    required=set(summary["top_ids"]+summary["bottom_ids"])
    if not required.issubset(notes["records"]):raise ValueError("Missing qualitative review of an extreme sample")
    if not set(notes["records"]).issubset({r["sample_id"] for r in records}):raise ValueError("Unknown review sample")
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    summary.update(input_sha256=before,script_sha256=digest_file(__file__),notes_sha256=digest_file(a.notes))
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    (out/"source_review.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2)+"\n")
    with (out/"ranking50.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["排名","样本ID","采用类型数","明确重复奖励","未知次数类型数","时长秒","人脸p10","待核条目数","分组"])
        for i,r in enumerate(ranked,1):
            w.writerow([i,r["sample_id"],r["counted_type_count"],repeat_bonus(r),len(r["unknown_occurrence_types"]),
                        r["duration_s"],r["face_short_side_p10"],r["pending_behavior_count"],"前10" if i<=10 else "后10" if i>len(ranked)-10 else "中间"])
    rows=[]
    for i,r in enumerate(ranked,1):rows.append(dict(rank=i,repeat_bonus=repeat_bonus(r),**r))
    (out/"ranking50.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    lines=["# 前10／后10排序效果检查","","## 结论","",notes["conclusion"],"",
        "这次逐条核对的是已有完整caption与提取统计，没有重新观看视频或听音。类型数差异本身由排序规则决定，不能当作准确率；以下原文复核用于检查排序是否符合文本内容。","",
        "## 排序口径","","1. 采用类型数降序。\n2. 同类型只对有依据的额外重复给奖励，最多2；眨眼、闭眼、微笑不靠次数获奖。\n3. 原始人脸短边p10降序；最后用ID稳定排序。","",
        "未知次数保持未知，局部待核不扣分。奖励缺失不等于事件没有发生；另做完全不使用次数的敏感性对照。本次未执行人脸门槛或训练准入，全部50段留在排序表。","",
        "## 两组概况","","| 指标 | 前10 | 后10 |","|---|---:|---:|"]
    for key,label in [("type_count_median","采用类型数中位数"),("duration_median_s","时长中位数／秒"),("face_p10_median","人脸p10中位数／像素"),
                      ("official_audio_supported_samples","有官方发声支持的片段"),("face_ge160_samples","人脸p10≥160的片段"),("pending_samples","含局部待核项的片段")]:
        lines.append(f"| {label} | {summary['top'][key]:.3f} | {summary['bottom'][key]:.3f} |")
    lines += ["",f"去掉次数奖励后，前10重合{summary['no_count_top_overlap']}/10，后10重合{summary['no_count_bottom_overlap']}/10。只说明本批次名单对次数奖励不敏感，不代表全量未知次数不会受影响。",
        "","两组时长中位数接近，前组也含短片段、后组也含长片段；未见本批排序被时长单独决定。但这不是经过匹配的时长因果检验。","",
        "## 前10逐条检查","","| 排名 | 样本 | 类型数 | 秒 | 人脸p10 | 原文复核 |","|---:|---|---:|---:|---:|---|"]
    for group in [rows[:10],rows[-10:]]:
        if group[0]["rank"]>10:lines += ["","## 后10逐条检查","","| 排名 | 样本 | 类型数 | 秒 | 人脸p10 | 原文复核 |","|---:|---|---:|---:|---:|---|"]
        for r in group:lines.append(f"| {r['rank']} | {r['sample_id']} | {r['counted_type_count']} | {r['duration_s']:.3f} | {r['face_short_side_p10']:.1f} | {notes['records'][r['sample_id']]} |")
    lines += ["","## 额外检查的中间位置反例",""]
    for r in rows:
        if r["sample_id"] in notes["records"] and r["sample_id"] not in required:
            lines += [f"- 第{r['rank']}名 `{r['sample_id']}`：{notes['records'][r['sample_id']]}"]
    lines += ["","## 使用建议","",notes["recommendation"],"","## 原始描述与采用项",""]
    for r in rows:
        if r["sample_id"] not in notes["records"]:continue
        lines += [f"### 第{r['rank']}名：{r['sample_id']}","",r["caption"],"","采用类型："+"、".join(r["counted_types"]),"",
                  "明确各类次数："+json.dumps(r["known_occurrences_by_type"],ensure_ascii=False),"",
                  "次数未知的类型："+"、".join(r["unknown_occurrence_types"]),""]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    if digest_file(a.input)!=before:raise ValueError("Source changed during audit")
    print(json.dumps({k:v for k,v in summary.items() if not k.endswith('_ids')},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
