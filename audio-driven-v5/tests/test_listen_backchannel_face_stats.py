import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from listen_benchmark import backchannel_face_stats as m


def fixture_rows():
    full = dict(sample_id="s1", pair="pair1", group_id="g1", split="train", listener_id="l1", speaker_id="p1",
                start=2., end=12., sources={"listener": {"archive": "a", "member": "l1.mp4"}},
                normalized_audio={"listener": {"path": "l1.wav"}},
                face_geometry={"listener": dict(status="measured", width=1000, height=500,
                    valid_frames=290, expected_frames=300, valid_fraction=290/300,
                    face_short_side_px={"p10": 128., "p50": 160.},
                    face_width_px={"p10": 150., "p50": 180.}, face_height_px={"p10": 128., "p50": 160.},
                    face_area_fraction={"p10": .04, "p50": .06})})
    source = dict(sample_id="s1", split="train", listener_id="l1", speaker_id="p1", source_start_s=2., source_end_s=12.,
                  source=full["sources"]["listener"], normalized_audio={"path": "l1.wav"}, source_manifest="pending",
                  source_line_1based=1, source_line_sha256="pending",
                  video_transform={"crop": {"x": 0, "y": 0, "width": 512, "height": 512}})
    def event(eid, start, end, cats, modality="visual"):
        return dict(event_id=eid, start=start, end=end, categories=cats, modality=modality,
                    description="原文 Yeah，不作替换", duplicate_observations_removed=1)
    candidate = dict(sample_id="s1", source=source, duration_s=10.,
                     events=[event("e1", 1., 3., ["smile", "laugh"]), event("e2", 2., 4., ["smile"]),
                             event("e3", 5., 6., ["listener_vocalization"], "audio")],
                     response_groups=[{"event_ids": ["e1", "e2"]}], original_observation_count=8,
                     raw_issue_count=0, semantic_issue_count=0)
    return full, candidate


def test_counts_union_and_no_content_mutation():
    full, candidate = fixture_rows()
    original = copy.deepcopy(candidate)
    row = m.sample_stats(candidate, full)
    assert row["event_count"] == 3  # not four categories, or four including group
    assert row["smile_count"] == 2 and row["laugh_count"] == 1
    assert row["event_coverage_s"] == 4 and row["visual_coverage_s"] == 3
    assert row["audio_coverage_s"] == 1 and row["event_coverage_fraction"] == .4
    assert row["face_short_side_p10"] == 128 and row["crop_width"] == 512
    assert row["face_area_px2_p50"] == 30000  # not independent W_p50*H_p50=28800
    assert row["non_smile_event_count"] == 2
    assert candidate == original


def test_missing_face_never_uses_crop_and_invalid_events_are_flagged():
    full, candidate = fixture_rows()
    full["face_geometry"] = {}
    candidate["events"][0]["start"] = -1
    candidate["events"][1]["end"] = float("nan")
    row = m.sample_stats(candidate, full)
    assert row["face_short_side_p10"] is None
    assert row["event_count"] == 3 and row["valid_timed_event_count"] == 1
    assert row["label_issue_flag"] == 1 and row["event_coverage_s"] == 1


def test_duplicate_event_ids_fail():
    full, candidate = fixture_rows()
    candidate["events"].append(copy.deepcopy(candidate["events"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        m.sample_stats(candidate, full)


def make_run(root):
    fusion, feature = root/"fusion", root/"feature"
    (fusion/"labels").mkdir(parents=True)
    (feature/"source_index").mkdir(parents=True)
    full, candidate = fixture_rows()
    source_file = root/"source.jsonl"
    line = (json.dumps(full, ensure_ascii=False)+"\n").encode()
    source_file.write_bytes(line)
    candidate["source"].update(source_manifest=str(source_file), source_line_sha256=hashlib.sha256(line).hexdigest())
    st = source_file.stat()
    db = feature/"source_index/offsets.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE offsets(file_id INTEGER,line INTEGER,offset INTEGER,PRIMARY KEY(file_id,line))")
        conn.execute("INSERT INTO offsets VALUES (0,1,0)")
    m.atomic_json(feature/"source_index/COMPLETE.json", dict(files={str(source_file):dict(file_id=0,identity=dict(size=st.st_size,mtime_ns=st.st_mtime_ns))},db_sha256=m.sha(db)))
    label = fusion/"labels/0000.jsonl"
    label.write_text(json.dumps(candidate,ensure_ascii=False)+"\n")
    entry = dict(shard="0000", done=True, labels="labels/0000.jsonl", counts=dict(samples=1,events=3),
                 files={"labels/0000.jsonl":dict(sha256=m.sha(label))})
    m.atomic_json(fusion/"index.json",dict(samples=1,shards=[entry]))
    m.atomic_json(fusion/"COMPLETE.json",dict(samples=1,index_sha256=m.sha(fusion/"index.json")))
    m.atomic_json(feature/"index.json",dict(samples=1))
    m.atomic_json(feature/"COMPLETE.json",dict(samples=1,index_sha256=m.sha(feature/"index.json")))
    m.atomic_json(feature/"config.json",dict(fusion_run=str(fusion)))
    return fusion, feature, candidate


def test_end_to_end_build_filter_inspect_resume_and_hash_guards(tmp_path):
    fusion, feature, original = make_run(tmp_path)
    output = tmp_path/"stats"
    args = argparse.Namespace(fusion_run=str(fusion), feature_run=str(feature), output=str(output), workers=1, resume=False)
    m.build(args)
    assert m.read(output/"summary.json")["events"] == 3
    assert m.read(output/"COMPLETE.json")["samples"] == 1
    inspected = m.inspect_sample(output,"s1")
    assert inspected["integrated_label"] == original
    selection = argparse.Namespace(stats=str(output), output=str(tmp_path/"selection"), min_events=3, min_density=0,
        min_face_p10=128, min_face_p50=160, min_face_valid=.9, exclude_label_issues=True, split="train", sort="events", limit=None)
    m.select(selection)
    assert (tmp_path/"selection/sample_ids.txt").read_text() == "s1\n"
    selection.min_face_p10=129; selection.output=str(tmp_path/"empty")
    m.select(selection)
    assert m.read(tmp_path/"empty/selection.json")["samples"] == 0
    args.resume=True; m.build(args)
    reader=m.SourceReader(feature)
    bad=copy.deepcopy(original["source"]);bad["source_line_sha256"]="bad"
    with pytest.raises(ValueError,match="hash"):
        reader.get(bad)
    bad=copy.deepcopy(original["source"]);bad["listener_id"]="wrong"
    with pytest.raises(ValueError,match="listener_id"):
        reader.get(bad)
    reader.close()
    label=fusion/"labels/0000.jsonl"
    label.write_text(label.read_text().replace("Yeah", "YEAH"))
    with pytest.raises(ValueError,match="Referenced label changed"):
        m.inspect_sample(output,"s1")


def test_shard_integrity_failure_prevents_complete(tmp_path):
    fusion, feature, _ = make_run(tmp_path)
    label=fusion/"labels/0000.jsonl"
    label.write_text(label.read_text()+"\n")
    with pytest.raises(Exception):
        m.build(argparse.Namespace(fusion_run=str(fusion),feature_run=str(feature),output=str(tmp_path/"bad"),workers=1,resume=False))
    assert not (tmp_path/"bad/COMPLETE.json").exists()


def test_invalid_selection_threshold_rejected():
    args=argparse.Namespace(min_events=0,min_density=float("nan"),min_face_p10=0,min_face_p50=0,min_face_valid=.9,limit=None)
    with pytest.raises(ValueError,match="Thresholds"):
        m.selection_query(args)
