"""Accepted dataset decisions, crop equivalence, and fail-closed evidence handling."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from listen_visual_quality.cli import load_policy
from listen_visual_quality.io import read, input_rows, canonical_hash
from listen_visual_quality.rules import analyze_focus, propose_crop
from listen_visual_quality.decision import decide, evidence_index


class VisualQualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy(ROOT / "configs/listen_visual_quality_v1.json")
        cls.rows = read(ROOT / "experiments/visual_quality_holdout120_20261030/data.json")["results"]
        cls.evidence = evidence_index(read(ROOT / "configs/listen_visual_quality_reviewed_evidence_20261030.json"))

    def test_existing_120_measurements_and_crop_proposals_identical(self):
        for old in self.rows:
            q = analyze_focus(copy.deepcopy(old["focus_samples"]), old["source"]["duration_s"], self.policy)
            for key in ("longest_low_s", "low_s", "low_fraction", "valid_samples", "total_samples", "intervals", "measured_fraction"):
                self.assertEqual(q[key], old["focus"][key], (old["ordinal"], key))
            self.assertEqual(propose_crop(old["raw_measurements"], old["source"], self.policy), old["crop"])

    def test_exact_user_v2_partition(self):
        excluded, review = [], []
        for old in self.rows:
            r = copy.deepcopy(old)
            r["focus"] = analyze_focus(r["focus_samples"], r["source"]["duration_s"], self.policy)
            d = decide(r, self.policy, self.evidence)
            if d["decision"] == "exclude":
                excluded.append(r["ordinal"])
            if d["decision"] == "review":
                review.append(r["ordinal"])
        self.assertEqual(excluded, [20, 39, 49, 52, 81])
        self.assertEqual(review, [])

    def test_old_accepted_low_head_is_not_excluded_by_coverage(self):
        old = read(ROOT / "experiments/visual_quality_crop_focus_20261030/data.json")["results"]
        r = next(r for r in old if r["ordinal"] == 10)
        r["status"] = "ok"
        r["raw_measurements"] = read(ROOT / "experiments/visual_quality_pilot_20261030/context_results" / (r["id"] + ".json"))["raw_measurements"]
        r["focus"] = analyze_focus(r["focus_samples"], r["source"]["duration_s"], self.policy)
        self.assertEqual(decide(r, self.policy, self.evidence)["decision"], "review")
        self.assertNotIn("clip_face_detection_failed", decide(r, self.policy, self.evidence)["reasons"])

    def test_failed_execution_is_not_silently_a_bad_video(self):
        r = copy.deepcopy(self.rows[0]);r["status"] = "error"
        self.assertEqual(decide(r, self.policy)["decision"], "review")

    def test_no_valid_face_detections_is_excluded(self):
        r = copy.deepcopy(self.rows[51])
        # A successful detector run with no observations is not an execution error.
        for q in r["raw_measurements"]:
            q["detected"] = False
        self.assertEqual(decide(r, self.policy)["decision"], "exclude")

    def test_tiny_detected_face_is_not_mislabeled_as_no_face(self):
        r = copy.deepcopy(self.rows[51])
        self.assertTrue(all(x["detected"] for x in r["raw_measurements"]))
        self.assertEqual(r["focus"]["valid_samples"], 0)
        self.assertEqual(decide(r, self.policy)["decision"], "review")
        self.assertEqual(decide(r, self.policy, self.evidence)["decision"], "exclude")

    def test_evidence_cannot_follow_an_id_to_other_times(self):
        r = copy.deepcopy(self.rows[48]);r["source"]["source_start_s"] += .1
        with self.assertRaises(ValueError):
            decide(r, self.policy, self.evidence)

    def test_no_detection_is_not_zero_sharpness(self):
        r = copy.deepcopy(self.rows[51])
        q = analyze_focus(r["focus_samples"], r["source"]["duration_s"], self.policy)
        self.assertEqual(q["valid_samples"], 0)
        self.assertEqual(q["intervals"], [])
        self.assertIsNone(q["laplacian_reference_p80"])
        self.assertTrue(all(x["lap_ratio"] is None for x in r["focus_samples"]))

    def test_duplicate_manifest_ids_rejected(self):
        row = {"sample_id": "test", "source": {"sample_id": "test"}}
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                list(input_rows(path))

    def test_unknown_evidence_cannot_be_treated_as_review(self):
        evidence = {"records": [{"sample_id": "test", "source_sha256": "a", "provenance": "test", "exclude_reasons": ["magic_rule"]}]}
        with self.assertRaises(ValueError):
            evidence_index(evidence)


if __name__ == "__main__":
    unittest.main()
