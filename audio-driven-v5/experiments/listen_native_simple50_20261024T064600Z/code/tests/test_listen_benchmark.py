"""Synthetic-media regressions; no pretrained model inference or external API."""
import copy
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import av
import numpy as np
from PIL import Image
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from listen_benchmark.assets import inspect_weights
from listen_benchmark.backends import hf_inputs
from listen_benchmark.evaluate import evaluate_system, match_events, subtract
from listen_benchmark.media import prepare, minicpm_native, uniform_video
from listen_benchmark.runner import select_samples, run_system, review_template, validate_config
from listen_labeling.client import prompt
from listen_benchmark.simple_timeline import validate_timeline
from p2_listen.contracts import atomic_json, read

DURATION = 1.28


def label(branch="omni", stage="audit"):
    keys = (["visual", "audio"] if branch == "omni" else [branch]) if stage == "discovery" else (
        ["head_nod", "listener_vocalization"] if branch == "omni" else
        ["head_nod" if branch == "visual" else "listener_vocalization"])
    events = [] if stage == "discovery" else [dict(type=k, start=.25, end=.5, owner="listener",
        confidence=1., description="synthetic", boundary_censored=False) for k in keys]
    return dict(reviewed={k: [[0., DURATION]] for k in keys}, unknown={k: [] for k in keys}, events=events, notes="")


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        video, audio = self.root / "v.mp4", self.root / "a.wav"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=25)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for i in range(32):
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 5, np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        sf.write(audio, np.sin(np.arange(20480) * .1) * .05, 16000)
        row = dict(sample_id="sample1", duration_s=DURATION, target_video=str(video), target_audio=str(audio),
                   experiment_split="train", time_origin="clip_start_seconds", video_role="listener",
                   audio_role="listener", audio_isolated=True, video_transform=dict(
                       crop=dict(x=4, y=4, width=24, height=24), source_size=[32, 32], output_size=[512, 512], resample="bicubic"))
        self.manifest = self.root / "manifest.json"
        atomic_json(self.manifest, [row])
        self.samples = select_samples(self.manifest, 1)
        self.config = read(ROOT / "configs/listen_benchmark.json")
        self.system = next(s for s in self.config["systems"] if s["name"] == "qwen3_omni_native")
        self.config["systems"] = [self.system]
        self.calls, self.failure_stage, self.finish = [], None, "stop"
        outer = self

        class Fake:
            def infer(self, media, text, stage, fps):
                outer.calls.append((text, stage, media["meta"]))
                value = "{bad-json" if outer.failure_stage == stage else json.dumps(label(stage=stage))
                return {"text": value, "finish_reason": outer.finish}
        self.factory = lambda *_: Fake()

    def run_benchmark(self, retry=False):
        with redirect_stdout(io.StringIO()):
            return run_system(self.config, self.system["name"], self.samples, self.root / "out", "cpu",
                              retry_failed=retry, backend_factory=self.factory)

    def fixture_gt(self):
        row = review_template(self.samples)[0]
        row.update(human_verified=True, reviewer="synthetic-reviewer", reviewed_at="2026-10-24",
                   evaluation_partition="heldout", stratum="random", label=label())
        return [row]

    def test_occurrence_prompt_runs_discovery_only_and_persists_exact_request(self):
        self.system.update(prompt_variant="backchannel_occurrences_v1", annotation_stages=["discovery"])
        result = self.run_benchmark()
        self.assertEqual(result["expected_requests"], 1)
        self.assertEqual([call[1] for call in self.calls], ["discovery"])
        self.assertEqual(result["results"][0]["status"], "schema_pass")
        folder = self.root / "out" / self.system["name"]
        saved = read(folder / "requests/sample1/discovery/request.json")
        self.assertEqual(saved["prompt"], self.calls[0][0])
        self.assertEqual(saved["system"]["prompt_variant"], "backchannel_occurrences_v1")
        # A changed prompt cannot silently reuse a completed experiment identity.
        self.system["prompt_variant"] = "baseline"
        with self.assertRaises(ValueError):
            self.run_benchmark()

    def test_default_benchmark_keeps_original_prompts_and_both_stages(self):
        result = self.run_benchmark()
        self.assertEqual(result["expected_requests"], 2)
        for text, stage, _ in self.calls:
            self.assertEqual(text, prompt("omni", DURATION, stage))

    def test_simple_prompt_keeps_only_actual_minimal_events(self):
        self.system.update(prompt_variant="backchannel_simple_v2", annotation_stages=["discovery"])
        expected = {"events": [{"start": .2, "end": .4, "modality": "visual", "behavior": "nods"}]}
        class Fake:
            def infer(inner, media, text, stage, fps):
                self.calls.append((text, stage, media["meta"]))
                return dict(text=json.dumps(expected), finish_reason="stop")
        self.factory = lambda *_: Fake()
        report = self.run_benchmark()
        item = report["results"][0]
        self.assertEqual(item["status"], "schema_pass")
        self.assertEqual(item["parsed"], expected)
        self.assertEqual(item["output_contract"], "minimal-event-timeline-v1")
        self.assertEqual(len(self.calls), 1)
        self.assertLess(len(self.calls[0][0].split()), 80)
        self.assertNotIn("OUTPUT CONTRACT", self.calls[0][0])

    def test_minimal_timeline_rejects_invalid_times_without_repair(self):
        for start, end in ((True, .4), (float('nan'), .4), (.5, .5), (.6, .4), (-.1, .5), (0, DURATION+1)):
            with self.assertRaises(ValueError):
                validate_timeline({"events": [dict(start=start, end=end, modality="audio", behavior="okay")]}, DURATION)
        self.assertEqual(validate_timeline({"events": []}, DURATION), {"events": []})

    def test_reject_invalid_experimental_stage_and_prompt_combinations(self):
        for stages in ([], ["discovery", "discovery"], ["typo"], "discovery"):
            cfg = copy.deepcopy(self.config)
            cfg["systems"][0]["annotation_stages"] = stages
            with self.assertRaisesRegex(ValueError, "annotation_stages"):
                validate_config(cfg)
        self.system["prompt_variant"] = "backchannel_occurrences_v1"
        with self.assertRaisesRegex(ValueError, "discovery only"):
            validate_config(self.config)

    def test_captioner_uses_audio_only_default_template(self):
        class Processor:
            def apply_chat_template(inner, messages, **kwargs):
                self.assertNotIn("enable_thinking", kwargs)
                self.assertEqual([p["type"] for p in messages[0]["content"]], ["audio"])
                return "default-captioner-template"
            def __call__(inner, **kwargs):
                return {"input_features": np.ones((1, 2))}
        system = dict(backend="captioner", branch="audio", representation="separate")
        media = prepare(self.samples[0], "audio", self.config["media"])
        _, template, _ = hf_inputs(Processor(), system, media, None, 8)
        self.assertEqual(template, "default-captioner-template")

    def test_native_audio_segments_preserve_tail_and_all_frames(self):
        media = prepare(self.samples[0], "omni", self.config["media"])
        def concat(images):
            out = Image.new("RGB", (512 * len(images), 512))
            for i, image in enumerate(images):
                out.paste(image, (512 * i, 0))
            return out
        content, evidence = minicpm_native(media, concat)
        chunks = [p for p in content if isinstance(p, np.ndarray)]
        np.testing.assert_array_equal(np.concatenate(chunks), media["audio"])
        self.assertEqual(len(chunks[-1]), 4480)
        self.assertEqual([i for s in evidence["segments"] for i in s["frame_indices"]], list(range(10)))
        self.assertFalse(evidence["tail_padded"])

    def test_native_audio_only_tail_does_not_drop_or_duplicate_evidence(self):
        media = prepare(self.samples[0], "omni", self.config["media"])
        media["duration_s"] = 1.005
        media["audio"] = np.arange(16080, dtype=np.float32)
        media["images"] = media["images"][:8]
        media["meta"]["video"]["frames"] = media["meta"]["video"]["frames"][:8]
        media["meta"]["video"]["last_frame_s"] = .96
        content, evidence = minicpm_native(media, lambda images: images[0])
        np.testing.assert_array_equal(np.concatenate([p for p in content if isinstance(p, np.ndarray)]), media["audio"])
        self.assertEqual(evidence["segments"][-1]["audio_samples"], [16000, 16080])
        self.assertEqual(evidence["segments"][-1]["frame_indices"], [])
        self.assertTrue(evidence["segments"][-1]["audio_tail_only"])
        # Missing visual evidence in a nonterminal second still fails explicitly.
        media["duration_s"] = 2.005
        media["audio"] = np.arange(32080, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "No listener frame"):
            minicpm_native(media, lambda images: images[0])

    def test_native_qwen_records_pts_offset_and_rejects_missing_grid(self):
        media = prepare(self.samples[0], "omni", self.config["media"])
        array, evidence = uniform_video(media, 8)
        self.assertEqual(array.shape, (10, 512, 512, 3))
        self.assertAlmostEqual(evidence["native_grid_tail_gap_s"], .03)
        self.assertAlmostEqual(evidence["frame_pts_s"][1], .16)
        self.assertAlmostEqual(evidence["grid_offset_s"][1], .035)
        media["meta"]["video"]["frames"].pop()
        with self.assertRaisesRegex(ValueError, "grid"):
            uniform_video(media, 8)

    def test_qwen_native_processor_flags_and_independent_audio(self):
        media = prepare(self.samples[0], "omni", self.config["media"])
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "template"
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": np.zeros((1, 12)), "input_features": np.zeros((1, 128, 10)),
                        "pixel_values_videos": np.zeros((1, 3, 32, 32))}
        processor = Processor()
        hf_inputs(processor, self.system, media, "instruction", 8)
        self.assertTrue(processor.kwargs["use_audio_in_video"])
        self.assertFalse(processor.kwargs["do_sample_frames"])
        self.assertEqual(processor.kwargs["fps"], 8)
        self.assertIs(processor.kwargs["audio"][0], media["audio"])
        self.assertEqual([x["type"] for x in processor.messages[0]["content"]], ["text", "video"])

    def test_captioner_never_receives_text_prompt(self):
        media = prepare(self.samples[0], "audio", self.config["media"])
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "audio only"
            def __call__(self, **kwargs):
                return {"input_ids": np.zeros((1, 12)), "input_features": np.zeros((1, 128, 10))}
        processor = Processor()
        system = dict(self.system, backend="captioner", branch="audio", representation="separate")
        hf_inputs(processor, system, media, "MUST NOT SEND", 8)
        self.assertEqual([p["type"] for p in processor.messages[0]["content"]], ["audio"])

    def test_ignored_audio_argument_is_a_hard_failure(self):
        media = prepare(self.samples[0], "audio", self.config["media"])
        class TextOnlyProcessor:
            def apply_chat_template(self, *args, **kwargs):
                return "text"
            def __call__(self, text=None, audio=None, **kwargs):
                return {"input_ids": np.zeros((1, 10))}
        system = dict(self.system, backend="qwen2_audio", branch="audio", representation="separate")
        with self.assertRaisesRegex(ValueError, "dropped listener audio"):
            hf_inputs(TextOnlyProcessor(), system, media, "instruction", 8)

    def test_audit_runs_when_discovery_fails_and_has_no_previous_answer(self):
        self.failure_stage = "discovery"
        report = self.run_benchmark()
        self.assertEqual([r["status"] for r in report["results"]], ["failed", "schema_pass"])
        self.assertEqual([x[1] for x in self.calls], ["discovery", "audit"])
        self.assertTrue(self.calls[1][0].startswith("Task: primary_audit"))
        self.assertNotIn("bad-json", self.calls[1][0])
        self.assertEqual(self.calls[0][2], self.calls[1][2])

    def test_resume_does_not_retry_failures_unless_explicit_and_keeps_attempts(self):
        self.failure_stage = "discovery"
        self.run_benchmark()
        self.failure_stage = None
        self.run_benchmark()
        self.assertEqual(len(self.calls), 2)
        report = self.run_benchmark(retry=True)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(report["results"][0]["first_attempt_status"], "failed")
        self.assertEqual(report["results"][0]["status"], "schema_pass")
        self.assertEqual(report["results"][0]["attempts"], 2)

    def test_config_change_requires_new_output(self):
        self.run_benchmark()
        self.config["generation"]["max_new_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.run_benchmark()

    def test_resume_recovers_raw_response_after_interruption(self):
        self.run_benchmark()
        path = self.root / "out" / self.system["name"] / "requests/sample1/discovery/attempt-0001.json"
        path.unlink()  # Simulate a crash after raw was persisted, before outcome.
        report = self.run_benchmark()
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(report["results"][0]["recovered_raw_without_regeneration"])
        self.assertEqual(report["results"][0]["status"], "schema_pass")

    def test_truncated_raw_is_preserved_not_accepted(self):
        self.finish = "length"
        report = self.run_benchmark()
        self.assertTrue(all(r["status"] == "failed" for r in report["results"]))
        self.assertEqual(len(list((self.root / "out").glob("*/requests/*/*/raw-*.json"))), 2)

    def test_media_change_blocks_generation(self):
        sf.write(self.samples[0]["target_audio"], np.zeros(20480), 16000)
        report = self.run_benchmark()
        self.assertEqual(self.calls, [])
        self.assertTrue(all(r["status"] == "failed" for r in report["results"]))

    def test_empty_review_template_is_not_gt(self):
        report = self.run_benchmark()
        inputs = read(self.root / "out" / self.system["name"] / "inputs.json")
        template = review_template(self.samples)
        template[0]["evaluation_partition"] = "heldout"
        with self.assertRaisesRegex(ValueError, "human_verified"):
            evaluate_system(report, inputs, template)

    def test_failed_audit_counts_false_negatives(self):
        self.failure_stage = "audit"
        report = self.run_benchmark()
        inputs = read(self.root / "out" / self.system["name"] / "inputs.json")
        metrics = evaluate_system(report, inputs, self.fixture_gt())["groups"]["all"]["classes"]
        for value in metrics.values():
            self.assertEqual(value["metrics"]["0.3"]["fn"], 1)
            self.assertEqual(value["metrics"]["0.3"]["recall"], 0)
            self.assertEqual(value["supported_fraction"], 0)

    def test_unknown_does_not_remove_gt_from_recall_denominator(self):
        report = self.run_benchmark()
        inputs = read(self.root / "out" / self.system["name"] / "inputs.json")
        audit = report["results"][1]["parsed"]
        audit["events"] = []
        audit["unknown"] = {k: [[0, DURATION]] for k in audit["reviewed"]}
        report = evaluate_system(report, inputs, self.fixture_gt())
        for value in report["groups"]["all"]["classes"].values():
            self.assertEqual(value["metrics"]["0.3"]["fn"], 1)
            self.assertEqual(value["supported_fraction"], 0)

    def test_duplicate_detection_is_false_positive(self):
        report = self.run_benchmark()
        inputs = read(self.root / "out" / self.system["name"] / "inputs.json")
        report["results"][1]["parsed"]["events"].append(copy.deepcopy(report["results"][1]["parsed"]["events"][0]))
        metrics = evaluate_system(report, inputs, self.fixture_gt())["groups"]["all"]["classes"]
        self.assertEqual(metrics["head_nod"]["metrics"]["0.3"]["fp"], 1)
        event = dict(start=.25, end=.5)
        self.assertEqual(len(match_events([event, event], [event], .3)), 1)

    def test_gt_crop_mismatch_rejected(self):
        report = self.run_benchmark()
        inputs = read(self.root / "out" / self.system["name"] / "inputs.json")
        gt = self.fixture_gt(); gt[0]["video_transform"]["crop"]["x"] += 1
        with self.assertRaisesRegex(ValueError, "crop"):
            evaluate_system(report, inputs, gt)

    def test_maximum_cardinality_matching_and_interval_union(self):
        a, b = dict(start=0., end=1.), dict(start=.8, end=1.8)
        broad, first = dict(start=0., end=1.8), dict(start=0., end=.7)
        self.assertEqual(len(match_events([broad, first], [a, b], .3)), 2)
        self.assertEqual(subtract([[0, 1], [.5, 2]], [[.75, 1.25]]), [[0, .75], [1.25, 2]])

    def test_weight_inventory_detects_empty_and_truncated_shards(self):
        model = self.root / "model"; model.mkdir()
        self.assertFalse(inspect_weights(model)["ready"])
        atomic_json(model / "config.json", {"model_type": "test"})
        header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        path = model / "model.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        self.assertTrue(inspect_weights(model)["ready"])
        path.write_bytes(path.read_bytes()[:-1])
        self.assertFalse(inspect_weights(model)["ready"])


if __name__ == "__main__":
    unittest.main()
