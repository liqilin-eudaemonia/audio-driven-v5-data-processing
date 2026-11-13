"""Synthetic media + real local HTTP transport; no pretrained weights or external APIs."""
import base64
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
import av
import numpy as np
from PIL import Image
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from listen_labeling.client import prompt, validate_endpoint
from listen_labeling.media import frames, waveform, crop_resize, validate_transform
from listen_labeling.realtalk import prepare_package
from listen_labeling.seamless import prepare_package as prepare_seamless, integer_crop
from listen_labeling.pipeline import export, preflight, run
from listen_labeling.schema import parse, validate_response
from p2_listen.contracts import atomic_json, digest, read, sha
from p2_listen.dataset import prepare, import_labels
from p2_listen.annotations import rasterize
import torch

DURATION = 1.28


def answer(branch, stage="audit"):
    if stage == "discovery":
        modalities = ["visual", "audio"] if branch == "omni" else [branch]
        return dict(reviewed={m: [[0, DURATION]] for m in modalities}, unknown={m: [] for m in modalities},
                    events=[dict(start=.6, end=.9, modality=m, owner="listener",
                                 behavior="自由描述_嘴唇轻抿后身体稍前倾" if m == "visual" else "自由描述_短促气声带尾部颤动",
                                 evidence="observed temporal change", uncertainty="", boundary_censored=False,
                                 interpretation="hypothesis only: possible engagement", confidence=.6) for m in modalities], notes="fixture")
    names = ["head_nod"] if branch == "visual" else ["listener_vocalization"]
    if branch == "omni":
        names = ["head_nod", "listener_vocalization"]
    events = [dict(type=name, start=.25 if name == "head_nod" else .3,
                   end=.75 if name == "head_nod" else .4, owner="listener",
                   description="synthetic test only", confidence=.8, boundary_censored=False) for name in names]
    return dict(reviewed={n: [[0, DURATION]] for n in names}, unknown={n: [] for n in names}, events=events, notes="fixture")


class LabelingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video, self.audio = self.root / "v.mp4", self.root / "a.wav"
        with av.open(str(self.video), "w") as container:
            stream = container.add_stream("mpeg4", rate=25)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for i in range(32):
                f = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 5, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(f):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        sf.write(self.audio, np.sin(np.arange(20480) * .1).astype(np.float32) * .05, 16000)
        self.row = dict(sample_id="s1", experiment_split="train", target_video=str(self.video), target_audio=str(self.audio),
                        duration_s=DURATION, time_origin="clip_start_seconds", video_role="listener", audio_role="listener", audio_isolated=True)
        self.manifest = self.root / "manifest.json"
        atomic_json(self.manifest, [self.row])
        self.cfg = read(Path(__file__).resolve().parents[1] / "configs/listen_labeling.json")
        self.requests, self.failures, self.responses = [], {}, {}
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(data)
                branch = data["model"]
                stage = "discovery" if data["messages"][0]["content"][0]["text"].startswith("Task: open_discovery") else "audit"
                key = (branch, stage)
                status = outer.failures.pop(key, 200)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                value = outer.responses.get(key, answer(branch, stage))
                self.wfile.write(json.dumps(dict(choices=[dict(finish_reason="stop", message={"content": json.dumps(value)})])).encode())
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        for name, endpoint in self.cfg["endpoints"].items():
            endpoint.update(base_url=f"http://127.0.0.1:{self.server.server_port}/v1", model=name, attempts=1)
        self.out = self.root / "output"

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.tmp.cleanup()

    def test_http_payload_resume_and_rich_events(self):
        result = run(self.manifest, self.cfg, self.out)
        self.assertEqual((result["succeeded"], result["failed"]), (1, 0))
        self.assertEqual(len(self.requests), 4)
        visual, audio = [r["messages"][0]["content"] for r in self.requests[:2]]
        self.assertNotIn("input_audio", [x["type"] for x in visual])
        self.assertNotIn("image_url", [x["type"] for x in audio])
        encoded = next(x["input_audio"]["data"] for x in audio if x["type"] == "input_audio")
        info = sf.info(io.BytesIO(base64.b64decode(encoded)))
        self.assertEqual((info.frames, info.samplerate), (20480, 16000))
        summary = read(result["summary"])
        label = read(summary["items"][0]["draft"])
        self.assertEqual(len(label["events"]), 2)
        self.assertEqual(label["events"][1]["end"] - label["events"][1]["start"], .4 - .3)
        self.assertFalse(label["provenance"]["human_verified"])
        rich = read(Path(result["summary"]).parent / "rich/s1.json")
        self.assertEqual(rich["visual"]["discovery"]["events"][0]["behavior"], "自由描述_嘴唇轻抿后身体稍前倾")
        self.assertNotIn("type", rich["visual"]["discovery"]["events"][0])
        run(self.manifest, self.cfg, self.out)
        self.assertEqual(len(self.requests), 4)
        self.cfg["endpoints"]["audio"]["max_tokens"] += 1
        run(self.manifest, self.cfg, self.out)
        self.assertEqual([r["model"] for r in self.requests], ["visual", "audio", "visual", "audio", "audio", "audio"])

    def test_failure_resume_keeps_successful_branch(self):
        self.failures[("audio", "audit")] = 503
        first = run(self.manifest, self.cfg, self.out)
        self.assertEqual(first["failed"], 1)
        self.assertNotIn("draft", read(first["summary"])["items"][0])
        open_path = read(first["summary"])["items"][0]["open_events"]["audio"]
        self.assertTrue(read(open_path)["events"])
        second = run(self.manifest, self.cfg, self.out)
        self.assertEqual(second["failed"], 0)
        self.assertEqual([r["model"] for r in self.requests], ["visual", "audio", "visual", "audio", "audio"])

    def test_preflight_and_media_guards(self):
        self.assertFalse(preflight(self.manifest, self.cfg, "split")["network_called"])
        self.assertEqual(self.requests, [])
        with self.assertRaisesRegex(ValueError, "budget"):
            frames(self.video, DURATION, dict(self.cfg["media"], max_frames=2))
        sf.write(self.audio, np.zeros((20480, 2)), 16000)
        with self.assertRaisesRegex(ValueError, "mono"):
            waveform(self.audio, DURATION, self.cfg["media"])
        self.row["audio_isolated"] = False
        atomic_json(self.manifest, [self.row])
        with self.assertRaisesRegex(ValueError, "isolated"):
            preflight(self.manifest, self.cfg, "split")

    def test_invalid_time_and_ownership_never_merge(self):
        value = answer("audio")
        value["events"][0]["end"] = DURATION + .1
        self.responses[("audio", "audit")] = value
        result = run(self.manifest, self.cfg, self.out)
        self.assertEqual(result["failed"], 1)
        value = answer("audio")
        value["events"][0]["owner"] = "speaker"
        with self.assertRaisesRegex(ValueError, "owner"):
            validate_response(value, "audio", DURATION)

    def test_unknown_and_missing_coverage_mask_negatives(self):
        visual = answer("visual")
        visual.update(reviewed={"head_nod": []}, unknown={"head_nod": [[0, DURATION]]}, events=[])
        self.responses[("visual", "audit")] = visual
        result = run(self.manifest, self.cfg, self.out)
        label = read(read(result["summary"])["items"][0]["draft"])
        grid = torch.tensor([[[0., .25], [.25, .5], [.5, .75], [.75, 1.]]])
        targets = rasterize(label, dict(video_intervals=grid, audio_intervals=grid))
        self.assertFalse(targets["head_nod"]["known"].any())
        self.assertTrue(targets["listener_vocalization"]["known"].all())

    def test_omni_gets_both_modalities(self):
        result = run(self.manifest, self.cfg, self.out, "omni")
        self.assertEqual(result["failed"], 0)
        types = {x["type"] for x in self.requests[0]["messages"][0]["content"]}
        self.assertTrue({"image_url", "input_audio"} <= types)
        self.cfg["endpoints"]["omni"]["capabilities"] = ["image_url"]
        with self.assertRaisesRegex(ValueError, "support"):
            run(self.manifest, self.cfg, self.out, "omni")

    def test_export_review_and_actual_p2_import(self):
        result = run(self.manifest, self.cfg, self.out)
        with self.assertRaisesRegex(ValueError, "No eligible"):
            export(result["summary"], self.root / "none.json")
        weak = self.root / "weak.json"
        export(result["summary"], weak, allow_weak_train=True)
        self.assertFalse(read(weak)[0]["provenance"]["human_verified"])
        summary = read(result["summary"])
        label = read(summary["items"][0]["draft"])
        record = dict(sample_id="s1", draft_sha256=digest(label), label=copy.deepcopy(label), human_verified=True,
                      reviewer="test-human", reviewed_at="2026-10-23T12:00:00Z")
        record["label"]["events"][0]["end"] = .8
        reviews = self.root / "reviews.json"
        atomic_json(reviews, [record])
        verified = self.root / "verified.json"
        export(result["summary"], verified, reviews)
        self.assertTrue(read(verified)[0]["provenance"]["human_verified"])
        # Exercise existing prepare/import code on dev, including its weak-label rejection.
        image = self.root / "i.png"
        Image.new("RGB", (32, 32)).save(image)
        obs = dict(schema="p2-listen-observation-v1", sample_id="s1", experiment_split="dev", official_split="dev",
                   group_id="g1", grouping_ids=["participant:1"], speaker_audio=str(self.audio), listener_image=str(image),
                   requested_duration_s=DURATION, geometry=dict(fps=25, height=32, width=32),
                   media_hashes=dict(speaker_audio=sha(self.audio), listener_image=sha(image)))
        target = dict(schema="p2-listen-target-v1", sample_id="s1", target_video=str(self.video), target_audio=str(self.audio),
                      duration_s=DURATION, admitted=True, media_hashes=label["provenance"]["media_hashes"])
        atomic_json(self.root / "obs.json", [obs]); atomic_json(self.root / "target.json", [target])
        p2cfg = dict(output=str(self.root / "p2"), data=dict(fps=25, size=32, min_seconds=1, max_frames=201))
        prepare(p2cfg, self.root / "obs.json", self.root / "target.json")
        with self.assertRaisesRegex(ValueError, "human verification"):
            import_labels(p2cfg, weak)
        self.assertEqual(import_labels(p2cfg, verified)["imported"], ["s1"])
        record["draft_sha256"] = "wrong"
        atomic_json(reviews, [record])
        with self.assertRaisesRegex(ValueError, "bind"):
            export(result["summary"], self.root / "bad.json", reviews)

    def test_dev_not_exported_as_weak(self):
        self.row["experiment_split"] = "dev"
        atomic_json(self.manifest, [self.row])
        result = run(self.manifest, self.cfg, self.out)
        with self.assertRaisesRegex(ValueError, "No eligible"):
            export(result["summary"], self.root / "bad.json", allow_weak_train=True)

    def test_changed_media_invalidates_only_audio_cache(self):
        run(self.manifest, self.cfg, self.out)
        sf.write(self.audio, np.zeros(20480), 16000)
        run(self.manifest, self.cfg, self.out)
        self.assertEqual([r["model"] for r in self.requests], ["visual", "audio", "visual", "audio", "audio", "audio"])

    def test_bounded_transient_retry(self):
        self.cfg["endpoints"]["audio"]["attempts"] = 2
        self.failures[("audio", "audit")] = 429
        with patch("listen_labeling.client.time.sleep"):
            result = run(self.manifest, self.cfg, self.out)
        self.assertEqual(result["failed"], 0)
        self.assertEqual([r["model"] for r in self.requests], ["visual", "audio", "visual", "audio", "audio"])

    def test_strict_json(self):
        self.assertEqual(parse('```json\n{"a": 1}\n```'), {"a": 1})
        for bad in ('{"a":1,"a":2}', '{"a":NaN}', 'prefix {"a":1}'):
            with self.assertRaises(ValueError):
                parse(bad)
        value = answer("visual")
        value["reviewed"]["head_nod"] = []
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_response(value, "visual", DURATION)

    def test_review_template_recovers_new_successes(self):
        atomic_json(self.manifest, [self.row, dict(self.row, sample_id="s2")])
        self.failures[("audio", "audit")] = 503
        first = run(self.manifest, self.cfg, self.out)
        template1 = Path(read(first["summary"])["review_template"])
        self.assertEqual(len(read(template1)), 1)
        second = run(self.manifest, self.cfg, self.out)
        template2 = Path(read(second["summary"])["review_template"])
        self.assertEqual(len(read(template2)), 2)
        self.assertNotEqual(template1, template2)
        self.assertEqual(len(read(template1)), 1)

    def test_truncation_is_saved_but_not_accepted(self):
        class Response:
            status_code = 200
            text = '{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}'
            def json(self):
                return json.loads(self.text)
        with patch("listen_labeling.client.requests.post", return_value=Response()):
            result = run(self.manifest, self.cfg, self.out)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(len(list(self.out.glob("branches/s1/*/*/*/attempt-*.json"))), 2)
        self.assertFalse(list(self.out.glob("branches/s1/*/*/*/result.json")))

    def test_media_change_blocks_old_export(self):
        result = run(self.manifest, self.cfg, self.out)
        sf.write(self.audio, np.zeros(20480), 16000)
        with self.assertRaisesRegex(ValueError, "changed"):
            export(result["summary"], self.root / "bad.json", allow_weak_train=True)

    def test_frame_pts_and_endpoint_credentials(self):
        content, meta = frames(self.video, DURATION, self.cfg["media"])
        times = [x["time_s"] for x in meta["frames"]]
        self.assertAlmostEqual(times[1], .16)  # Actual 25-FPS PTS; not invented .125 s.
        self.assertTrue(any("0.160000" in x.get("text", "") for x in content))
        cfg = dict(self.cfg["endpoints"]["audio"], api_key="secret")
        with self.assertRaisesRegex(ValueError, "api_key_env"):
            validate_endpoint(cfg, "audio")

    def test_discovery_prompt_has_no_behavior_taxonomy(self):
        for branch in ("visual", "audio", "omni"):
            text = prompt(branch, DURATION, "discovery")
            self.assertNotIn("Allowed types", text)
            self.assertNotIn("head_nod", text)
            self.assertNotIn("listener_vocalization", text)
            self.assertIn("interpretation", text)
            self.assertIn("evidence", text)

    def test_audit_reinspects_media_without_discovery_text(self):
        run(self.manifest, self.cfg, self.out)
        for index, request in enumerate(self.requests):
            self.assertEqual(len(request["messages"]), 1)
            content = request["messages"][0]["content"]
            if index >= 2:
                self.assertTrue(content[0]["text"].startswith("Task: primary_audit"))
                self.assertNotIn("自由描述_", json.dumps(content, ensure_ascii=False))
                # Same original frames or waveform in both stages, not textual pseudo-media.
                self.assertEqual(content[1:], self.requests[index - 2]["messages"][0]["content"][1:])

    def test_empty_discovery_does_not_suppress_audit_positives(self):
        for branch in ("visual", "audio"):
            value = answer(branch, "discovery")
            value["events"] = []
            self.responses[(branch, "discovery")] = value
        result = run(self.manifest, self.cfg, self.out)
        label = read(read(result["summary"])["items"][0]["draft"])
        self.assertEqual(len(label["events"]), 2)

    def test_open_observation_never_becomes_an_implicit_training_label(self):
        value = answer("visual", "discovery")
        value["events"][0]["behavior"] = "head_nod 或其他细微动作，暂不能命名"
        self.responses[("visual", "discovery")] = value
        audit = answer("visual")
        audit.update(events=[], reviewed={"head_nod": []}, unknown={"head_nod": [[0, DURATION]]})
        self.responses[("visual", "audit")] = audit
        result = run(self.manifest, self.cfg, self.out)
        item = read(result["summary"])["items"][0]
        label = read(item["draft"])
        self.assertEqual(label["coverage"]["head_nod"], [])
        self.assertFalse(any(e["type"] == "head_nod" for e in label["events"]))
        self.assertEqual(read(item["open_events"]["visual"])["events"][0]["behavior"], value["events"][0]["behavior"])

    def test_discovery_schema_preserves_uncertain_unfamiliar_events(self):
        value = answer("audio", "discovery")
        value["events"][0].update(owner="uncertain", uncertainty="可能有串音", custom_detail={"texture": "无法命名"})
        self.assertEqual(validate_response(value, "audio", DURATION, "discovery"), value)
        self.responses[("audio", "discovery")] = value
        result = run(self.manifest, self.cfg, self.out)
        item = read(result["summary"])["items"][0]
        self.assertIn("open_event_uncertainty", item["review_reasons"])
        self.assertEqual(read(item["open_events"]["audio"]), value)

    def test_discovery_rejects_unsupported_evidence_and_invalid_time(self):
        for field, bad in (("modality", "audio"), ("evidence", ""), ("end", DURATION + .1),
                           ("confidence", float("nan")), ("owner", "uncertain")):
            value = answer("visual", "discovery")
            value["events"][0][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_response(value, "visual", DURATION, "discovery")

    def test_audit_only_accepts_its_training_classes(self):
        value = answer("visual")
        value["events"][0]["type"] = "smile"
        with self.assertRaisesRegex(ValueError, "event type"):
            validate_response(value, "visual", DURATION, "audit")
        with self.assertRaisesRegex(ValueError, "per-modality"):
            validate_response(answer("visual"), "visual", DURATION, "discovery")

    def test_omni_two_stages_and_versioned_provenance(self):
        result = run(self.manifest, self.cfg, self.out, "omni")
        self.assertEqual(len(self.requests), 2)
        for request in self.requests:
            types = {x["type"] for x in request["messages"][0]["content"]}
            self.assertTrue({"image_url", "input_audio"} <= types)
        label = read(read(result["summary"])["items"][0]["draft"])
        provenance = label["provenance"]
        self.assertEqual(provenance["annotator_version"], "listen-labeling-v2-open")
        self.assertEqual(set(provenance["branch_fingerprints"]["omni"]), {"discovery", "audit"})

    def transform(self):
        return dict(crop=dict(x=7, y=3, width=17, height=19), source_size=[32, 32],
                    output_size=[512, 512], resample="bicubic")

    def test_crop_resize_region_then_exact_512(self):
        source = Image.new("RGB", (32, 32), "red")
        source.paste(Image.new("RGB", (17, 19), "blue"), (7, 3))
        output = crop_resize(source, self.transform())
        self.assertEqual(output.size, (512, 512))
        self.assertEqual(output.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(output.getpixel((511, 511)), (0, 0, 255))
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            crop_resize(output, self.transform())

    def test_crop_rejects_missing_outside_and_wrong_resolution(self):
        for key, value in (("crop", None), ("crop", dict(x=20, y=0, width=17, height=19)),
                           ("output_size", [256, 256]), ("resample", "nearest")):
            transform = self.transform()
            transform[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_transform(transform)

    def test_model_payload_is_cropped_512_and_cache_binds_crop(self):
        self.row["video_transform"] = self.transform()
        atomic_json(self.manifest, [self.row])
        result = run(self.manifest, self.cfg, self.out)
        self.assertEqual(result["failed"], 0)
        for request in self.requests:
            for part in request["messages"][0]["content"]:
                if part["type"] == "image_url":
                    data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                    self.assertEqual(Image.open(io.BytesIO(data)).size, (512, 512))
        self.row["video_transform"]["crop"]["x"] += 1
        atomic_json(self.manifest, [self.row])
        run(self.manifest, self.cfg, self.out)
        self.assertEqual([r["model"] for r in self.requests], ["visual", "audio", "visual", "audio", "visual", "visual"])
        label = read(read(result["summary"])["items"][0]["draft"])
        self.assertEqual(label["provenance"]["video_transform"], self.transform())

    def test_visual_only_never_sends_mixed_audio_or_exports_test_package(self):
        self.row.update(audio_role="mixed", audio_isolated=False, experiment_split="unassigned",
                        training_admitted=False, video_transform=self.transform())
        atomic_json(self.manifest, [self.row])
        self.assertFalse(preflight(self.manifest, self.cfg, "visual")["network_called"])
        result = run(self.manifest, self.cfg, self.out, "visual")
        self.assertEqual(result["failed"], 0)
        self.assertEqual([r["model"] for r in self.requests], ["visual", "visual"])
        self.assertNotIn("input_audio", str(self.requests))
        label = read(read(result["summary"])["items"][0]["draft"])
        self.assertEqual(label["coverage"]["listener_vocalization"], [])
        self.assertEqual(label["unknown"]["listener_vocalization"], [[0, DURATION]])
        with self.assertRaisesRegex(ValueError, "No eligible"):
            export(result["summary"], self.root / "bad.json", allow_weak_train=True)
        for mode in ("split", "omni"):
            with self.assertRaises(ValueError):
                run(self.manifest, self.cfg, self.out, mode)
        self.assertEqual(len(self.requests), 2)

    def test_realtalk_relative_paths_null_crop_and_original_frame_coordinates(self):
        metadata = self.root / "meta.json"
        atomic_json(metadata, {})
        raw = dict(sample_id="001", pair_id="video__0001", video_id="video", speaker="p0", listener="p1",
                   start_frame=100, end_frame=132, start_seconds=4., end_seconds=5.28, frame_count=32, fps=25,
                   duration_seconds=DURATION, source_width=32, source_height=32, crop=self.transform()["crop"],
                   original_video="v.mp4", listener_video="DO_NOT_OPEN_THIS_CROPPED_VIDEO.mp4", audio="a.wav",
                   metadata="meta.json", frame_annotations="meta.json", training_admitted=False, flags=["review"])
        package = self.root / "samples.json"
        atomic_json(package, {"samples": [raw, dict(raw, sample_id="002", pair_id="video__0002", crop=None)]})
        summary = prepare_package(package, self.root / "prepared", self.cfg["media"], ["001"])
        self.assertEqual((summary["ready"], summary["missing_crop"], summary["failed"]), (1, 1, 0))
        row = read(summary["manifest"])[0]
        self.assertEqual(row["target_video"], str(self.video))
        self.assertEqual(row["video_transform"], self.transform())
        self.assertFalse(row["audio_isolated"])
        self.assertEqual(row["source_reference"]["start_frame"], 100)
        preview = read(summary["report"])["previews"][0]
        with Image.open(preview["model_frame"]) as image:
            self.assertEqual(image.size, (512, 512))
        self.assertEqual(sha(preview["model_frame"]), preview["model_jpeg_sha256"])
        self.assertEqual(self.requests, [])

    def seamless_item(self):
        atomic_json(self.root / "meta.json", {})
        sf.write(self.root / "speaker.wav", np.zeros(20480), 16000)
        item = dict(ordinal=1, sample_id="seamless_001", pair="pair1", split="train", duration_s=DURATION,
                    source_start_s=4., source_end_s=5.28, segment_in_context_s=[3., 4.28],
                    crop_proposals={"listener": dict(status="proposal_only", x=7.3, y=3.9, size=17.4)},
                    roles={}, metadata_file="meta.json", quality_evidence_file="meta.json")
        for i, role in enumerate(("speaker", "listener")):
            item["roles"][role] = dict(person_id="person"+str(i), source_index=i, clip_video="v.mp4",
                                       audio_wav="speaker.wav" if role == "speaker" else "a.wav", context_video="v.mp4",
                                       media_checks=dict(clip=dict(width=32, height=32, sample_rate=16000,
                                                                  pcm_source_start_sample=64000, pcm_source_end_sample=84480)))
        return item

    def test_seamless_roles_float_crop_and_local_timeline(self):
        item = self.seamless_item()
        path = self.root / "items.json"
        atomic_json(path, [item])
        result = prepare_seamless(path, self.root / "seamless", self.cfg["media"], ["001"])
        self.assertEqual((result["ready"], result["failed"]), (1, 0))
        row = read(result["manifest"])[0]
        self.assertEqual(row["target_audio"], str(self.audio))
        self.assertEqual(row["video_transform"]["crop"], dict(x=7, y=4, width=17, height=17))
        self.assertEqual(row["duration_s"], DURATION)
        self.assertEqual(row["source_reference"]["source_start_s"], 4.)
        self.assertFalse(row["source_reference"]["context_used_for_annotation"])
        self.assertFalse(row["training_admitted"])
        run(result["manifest"], self.cfg, self.out)
        self.assertEqual(len(self.requests), 4)
        for request in self.requests:
            for part in request["messages"][0]["content"]:
                if part["type"] == "input_audio":
                    audio, rate = sf.read(io.BytesIO(base64.b64decode(part["input_audio"]["data"])))
                    target, _ = sf.read(self.audio)
                    np.testing.assert_array_equal(audio, target)
                    self.assertEqual(rate, 16000)

    def test_seamless_invalid_audio_support_and_missing_crop(self):
        item = self.seamless_item()
        item["roles"]["listener"]["media_checks"]["clip"]["pcm_source_end_sample"] += 1
        path = self.root / "items.json"
        missing = copy.deepcopy(item)
        missing.update(sample_id="missing_crop", ordinal=2, crop_proposals={"listener": {"status": "unmeasured"}})
        atomic_json(path, [item, missing])
        result = prepare_seamless(path, self.root / "seamless_bad", self.cfg["media"])
        self.assertEqual((result["ready"], result["missing_crop"], result["failed"]), (0, 1, 1))
        self.assertEqual(self.requests, [])

    def test_float_crop_is_explicitly_rounded_and_never_clamped(self):
        proposal = dict(status="proposal_only", x=1.5, y=2.5, size=20.5)
        self.assertEqual(integer_crop(proposal, 32, 32)["crop"], dict(x=2, y=3, width=21, height=21))
        for value in (float("nan"), -1, 40):
            with self.assertRaises(ValueError):
                integer_crop(dict(proposal, x=value), 32, 32)


if __name__ == "__main__":
    unittest.main()
