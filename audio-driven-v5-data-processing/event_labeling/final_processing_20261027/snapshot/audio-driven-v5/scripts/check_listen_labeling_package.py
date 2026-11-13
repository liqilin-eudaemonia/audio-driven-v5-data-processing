#!/usr/bin/env python3
"""Transport-only smoke on real media. Mock responses are all unknown, NOT annotations."""
import argparse
import base64
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    import numpy as np
    from PIL import Image
    import soundfile as sf
    from p2_listen.contracts import atomic_json, read, sha
    from listen_labeling.pipeline import load_manifest, run
    from listen_labeling.schema import CLASSES
    from listen_labeling.media import frames

    root = Path(args.output).resolve()
    samples = load_manifest(args.manifest)[:args.limit]
    config = read(args.config)
    expected = []
    # Independent audit of raw media selection and transport dimensions/content.
    for row in samples:
        parts, _ = frames(row["target_video"], row["duration_s"], config["media"], row.get("video_transform"))
        expected_images = [hashlib.sha256(base64.b64decode(p["image_url"]["url"].split(",", 1)[1])).hexdigest()
                           for p in parts if p["type"] == "image_url"]
        target, rate = sf.read(row["target_audio"], dtype="float32")
        expected.append((row, expected_images, target, rate))
    trace = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass
        def do_POST(self):
            record = {}
            try:
                index = len(trace)
                row, image_hashes, target, rate = expected[index // 4]
                branch = "visual" if index % 2 == 0 else "audio"
                stage = "discovery" if index % 4 < 2 else "audit"
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                content = payload["messages"][0]["content"]
                text = content[0]["text"]
                assert text.startswith("Task: open_discovery" if stage == "discovery" else "Task: primary_audit")
                assert len(payload["messages"]) == 1 and payload["model"] == "transport-mock-" + branch
                images = [p["image_url"]["url"] for p in content if p["type"] == "image_url"]
                audios = [p["input_audio"]["data"] for p in content if p["type"] == "input_audio"]
                record = dict(sample_id=row["sample_id"], branch=branch, stage=stage)
                if branch == "visual":
                    assert not audios
                    jpegs = [base64.b64decode(x.split(",", 1)[1]) for x in images]
                    assert [hashlib.sha256(x).hexdigest() for x in jpegs] == image_hashes
                    for data in jpegs:
                        with Image.open(io.BytesIO(data)) as image:
                            assert image.size == (512, 512)
                    record["verified_512_frames"] = len(jpegs)
                else:
                    assert not images and len(audios) == 1
                    wave, sent_rate = sf.read(io.BytesIO(base64.b64decode(audios[0])), dtype="float32")
                    assert sent_rate == rate and wave.shape == target.shape
                    error = float(np.max(np.abs(wave - target), initial=0))
                    assert error <= 1 / 32768 + 1e-7, error
                    record.update(verified_listener_wav=True, samplerate=rate, samples=len(wave), pcm16_max_error=error)
                keys = [branch] if stage == "discovery" else CLASSES[branch]
                response = dict(reviewed={k: [] for k in keys}, unknown={k: [[0, row["duration_s"]]] for k in keys},
                                events=[], notes="MOCK_TRANSPORT_ONLY: no observation or annotation was performed")
                body = dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(response)))])
                record["passed"] = True
                self.send_response(200)
            except Exception as exc:
                record.update(passed=False, error=type(exc).__name__ + ": " + str(exc))
                body = dict(error="Mock transport validation failed")
                self.send_response(400)
            trace.append(record)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = copy.deepcopy(config)
        for branch in ("visual", "audio"):
            cfg["endpoints"][branch].update(base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="transport-mock-" + branch, revision="MOCK_NOT_A_MODEL", api_key_env="", attempts=1)
        for row in samples:
            row["training_admitted"] = False
        atomic_json(root / "test_manifest.json", samples, immutable=True)
        result = run(root / "test_manifest.json", cfg, root / "mock_outputs")
        all_unknown = True
        for item in read(result["summary"])["items"]:
            if item["status"] != "draft":
                all_unknown = False
                continue
            label = read(item["draft"])
            all_unknown &= not label["events"] and not any(label["coverage"].values())
        passed = (result["failed"] == 0 and len(trace) == 4 * len(samples) and
                  all(r["passed"] for r in trace) and all_unknown)
        report = dict(passed=passed, real_model_called=False, real_media=True, model_quality_evaluated=False,
                      sample_count=len(samples), http_requests=len(trace), all_mock_labels_unknown=all_unknown,
                      source_manifest=str(Path(args.manifest).resolve()), source_manifest_sha256=sha(args.manifest),
                      trace=trace, run=result, training_started=False)
        atomic_json(root / "transport_report.json", report)
        print(json.dumps(dict(passed=passed, samples=len(samples), requests=len(trace),
                              real_model_called=False, report=str(root / "transport_report.json")), indent=2))
        return 0 if passed else 1
    finally:
        server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    sys.exit(main())
