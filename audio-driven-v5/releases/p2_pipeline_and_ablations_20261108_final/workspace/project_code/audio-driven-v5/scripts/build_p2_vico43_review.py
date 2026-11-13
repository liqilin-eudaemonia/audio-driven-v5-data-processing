"""Publish all43 existing ViCo outputs using the existing ViCo review layout.

CPU only; source videos/checkpoints/score receipts are never modified.
"""
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"experiments/p2_vico256_combined8000_20261106"
MOTION = ROOT/"experiments/p2_vico_motion8000_20261106"
OUT = ROOT/"data_previews/listen_benchmark_review/p2_vico43_combined8000_20261106"
ARM = "base_event_feature_whisper_8000"
URL = "http://localhost:8876/listen_benchmark_review/p2_vico43_combined8000_20261106/"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):h.update(block)
    return h.hexdigest()


def save(p, data):
    temp = p.with_suffix(p.suffix+".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temp.replace(p)


def link(source, dest):
    source = source.resolve();dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():assert dest.resolve() == source, dest
    elif dest.exists():raise FileExistsError(dest)
    else:dest.symlink_to(source)
    return str(dest.relative_to(OUT))


def transcode(source, dest):
    checksum = sha(source);receipt = dest.with_suffix(".json")
    if dest.exists():
        r = json.loads(receipt.read_text())
        assert r["input_sha256"] == checksum and r["output_sha256"] == sha(dest)
        return r
    temp = dest.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(source),
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-threads", "1", "-crf", "18",
        "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)], check=True)
    temp.replace(dest)
    r = dict(input=str(source), input_sha256=checksum, output_sha256=sha(dest),
             purpose="browser-only H264 preview; original MPEG4 score input unchanged; no resize or temporal resampling",
             encoding="libx264 crf18 yuv420p; preview does not participate in scoring")
    save(receipt, r);return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads((MOTION/"RESULTS.json").read_text())
    assert json.loads((MOTION/"FINAL_AUDIT.json").read_text())["passed"]
    assert json.loads((BASE/"FINAL_AUDIT.json").read_text())["passed"]
    motion_p = json.loads((MOTION/"protocol.json").read_text())
    sources = {s["id"]: s for s in motion_p["samples"]}
    metrics = {s["id"]: s for s in results["per_clip"]}
    pre = json.loads((BASE/"INPUT_PREFLIGHT.json").read_text())["samples_audit"]
    old = ROOT/"data_previews/vico_paper_test"
    metadata = {s["uuid"]: s for s in json.loads((old/"clips.json").read_text())}
    assert len(pre) == len(metadata) == 43

    def one(pair):
        number, item = pair;uid = item["id"];src = sources[uid]["streams"]
        dest = OUT/"media"/uid;dest.mkdir(parents=True, exist_ok=True)
        previews = {}
        for role in ("speaker", "gt"):
            previews[role] = transcode(Path(src[role]["path"]), dest/(role+".mp4"))
        native = BASE/"generated"/ARM/"vico"/uid
        generation_meta = json.loads((native/"metadata.json").read_text())
        assert generation_meta["step"] == 8000 and generation_meta["arm"] == ARM
        assert generation_meta["target_description_access"] is False
        assert generation_meta["listener_future_input"] is False
        files = dict(native_video=link(native/"generated.mp4", dest/"generated_512.mp4"),
            scoring_video=link(Path(src["prediction"]["path"]), dest/"generated_256.mp4"),
            generated_wav=link(native/"listener.wav", dest/"generated.wav"),
            first_frame=link(BASE/"conditions/vico"/uid/"first_frame.png", dest/"first_frame.png"),
            raw_gt=link(Path(src["gt"]["path"]), dest/"gt_original_score_input.mp4"))
        source_audio = Path(item["speaker_audio_path"])
        assert sha(source_audio) == item["speaker_audio_sha256"]
        wave, sr = sf.read(source_audio, dtype="float32", always_2d=True);wave = wave.mean(1)
        g = math.gcd(sr, 16000);wave = resample_poly(wave, 16000//g, sr//g).astype(np.float32)
        duration = item["original_gt"]["duration_s"];count = round(duration*16000)
        wave = np.pad(wave, (0, max(0, count-len(wave))))[:count]
        assert hashlib.sha256(wave.astype("<f4").tobytes()).hexdigest() == item["waveform_sha256"]
        if (dest/"speaker.wav").exists():
            prior, prior_rate = sf.read(dest/"speaker.wav", dtype="float32")
            assert prior_rate == 16000 and np.array_equal(prior, wave)
        else:sf.write(dest/"speaker.wav", wave, 16000, subtype="FLOAT")
        plans = json.loads((BASE/"conditions/vico"/uid/"qwen/plan.json").read_text())
        quality = {r: json.loads((MOTION/"features"/r/(uid+".json")).read_text()) for r in ("speaker", "gt", "prediction")}
        m = metadata[uid]
        record = dict(id=uid, index=number, duration=duration, frames=item["original_gt"]["frames"],
            attitude=m["attitude"], speaker_id=m["speaker_id"], listener_id=m["listener_id"],
            speaker_video=str((dest/"speaker.mp4").relative_to(OUT)), gt_video=str((dest/"gt.mp4").relative_to(OUT)),
            speaker_wav=str((dest/"speaker.wav").relative_to(OUT)), **files, plan=plans,
            metrics=metrics[uid]["metrics_raw"], timing=metrics[uid]["timing_audit"],
            original_dataset_page="/vico_paper_test/",
            original_listener="/vico_paper_test/"+m["listener"],
            original_speaker="/vico_paper_test/"+m["speaker"],
            source_sha256={r: info["sha256"] for r, info in src.items()},
            preview_transcodes=previews,
            audio=dict(speaker_wav_sha256=sha(dest/"speaker.wav"), input_waveform_sha256=item["waveform_sha256"],
                       generated_wav_sha256=sha(native/"listener.wav"),
                       generated_audio_info=sf.info(native/"listener.wav").__dict__,
                       policy="speaker WAV is exact inference16k waveform; generated original48k stereo float WAV; mixed uses0.5 gain per source; no listener GT audio supplied"),
            quality={r: dict(missing_faces=q["missing_face_frames"], multi_face_frames=q["multi_face_frames"]) for r, q in quality.items()})
        # soundfile's extra_info can be a plain string; retain only scalar metadata.
        record["audio"]["generated_audio_info"] = {k: v for k, v in record["audio"]["generated_audio_info"].items() if isinstance(v, (str, int, float, bool))}
        print("prepared", number, uid, flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        cases = list(pool.map(one, enumerate(pre, 1)))
    document = dict(schema="vico43-combined8000-review-v1", cases=cases, count=43,
        model="AV + Event + Feature + speaker Whisper · step8000", sampling="all43 official paper test clips in original order; no selection",
        summary=results["metrics_table_scale"], coverage=results["coverage_clips"],
        image_summary=json.loads((BASE/"RESULTS.json").read_text())["metrics"],
        table_scale=results["table_scale"], strict_customlistener_table_reproduction=False,
        generation=dict(size=512, fps=25), scoring=dict(size=256, fps=30),
        source_results_sha256=sha(MOTION/"RESULTS.json"), source_generation_sha256=sha(BASE/"RESULTS.json"),
        reused_layout="/vico_paper_test/; synchronized playback/audio pattern from p2_seamless200_review.html")
    save(OUT/"data.json", document)
    for name, source in {"RESULTS.md":MOTION/"RESULTS.md", "metrics.json":MOTION/"RESULTS.json", "summary.csv":MOTION/"summary.csv",
                         "protocol.json":MOTION/"protocol.json", "SCORING_PROTOCOL.json":MOTION/"SCORING_PROTOCOL.json", "FINAL_AUDIT.json":MOTION/"FINAL_AUDIT.json"}.items():
        shutil.copyfile(source, OUT/name)
    base_html = (old/"index.html").read_text()
    base_styles = re.search(r"<style>(.*?)</style>", base_html, re.S).group(1)
    page = (ROOT/"scripts/p2_vico43_review.html").read_text().replace("{{BASE_STYLES}}", base_styles)
    (OUT/"index.html").write_text(page)
    nav = '<p id="vico-generated-nav" style="margin-top:10px"><a href="'+URL.replace("http://localhost:8876", "")+'">查看联合8000步全部43条生成结果：speaker／GT／生成视频同步对照 →</a></p>'
    if 'id="vico-generated-nav"' not in base_html:
        (old/"index.html").write_text(base_html.replace("</header>", nav+"\n</header>", 1))
    save(MOTION/"VISUALIZATION_PUBLISHED.json", dict(url=URL, cases=43, preview=str(OUT),
        generated_media="symlinks to original512/25 and scored256/30; no regeneration or transcoding",
        gt_and_speaker="86 browser-compatible H264 previews; original score files unchanged",
        data_sha256=sha(OUT/"data.json"), template_sha256=sha(ROOT/"scripts/p2_vico43_review.html")))
    print(URL)


if __name__ == "__main__":main()
