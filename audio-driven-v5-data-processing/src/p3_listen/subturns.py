"""Continuous <=10 s subturns and per-source loudness normalization."""
import copy
import concurrent.futures
import json
import math
from pathlib import Path
import re
import subprocess

from .common import ROOT, digest, read, sha, write
from .media import command, uri

OUT = ROOT / "data_previews/seamless_subturns_v5"
POLICY = dict(target_lufs=-18., max_gain_db=45., min_gain_db=-18., peak_db=-1., silence_threshold_lufs=-70.,
              scope="whole original participant track; same gain for every role and subturn",
              filter="constant gain, oversampled peak limiter with latency compensation")


def cut_points(start, end, words, listener_events, max_seconds=10.):
    """Preserve all time, favor pauses; label hard cuts explicitly."""
    if not math.isfinite(start + end) or end <= start or max_seconds <= 0:
        raise ValueError("Invalid interval")
    relevant = sorted([w for w in words if w["end"] > start and w["start"] < end], key=lambda w: w["start"])
    gaps = []
    for a, b in zip(relevant, relevant[1:]):
        if b["start"] >= a["end"]:
            gaps.append(((a["end"] + b["start"]) / 2, b["start"] - a["end"]))
    result, left = [], start
    while end - left > max_seconds + 1e-8:
        remaining = math.ceil((end - left) / max_seconds - 1e-10)
        ideal = left + (end - left) / remaining
        # Ensure remaining subturns fit, without making tiny leading segments.
        lo = max(left + min(4., (end - left) / remaining), end - (remaining - 1) * max_seconds)
        hi = left + max_seconds
        choices = [(p, gap) for p, gap in gaps if lo - 1e-8 <= p <= hi + 1e-8]
        safe = [(p, gap) for p, gap in choices if not any(e["start"] < p < e["end"] for e in listener_events)]
        if safe or choices:
            chosen = safe or choices
            point, gap = min(chosen, key=lambda x: abs(x[0] - ideal) - min(x[1], .8) * 2)
            reason = "speaker_pause" if gap >= .15 else "speaker_word_gap"
        else:
            point, reason = hi, "hard_10s_no_word_gap"
        result.append(dict(start=round(left, 9), end=round(point, 9), end_reason=reason,
                           listener_event_cut=any(e["start"] < point < e["end"] for e in listener_events)))
        left = point
    result.append(dict(start=round(left, 9), end=end, end_reason="parent_speaker_end", listener_event_cut=False))
    if any(r["end"] - r["start"] > max_seconds + 1e-7 or r["end"] <= r["start"] for r in result):
        raise ValueError("Subturn length contract violated")
    return result


def loudness(source):
    proc = subprocess.run(["ffmpeg", "-v", "info", "-nostdin", "-threads", "1", "-i", str(source),
        "-af", "loudnorm=I=-18:TP=-1:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-2000:])
    match = re.search(r'\{\s*"input_i".*?\}', proc.stderr, re.S)
    if not match:
        raise ValueError("Missing EBU R128 loudness measurement")
    return {k: float(v) if math.isfinite(float(v)) else None for k, v in json.loads(match.group()).items()
            if k in ("input_i", "input_tp", "input_lra", "input_thresh")}


def normalize(source_id, source, speech_seconds):
    folder = OUT / "normalized_audio"
    folder.mkdir(parents=True, exist_ok=True)
    path, marker = folder / (source_id + ".wav"), folder / (source_id + ".json")
    expected = dict(source=source, policy=POLICY, annotated_speech_seconds=speech_seconds)
    if marker.exists():
        value = read(marker)
        if value["contract"] == expected and sha(path) == value["sha256"]:
            return value
    original = uri(source, ".wav")
    before = loudness(original)
    integrated = before["input_i"]
    voiced = speech_seconds >= 2 and integrated is not None and integrated >= POLICY["silence_threshold_lufs"]
    gain = (0. if not voiced else
            max(POLICY["min_gain_db"], min(POLICY["max_gain_db"], POLICY["target_lufs"] - integrated)))
    peak = 10 ** (POLICY["peak_db"] / 20)
    # FFmpeg 4.2 lacks alimiter's latency option. Its 5ms lookahead delays by
    # attack_samples-1 (verified with an impulse); flush then remove exactly that delay.
    pending = path.with_suffix(".pending.wav")
    history=[]
    for attempt in range(4):
        filters = (f"volume={gain:.6f}dB,aresample=192000,apad=pad_len=959,"
                   f"alimiter=limit={peak}:level=false:attack=5:release=50,"
                   "atrim=start_sample=959,asetpts=PTS-STARTPTS,aresample=48000")
        command(["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", "1", "-i", original,
                 "-af", filters, "-ac", "1", "-ar", "48000", "-c:a", "pcm_s24le", str(pending)])
        after = loudness(pending)
        history.append(dict(gain_db=gain,measurement=after))
        if not voiced or after["input_i"] is None or abs(after["input_i"]-POLICY["target_lufs"])<=.3 or attempt==3:
            break
        adjusted=max(POLICY["min_gain_db"],min(POLICY["max_gain_db"],gain+POLICY["target_lufs"]-after["input_i"]))
        if abs(adjusted-gain)<.01: break
        gain=adjusted
    import soundfile as sf
    if sf.info(str(pending)).frames <= 0:
        raise ValueError("Empty normalized audio")
    pending.replace(path)
    value = dict(contract=expected, path=str(path), sha256=sha(path), gain_db=gain,
                 before=before, after=after, sample_rate=48000, measurement_history=history,
                 target_reached=after["input_i"] is not None and abs(after["input_i"] - POLICY["target_lufs"]) <= 2)
    write(marker, value)
    print(dict(normalized=source_id, before=integrated, gain_db=gain, after=after["input_i"]), flush=True)
    return value


def prepare(workers=4, normalize_audio=True):
    original = read(ROOT / "data_previews/seamless_speaker_span_v4/manifest.json")
    selection = read(ROOT / "data_previews/seamless_face_listening_v1/selection.json")
    records = {r["id"]: r for p in selection["attempts"] for r in p["records"]}
    sources = {s["id"]: s["source"] for p in original["pairs"] for s in p["sources"]}
    normalized = {}
    if normalize_audio:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = {pool.submit(normalize, key, value, sum(max(0,s.get("end",0)-s.get("start",0))
                     for s in records[key]["transcript"] if s.get("transcript","").strip())): key for key, value in sources.items()}
            for future in concurrent.futures.as_completed(tasks):
                normalized[tasks[future]] = future.result()
    output = copy.deepcopy(original)
    audit = []
    for pair in output["pairs"]:
        clips = []
        for parent in pair["clips"]:
            speaker = records[parent["speaker"]]
            words = [dict(start=w["start"], end=w["end"], text=w.get("word", "")) for s in speaker["transcript"]
                     for w in s.get("words", []) if isinstance(w.get("start"), (int, float)) and isinstance(w.get("end"), (int, float))]
            activity = parent["listener_events"] + [dict(start=parent["start"] + a, end=parent["start"] + b)
                                                      for a, b in parent["listener_vad"]]
            cuts = cut_points(parent["start"], parent["end"], words, activity)
            for i, cut in enumerate(cuts):
                c = copy.deepcopy(parent)
                a, b = cut["start"], cut["end"]
                c.update(start=a, end=b, duration=round(b-a, 9), segmentation_revision=5,
                         parent_clip_path=parent["path"], parent_clip_start=parent["start"], parent_clip_end=parent["end"],
                         subturn_index=i + 1, subturn_count=len(cuts), subturn_boundary_reason=cut["end_reason"],
                         subturn_listener_event_cut=cut["listener_event_cut"],
                         path=parent["path"] + f"_sub{i+1:02d}",
                         start_reason=parent["start_reason"] if i == 0 else cuts[i-1]["end_reason"],
                         end_reason=parent["end_reason"] if i == len(cuts)-1 else cut["end_reason"])
                for role in ("speaker", "listener"):
                    c[role + "_vad"] = [[max(a, parent["start"]+x)-a, min(b, parent["start"]+y)-a]
                                        for x, y in parent[role + "_vad"] if parent["start"]+y>a and parent["start"]+x<b]
                    c[role + "_vad_fraction"] = sum(y-x for x,y in c[role + "_vad"])/(b-a)
                    person = records[parent[role]]
                    c[role + "_text"] = " ".join(w.get("word", "") for s in person["transcript"] for w in s.get("words", [])
                        if isinstance(w.get("start"),(float,int)) and isinstance(w.get("end"),(float,int)) and w["end"]>a and w["start"]<b)
                c["listener_events"] = [e for e in parent["listener_events"] if e["end"]>a and e["start"]<b]
                c["kind"] = ("feedback_candidate" if any(e["label"] == "feedback_candidate" for e in c["listener_events"])
                             else "vocal_candidate" if c["listener_events"] or c["listener_vad"] else "quiet_candidate")
                for e in c["parent_events"]:
                    e["in_export"] = e["end"]>a and e["start"]<b
                indices = [j for j,t in enumerate(parent["emotion_sample_times"]) if a<=parent["start"]+t<b]
                c["emotion_sample_times"] = [parent["start"]+parent["emotion_sample_times"][j]-a for j in indices]
                c["emotion"] = {role: {k:[values[j] for j in indices] for k,values in tracks.items()}
                                for role,tracks in parent["emotion"].items()}
                c["normalized_audio"] = {role:normalized[parent[role]] for role in ("speaker","listener")} if normalized else {}
                clips.append(c)
            audit.append(dict(parent=parent["path"], original_seconds=parent["duration"],
                              children=len(cuts), retained_seconds=sum(x["end"]-x["start"] for x in cuts), cuts=cuts))
        pair["clips"] = clips
        pair["segmentation_revision"] = 5
    allclips = [c for p in output["pairs"] for c in p["clips"]]
    output.update(segmentation_revision=5, parent_segmentation_revision=4,
                  subturn_policy="continuous non-overlapping <=10s; prefer speaker pauses/word gaps and avoid listener activity",
                  normalization_policy=POLICY, normalization_complete=bool(normalized),
                  counts=dict(pairs=len(output["pairs"]), clips=len(allclips), seconds=sum(c["duration"] for c in allclips)))
    summary = dict(output["counts"], max_seconds=max(c["duration"] for c in allclips),
                   split_parents=sum(a["children"]>1 for a in audit),
                   hard_cuts=sum(c["end_reason"]=="hard_10s_no_word_gap" for a in audit for c in a["cuts"]),
                   listener_activity_cuts=sum(c["listener_event_cut"] for a in audit for c in a["cuts"]),
                   normalized_sources=len(normalized),
                   loudness_outliers=[k for k,v in normalized.items() if not v["target_reached"]])
    for a in audit:
        if abs(a["original_seconds"]-a["retained_seconds"])>1e-6:
            raise ValueError("Subturns lost source time")
    write(OUT / ("manifest.json" if normalize_audio else "planned_manifest.json"), output)
    write(OUT / "subturn_audit.json", audit)
    write(OUT / "summary.json", summary)
    print(summary, flush=True)


def render_clip(pair, c):
    folder = OUT / c["path"]
    marker = folder / "complete.json"
    contract = digest(dict(start=c["start"], end=c["end"], crops=c["crop_boxes"], audio=c["normalized_audio"]))
    if marker.exists() and read(marker)["contract"] == contract:
        return
    folder.mkdir(parents=True, exist_ok=True)
    duration = c["duration"]
    # Floor frames ensures exported media never exceeds 10 seconds.
    frames = math.floor(duration * 25 + 1e-7)
    for role in ("speaker", "listener"):
        source = pair["sources"][c[role + "_index"]]["source"]
        box = c["crop_boxes"][role]
        wave = c["normalized_audio"][role]["path"]
        command(["ffmpeg","-v","error","-nostdin","-y","-threads","1","-ss",str(c["start"]),"-i",wave,
                 "-t",str(duration),"-c:a","aac","-b:a","160k",str(folder / (role + ".m4a"))])
        command(["ffmpeg","-v","error","-nostdin","-y","-threads","1","-ss",str(c["start"]),"-i",uri(source,".mp4"),
                 "-i",str(folder/(role+".m4a")),"-map","0:v","-map","1:a",
                 "-vf",f"setpts=PTS-STARTPTS,crop={box['size']}:{box['size']}:{box['x']}:{box['y']},scale=512:512,fps=25,setsar=1",
                 "-frames:v",str(frames),"-t",str(duration),"-c:v","libx264","-threads","1","-preset","veryfast","-crf","20",
                 "-pix_fmt","yuv420p","-c:a","copy","-movflags","+faststart",str(folder/(role+"_512.mp4"))])
    command(["ffmpeg","-v","error","-nostdin","-y","-threads","1","-i",str(folder/"speaker_512.mp4"),
             "-threads","1","-i",str(folder/"listener_512.mp4"),"-filter_complex_threads","1","-filter_complex",
             "[0:v][1:v]hstack[v];[0:a][1:a]amix=inputs=2:normalize=0,aresample=48000,apad=pad_len=239,alimiter=limit=0.891250938:level=false:attack=5:release=50,atrim=start_sample=239,asetpts=PTS-STARTPTS[a]",
             "-map","[v]","-map","[a]","-frames:v",str(frames),"-t",str(duration),"-c:v","libx264","-threads","1",
             "-preset","veryfast","-crf","21","-c:a","aac","-b:a","160k","-movflags","+faststart",str(folder/"faces.mp4")])
    command(["ffmpeg","-v","error","-nostdin","-y","-i",str(folder/"faces.mp4"),"-frames:v","1",str(folder/"poster.jpg")])
    write(folder/"metadata.json", c)
    write(marker, dict(contract=contract, frames=frames))


def render(workers=6):
    manifest = read(OUT/"manifest.json")
    jobs = [(p,c) for p in manifest["pairs"] for c in p["clips"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(render_clip,p,c) for p,c in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(futures)):
            f.result()
            if i%20==0 or i==len(jobs)-1:
                print(dict(rendered=i+1,total=len(jobs)),flush=True)
    for pair in manifest["pairs"]:
        target=OUT/"media"/pair["pair"]/"conversation.mp4"
        if not target.exists():
            target.symlink_to(ROOT/"data_previews/seamless_speaker_span_v4"/pair["conversation_path"])
        write(target.parent/"pair.json",pair)


def render_contexts(workers=4):
    manifest=read(OUT/"manifest.json")
    def one(pair,c):
        folder=OUT/c["path"]
        folder.mkdir(parents=True,exist_ok=True)
        target=folder/"context.mp4"
        if target.exists(): return
        original=ROOT/"data_previews/seamless_speaker_span_v4"/pair["conversation_path"]
        command(["ffmpeg","-v","error","-nostdin","-y","-threads","1","-ss",str(c["start"]),"-i",str(original),
                 "-t",str(c["duration"]),"-vf","fps=15","-an","-c:v","libx264","-threads","1","-preset","veryfast",
                 "-crf","23","-movflags","+faststart",str(target)])
    jobs=[(p,c) for p in manifest["pairs"] for c in p["clips"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,p,c) for p,c in jobs]
        for i,f in enumerate(concurrent.futures.as_completed(futures)):
            f.result()
            if i%100==0:print(dict(contexts=i+1,total=len(jobs)),flush=True)
