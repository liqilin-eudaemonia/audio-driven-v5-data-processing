"""Drop listener crops that show burned-in subtitles, logos, or other graphics.

The head-shoulder crop stays as it is. Qwen3-VL sees frames sampled at 2 fps
across the whole clip, plus a frame at the end. One yes is enough to drop it.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path("/private/liqi_workspace/project_code/audio-driven-v5/experiments/speakervid_listener_face224_20261103")
MODEL = "/private/ckpts/Qwen3-VL-8B-Instruct"
PROMPT = """这些图是同一段 listener 裁剪视频按时间顺序抽出的画面。只判断画面里有没有后期叠加上去的东西：
字幕或其他屏幕文字、台标、logo、水印、贴图、色块边框。
衣服上的图案、话筒、房间里的真实物体，不算。
只要任意一帧出现上述叠加，就整段去掉。
只输出 JSON，不要其他文字：{"drop": true或false, "reasons": ["subtitle", "logo", "watermark", "border", "other_graphic"], "note": "一句说明"}
没有这些问题时 reasons 用空数组。"""


def sample_times(duration, fps):
    step = 1.0 / fps
    times, t = [], 0.0
    while t < duration - 0.05:
        times.append(round(t, 3))
        t += step
    tail = round(max(0.0, duration - 0.05), 3)
    if not times or tail - times[-1] > 0.2:
        times.append(tail)
    return times


def frames_at(path, times):
    cap = cv2.VideoCapture(str(path))
    images = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))
    cap.release()
    if len(images) < max(3, len(times) // 2):
        raise RuntimeError(f"only decoded {len(images)} of {len(times)} frames from {path}")
    return images


def parse_answer(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON in model output: " + text[:400])
    data = json.loads(match.group(0))
    if type(data.get("drop")) is not bool:
        raise ValueError("drop is not a boolean: " + text[:400])
    reasons = data.get("reasons") or []
    if not isinstance(reasons, list):
        raise ValueError("reasons is not a list")
    return {"drop": data["drop"], "reasons": [str(x) for x in reasons], "note": str(data.get("note") or "")}


def trial_rows(rows, count):
    chosen = [rows[0], rows[2]]
    seen = {rows[0]["video_id"]}
    for row in rows:
        if row["video_id"] in seen:
            continue
        seen.add(row["video_id"])
        chosen.append(row)
        if len(chosen) >= count:
            break
    return chosen[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", type=Path, default=ROOT / "selected_400.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "overlay_trial" / "results.jsonl")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--trial", type=int, default=0, help="If set, judge this many clips, always including the first and third.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.list.open()]
    if args.trial:
        rows = trial_rows(rows, args.trial)
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": args.device},
        attn_implementation="sdpa", trust_remote_code=True, local_files_only=True,
    ).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if args.out.exists():
        done = {json.loads(line)["id"]: json.loads(line) for line in args.out.open() if line.strip()}
    with args.out.open("a") as handle:
        for index, row in enumerate(rows, 1):
            if row["id"] in done and done[row["id"]].get("status") == "ok":
                continue
            times = sample_times(row["file_dur"], args.fps)
            images = frames_at(row["listener_mp4"], times)
            content = []
            for t, image in zip(times, images):
                content.append({"type": "text", "text": f"{t:.2f}s"})
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": PROMPT})
            messages = [{"role": "user", "content": content}]
            template = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            inputs = processor(text=template, images=images, return_tensors="pt")
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=180)
            prompt_len = int(inputs["input_ids"].shape[1])
            text = processor.tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
            try:
                parsed = parse_answer(text)
                record = dict(id=row["id"], video_id=row["video_id"], status="ok", frames=len(images),
                              frame_times_s=times[:len(images)], **parsed, raw=text)
            except Exception as exc:
                record = dict(id=row["id"], video_id=row["video_id"], status="parse_error",
                              frames=len(images), error=str(exc), raw=text)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)} {record['id']} {record.get('drop', record['status'])} {record.get('reasons', '')}", flush=True)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
