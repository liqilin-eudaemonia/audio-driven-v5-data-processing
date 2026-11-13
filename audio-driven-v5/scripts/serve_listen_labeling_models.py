#!/usr/bin/env python3
"""Local smoke-test HTTP adapter for shared Transformers models; bind loopback only."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen3-omni", "minicpm", "joy"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-generation-seconds", type=int, default=180)
    args = parser.parse_args()
    import torch
    from PIL import Image
    import soundfile as sf
    torch.set_num_threads(4)
    if args.backend == "qwen3-omni":
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        from qwen_omni_utils import process_mm_info
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model_path, dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
        model.disable_talker()
        processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    elif args.backend == "minicpm":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", init_vision=True, init_audio=True, init_tts=False)
        model = model.eval().cuda()
        processor = None
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()
    torch.cuda.empty_cache()
    lock = threading.Lock()
    print(json.dumps(dict(status="ready", backend=args.backend, model=args.model_path, port=args.port,
                          memory_allocated=torch.cuda.memory_allocated())), flush=True)

    def infer(payload):
        if payload.get("model") != args.served_model_name or payload.get("temperature") != 0:
            raise ValueError("Unexpected model or nonzero temperature")
        messages = payload["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("Smoke adapter expects a single independent user request")
        max_tokens = payload["max_tokens"]
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens out of range")
        began = time.monotonic()
        print(json.dumps(dict(event="request_started", backend=args.backend, max_tokens=max_tokens)), flush=True)
        with tempfile.TemporaryDirectory(prefix="listen_labeling_") as tmp:
            parts, native, n_images, n_audio = [], [], 0, 0
            audio_rates = []
            for part in messages[0]["content"]:
                if part["type"] == "text":
                    parts.append(part)
                    native.append(part["text"])
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/jpeg;base64,"):
                        raise ValueError("Only inline JPEG input supported")
                    data = base64.b64decode(url.split(",", 1)[1], validate=True)
                    with Image.open(io.BytesIO(data)) as image:
                        if image.size != (512, 512):
                            raise ValueError("Expected exactly 512x512 annotation input")
                        picture = image.convert("RGB").copy()
                    parts.append(dict(type="image", image=picture))
                    native.append(picture)
                    n_images += 1
                elif part["type"] == "input_audio":
                    if args.backend == "joy":
                        raise ValueError("Joy visual model must not receive audio")
                    if part["input_audio"]["format"] != "wav":
                        raise ValueError("Expected WAV input")
                    data = base64.b64decode(part["input_audio"]["data"], validate=True)
                    path = Path(tmp) / f"audio_{n_audio}.wav"
                    path.write_bytes(data)
                    audio, rate = sf.read(path, dtype="float32")
                    if audio.ndim != 1:
                        raise ValueError("Expected independent mono audio")
                    parts.append(dict(type="audio", audio=str(path)))
                    audio_rates.append(rate)
                    if args.backend == "minicpm":
                        import librosa
                        if rate != 16000:
                            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
                        native.append(audio)
                    n_audio += 1
                else:
                    raise ValueError("Unsupported message content")
            if n_images > 128 or n_audio > 1:
                raise ValueError("Media count exceeds budget")
            with torch.inference_mode():
                if args.backend == "minicpm":
                    # Fresh non-streaming chat, no duplex session or generated speech.
                    text = model.chat(msgs=[dict(role="user", content=native)], max_new_tokens=max_tokens,
                        max_inp_length=32768, do_sample=False, enable_thinking=False, generate_audio=False,
                        omni_mode=bool(n_images and n_audio), max_slice_nums=1, use_image_id=False)
                    if not isinstance(text, str):
                        raise ValueError("MiniCPM returned unexpected output")
                    # Conservative truncation check; custom chat does not expose generation IDs.
                    count = len(model.processor.tokenizer.encode(text, add_special_tokens=False))
                    prompt_tokens = None
                    finish = "length" if count >= max_tokens else "stop"
                else:
                    msgs = [dict(role="user", content=parts)]
                    if args.backend == "qwen3-omni":
                        text_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                        audios, images, videos = process_mm_info(msgs, use_audio_in_video=False)
                        inputs = processor(text=text_prompt, audio=audios, images=images, videos=videos,
                                           return_tensors="pt", padding=True, use_audio_in_video=False)
                    else:
                        inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                               return_dict=True, return_tensors="pt")
                    inputs = inputs.to(model.device).to(model.dtype)
                    prompt_tokens = int(inputs["input_ids"].shape[1])
                    print(json.dumps(dict(event="generation_started", backend=args.backend, prompt_tokens=prompt_tokens,
                                          preprocessing_seconds=time.monotonic()-began)), flush=True)
                    if prompt_tokens + max_tokens > 32768:
                        raise ValueError("Request exceeds 32768 token smoke budget; no silent truncation")
                    class Progress:
                        def __init__(self):
                            self.first, self.count = True, 0
                        def put(self, tokens):
                            if self.first:
                                self.first = False
                                return
                            self.count += tokens.numel()
                            if self.count % 128 == 0:
                                print(json.dumps(dict(event="generation_progress", backend=args.backend,
                                                      tokens=self.count, elapsed=time.monotonic()-began)), flush=True)
                        def end(self):
                            pass
                    generation_started = time.monotonic()
                    if args.backend == "qwen3-omni":
                        output = model.generate(**inputs, return_audio=False, thinker_max_new_tokens=max_tokens,
                                                thinker_streamer=Progress(), thinker_max_time=args.max_generation_seconds,
                                                do_sample=False, use_audio_in_video=False)
                    else:
                        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                                streamer=Progress(), max_time=args.max_generation_seconds)
                    sequences = output[0] if isinstance(output, tuple) else output
                    ids = sequences[:, prompt_tokens:]
                    count = int(ids.shape[1])
                    eos = processor.tokenizer.eos_token_id
                    finish = "length" if count >= max_tokens and int(ids[0, -1]) != eos else "stop"
                    if time.monotonic() - generation_started >= args.max_generation_seconds and int(ids[0, -1]) != eos:
                        finish = "length"
                    text = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            elapsed = time.monotonic() - began
            print(json.dumps(dict(event="request", backend=args.backend, seconds=elapsed, images=n_images,
                                  audio=n_audio, prompt_tokens=prompt_tokens, completion_tokens=count, finish_reason=finish)), flush=True)
            return dict(id="listen-local-" + str(time.time_ns()), model=args.served_model_name,
                        choices=[dict(index=0, finish_reason=finish, message=dict(role="assistant", content=text))],
                        usage=dict(prompt_tokens=prompt_tokens, completion_tokens=count),
                        annotation_runtime=dict(backend=args.backend, seconds=elapsed, input_images=n_images,
                            input_audio_sample_rates=audio_rates, audio_encoder_sample_rate=16000 if n_audio else None,
                            model_path=args.model_path, native_video_sync=False))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/v1/models":
                self.send_error(404); return
            self.reply(200, dict(data=[dict(id=args.served_model_name)]))
        def reply(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body, ensure_ascii=False).encode())
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404); return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 64 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                with lock:
                    result = infer(payload)
                self.reply(200, result)
            except Exception as exc:
                traceback.print_exc()
                torch.cuda.empty_cache()
                self.reply(400 if isinstance(exc, ValueError) else 500,
                           dict(error=dict(type=type(exc).__name__, message=str(exc))))
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
