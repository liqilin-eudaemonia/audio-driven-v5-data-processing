"""Explicit OpenAI-compatible HTTP transport; audio support is a server requirement."""
import os
import math
import json
import time
from urllib.parse import urlsplit
import requests
from p2_listen.contracts import atomic_json, digest
from .schema import CLASSES, parse, validate_response


def prompt(branch, duration, stage="audit"):
    modality = {"visual": "silent timestamped listener frames only", "audio": "isolated listener audio only",
                "omni": "timestamped listener frames and isolated listener audio"}[branch]
    if stage == "discovery":
        modalities = ["visual", "audio"] if branch == "omni" else [branch]
        shape = json.dumps(dict(reviewed={m: [] for m in modalities}, unknown={m: [] for m in modalities}, events=[], notes=""))
        return f'''Task: open_discovery
Observe a {duration:.9g}-second listening clip using {modality}. Discover and describe observable behaviors or sounds that may be listening feedback, without a predefined behavior taxonomy.
Do not force observations into named categories or an other bucket. Retain subtle, brief, unfamiliar and compound behaviors. Allow overlapping events. Describe concrete changes over time; do not turn every ordinary movement into a definite communicative backchannel.
Return ONE JSON object with exactly reviewed, unknown, events, notes. reviewed and unknown each have exactly these modality keys: {modalities}, with lists of [start,end] intervals. reviewed records inspected media only, NOT exhaustive detection or negative labels for any behavior. unknown records unjudgeable spans.
Required JSON container shape: {shape}
All four top-level keys are mandatory, including events when empty. reviewed and unknown MUST be objects keyed by modality, never lists. Their values contain numeric interval pairs only, never event objects. Event objects belong only in events. The empty lists above illustrate structure; fill inspected/unknown support from the actual media.
Interval syntax example: [[0.0, {min(.5, duration):.9g}]]. Use actual inspected bounds, not these example values. Full-clip inspected support is [[0.0, {duration:.9g}]] only if you inspected the whole clip. Use [] for no intervals. null, [null], strings and objects are invalid interval entries.
Each event must have start, end, modality, owner, behavior, evidence, uncertainty, boundary_censored. modality identifies the direct evidence source ({modalities}); represent separate visual and audible observations as separate events even when they overlap. owner is listener or uncertain. behavior is a free-text description, evidence is the directly observed motion or audible change, uncertainty is a string (empty if none), boundary_censored is boolean. Do not output a required type or category field. Optional confidence (0..1) and descriptive details are permitted.
The modality field inside an event MUST be a single string, never an array. start/end are numbers, not strings. If you observe no events, retain events=[] and still report inspected/unknown intervals explicitly.
Separate observable facts from interpretation. If a possible communicative meaning is recorded, put it only in an optional interpretation string as an uncertain hypothesis, never in place of evidence. Do not claim agreement, intent, emotion, or causality as observed fact. No speaker context is supplied.
Use only evidence from the supplied modality. Silent frames cannot establish audible sounds; audio cannot establish visible motion. Preserve very brief events without a 300ms minimum. Uncertain source ownership must be explicit, not silently assigned to the listener.
Times are seconds from clip start: 0 <= start < end <= {duration:.9g}. Use boundary_censored=true if an event is cut by a clip edge. Mark uncertain timing instead of inventing precision. Every event must fall within reviewed support for its modality. notes is a string. An empty event list does not establish that feedback is absent.'''
    if stage != "audit":
        raise ValueError("Unknown annotation stage")
    names = CLASSES[branch]
    shape = json.dumps(dict(reviewed={n: [] for n in names}, unknown={n: [] for n in names}, events=[], notes=""))
    rules = []
    if branch in ("visual", "omni"):
        rules.append("For head_nod require a visible temporal nodding motion, not a single downward pose or an arbitrary position change. Mark onset at the visible start of that motion and end when it settles. A repeated continuous nod sequence may be one event with count. If sampling or occlusion prevents judgment, mark unknown.")
    if branch in ("audio", "omni"):
        rules.append("For listener_vocalization inspect the actual waveform for listener vocal sounds, including very brief feedback; do not use a minimum 300ms threshold. Mark audible onset and offset, splitting separated sounds. Do not count breathing, noise or another speaker. Uncertain ownership goes in unknown, not a positive event. Optional subtype may describe the sound freely.")
    return f'''Task: primary_audit
Independently inspect the original {duration:.9g}-second clip using {modality} for these current training targets only: {names}. No prior discovery output is supplied; do not assume an omitted event is absent.
Return ONE JSON object, no prose, with exactly reviewed, unknown, events, notes. reviewed and unknown each have exactly these keys: {names}, with lists of [start,end] intervals.
Required JSON container shape: {shape}
All four keys are mandatory. reviewed and unknown MUST be objects keyed by target name whose values are lists of numeric interval pairs; never put event objects there. Event objects belong only in events. The empty lists illustrate structure, not conclusions about this clip.
Interval syntax example: [[0.0, {min(.5, duration):.9g}]]. Replace example bounds with your inspected bounds. Full-clip coverage is [[0.0, {duration:.9g}]] only after inspecting the whole clip for that target. Use [] for no intervals, never null or [null].
reviewed must explicitly identify spans inspected for that target, including checked negative spans. Only reviewed spans outside unknown may support training; events=[] alone never establishes coverage. unknown includes uncertain timing, source ownership or any unjudgeable span.
reviewed does NOT mean detected events. If you inspected the entire clip and confidently found no target event, reviewed must contain [[0.0, {duration:.9g}]], unknown is [], and events is []. If the entire clip is unjudgeable, put [[0.0, {duration:.9g}]] in unknown. Empty reviewed means you did not inspect any usable span; do not use it to mean no events.
events must have type (one of {names}), start, end, owner="listener", description, confidence (0..1), boundary_censored (boolean). Record directly supported positive events only; descriptions must not substitute inferred intent or emotion for evidence.
{" ".join(rules)}
Use only evidence from the supplied modality. Allow simultaneous events and different boundaries across modalities. Every event must lie within its reviewed coverage. Times are seconds from clip start: 0 <= start < end <= {duration:.9g}. Set boundary_censored=true for clip-edge truncation. notes is a string.'''


def validate_endpoint(cfg, branch):
    if set(cfg) != {"base_url", "model", "revision", "capabilities", "api_key_env", "max_tokens", "timeout_s", "attempts"}:
        raise ValueError("Unexpected endpoint fields; store credentials only in api_key_env")
    if not isinstance(cfg["revision"], str) or not cfg["revision"].strip():
        raise ValueError("Explicit model deployment revision required for cache identity")
    url = urlsplit(cfg["base_url"])
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("Use plain http(s) base_url without embedded credentials/query")
    if not isinstance(cfg["model"], str) or not cfg["model"] or "REPLACE" in cfg["model"] or "REPLACE" in cfg["base_url"]:
        raise ValueError("Configure an explicit model and endpoint before calling")
    required = {"image_url"} if branch == "visual" else {"input_audio"}
    if branch == "omni":
        required.add("image_url")
    if not required <= set(cfg.get("capabilities", [])):
        raise ValueError("Endpoint must explicitly support all requested content types")
    for key in ("timeout_s", "max_tokens"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("Invalid endpoint budget")
    if type(cfg["max_tokens"]) is not int:
        raise ValueError("max_tokens must be an integer")
    if type(cfg["attempts"]) is not int or not 1 <= cfg["attempts"] <= 5:
        raise ValueError("attempts must be 1..5")


def call(cfg, branch, duration, content, artifact_dir, fingerprint, request_meta, stage="audit", chat_template_kwargs=None, response_format=None):
    validate_endpoint(cfg, branch)
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key_env"):
        key = os.environ.get(cfg["api_key_env"])
        if not key:
            raise ValueError("Configured API key environment variable is missing")
        headers["Authorization"] = "Bearer " + key
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt(branch, duration, stage)}, *content]}]
    payload = dict(model=cfg["model"], messages=messages, temperature=0, max_tokens=cfg["max_tokens"])
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if response_format is not None:
        payload["response_format"] = response_format
    # Store descriptors and request hash, never credentials or huge base64 payload copies.
    atomic_json(artifact_dir / "request.json", dict(fingerprint=fingerprint, stage=stage, endpoint=cfg,
                request_sha256=digest(payload), prompt=prompt(branch, duration, stage), media=request_meta,
                chat_template_kwargs=chat_template_kwargs, response_format=response_format), immutable=True)
    for attempt in range(cfg["attempts"]):
        stamp = str(time.time_ns())
        try:
            response = requests.post(cfg["base_url"].rstrip("/") + "/chat/completions", json=payload,
                                     headers=headers, timeout=cfg["timeout_s"])
        except (requests.Timeout, requests.ConnectionError) as exc:
            atomic_json(artifact_dir / f"attempt-{stamp}.json", {"transport_error": type(exc).__name__})
            if attempt + 1 == cfg["attempts"]:
                raise RuntimeError("Transport failed after bounded retries") from None
        else:
            atomic_json(artifact_dir / f"attempt-{stamp}.json", {"status": response.status_code, "body": response.text})
            if response.status_code in (408, 429, 500, 502, 503, 504):
                if attempt + 1 == cfg["attempts"]:
                    raise RuntimeError(f"HTTP {response.status_code} after bounded retries")
            elif response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}; see saved response")
            else:
                envelope = response.json()
                choice = envelope["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Incomplete/non-stop response; no labels accepted")
                parsed = validate_response(parse(choice["message"]["content"]), branch, duration, stage)
                return dict(fingerprint=fingerprint, stage=stage, parsed=parsed, raw_response=envelope, request_meta=request_meta)
        time.sleep(min(2 ** attempt, 4))
    raise RuntimeError("Unreachable retry state")
