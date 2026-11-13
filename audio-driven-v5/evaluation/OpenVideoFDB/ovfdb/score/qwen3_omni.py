"""Text and vision judge calls against a local Qwen3-Omni checkpoint.

The checkpoint path is ``QWEN3_OMNI_PATH``. Weights stay in a long-running
worker started with ``QWEN3_OMNI_PYTHON`` (the qwenomni environment), because
the scorer's own environment does not contain that model class.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

from ovfdb.score.judge import JudgeUnavailable

DEFAULT_PYTHON = "/private/software/conda/envs/qwenomni/bin/python"
WORKER = Path(__file__).with_name("qwen3_omni_worker.py")
_LOCK = threading.Lock()
_PROC: subprocess.Popen | None = None
_SEQ = 0


def max_images() -> int:
    raw = os.environ.get("QWEN3_OMNI_MAX_IMAGES", "8").strip() or "8"
    try:
        value = int(raw)
    except ValueError as exc:
        raise JudgeUnavailable(f"QWEN3_OMNI_MAX_IMAGES must be an integer, got {raw!r}") from exc
    if value < 1:
        raise JudgeUnavailable("QWEN3_OMNI_MAX_IMAGES must be at least 1")
    return value


def _python() -> str:
    chosen = os.environ.get("QWEN3_OMNI_PYTHON", "").strip() or DEFAULT_PYTHON
    if not Path(chosen).is_file():
        raise JudgeUnavailable(
            f"QWEN3_OMNI_PYTHON is not an executable: {chosen}. "
            "Point it at the environment that provides Qwen3OmniMoeForConditionalGeneration."
        )
    return chosen


def _checkpoint() -> str:
    path = os.environ.get("QWEN3_OMNI_PATH", "").strip()
    if not path:
        raise JudgeUnavailable(
            "QWEN3_OMNI_PATH is not set. Put the Qwen3-Omni model directory there before scoring."
        )
    if not Path(path).is_dir():
        raise JudgeUnavailable(f"QWEN3_OMNI_PATH is not a directory: {path}")
    return path


def _worker() -> subprocess.Popen:
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        return _PROC
    _checkpoint()
    env = os.environ.copy()
    env["QWEN3_OMNI_PATH"] = _checkpoint()
    _PROC = subprocess.Popen(
        [_python(), str(WORKER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, bufsize=1,
    )
    return _PROC


def ask(system: str, user: str, images: list[Path] | None = None) -> tuple[str, dict]:
    """One judge completion. ``images`` are JPEG frames for the generation rubric."""
    global _SEQ
    paths = [str(Path(p)) for p in (images or [])]
    with _LOCK:
        worker = _worker()
        _SEQ += 1
        request_id = _SEQ
        assert worker.stdin is not None and worker.stdout is not None
        worker.stdin.write(json.dumps({
            "id": request_id, "system": system, "user": user, "images": paths,
        }, ensure_ascii=False) + "\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
        if not line:
            err = ""
            if worker.stderr is not None:
                err = worker.stderr.read()[-1200:]
            raise JudgeUnavailable(f"Qwen3-Omni worker exited before replying. {err}".strip())
        payload = json.loads(line)
    if payload.get("error"):
        raise JudgeUnavailable(f"Qwen3-Omni judge failed: {payload['error']}")
    usage = {
        "prompt_tokens": int(payload.get("prompt_tokens") or 0),
        "completion_tokens": int(payload.get("completion_tokens") or 0),
    }
    return str(payload.get("text") or ""), usage
