#!/usr/bin/env python3
"""Local entry for the VideoFDB scorer.

The dataset directory is /public/0_DATA/2_Avatar/VideoFDB. Commands run the
OpenVideoFDB environment on this machine and pass that directory through.
Hugging Face credentials already stored for the hub are used when HF_TOKEN
is not set. Judge and agent API keys still come from the environment or
evaluation/OpenVideoFDB/.env.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DATA = Path("/public/0_DATA/2_Avatar/VideoFDB")
OVFDB = Path(__file__).resolve().parent / "OpenVideoFDB"
PY = OVFDB / ".venv" / "bin" / "python"
OVFDB_BIN = OVFDB / ".venv" / "bin" / "ovfdb"


def _token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


QWEN_PYTHON = "/private/software/conda/envs/qwenomni/bin/python"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["VIDEOFDB_DATA"] = str(DATA)
    token = _token()
    if token and not env.get("HF_TOKEN"):
        env["HF_TOKEN"] = token
    if not env.get("QWEN3_OMNI_PYTHON") and Path(QWEN_PYTHON).is_file():
        env["QWEN3_OMNI_PYTHON"] = QWEN_PYTHON
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = env.get("VIDEOFDB_CUDA", "7")
    return env


def _run(args: list[str]) -> int:
    if not OVFDB_BIN.exists():
        sys.stderr.write(f"missing {OVFDB_BIN}; install OpenVideoFDB first\n")
        return 1
    cmd = [str(OVFDB_BIN), *args]
    return subprocess.call(cmd, cwd=OVFDB, env=_env())


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        sys.stdout.write(
            "usage:\n"
            "  videofdb_local.py snapshot [--validation-only]\n"
            "  videofdb_local.py manifest <validation|test>\n"
            "  videofdb_local.py bind <manifest.jsonl> <generation_map.jsonl> <run_dir>\n"
            "  videofdb_local.py score <run_dir> <manifest.jsonl>\n"
            "  videofdb_local.py report <run_dir> [<run_dir> ...] --manifest <manifest.jsonl>\n"
            "\n"
            "generation_map.jsonl rows:\n"
            '  {"clip_id":"<sample_id>","bucket":"perception","audio":"/path/listener.wav","video":"/path/listener.mp4"}\n'
            "bucket may be omitted when one file covers both rows of that clip.\n"
            "video alone is enough; its audio track is extracted.\n"
        )
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "snapshot":
        return _run(["nvidia", "snapshot", "--local-dir", str(DATA), *rest])
    if cmd == "manifest":
        split = rest[0] if rest else "validation"
        out = OVFDB / "manifest" / f"nvidia_{split}.jsonl"
        return _run(["nvidia", "manifest", "--split", split, "--local-dir", str(DATA), "--out", str(out)])
    if cmd == "bind":
        if len(rest) != 3:
            sys.stderr.write("bind needs manifest, generation_map, run_dir\n")
            return 2
        return _run(["bind-prepared", rest[0], rest[1], rest[2]])
    if cmd == "score":
        if len(rest) != 2:
            sys.stderr.write("score needs run_dir and manifest\n")
            return 2
        return _run(["score", rest[0], "--manifest", rest[1], "--judge", "qwen3-omni"])
    if cmd == "report":
        return _run(["report", *rest])
    sys.stderr.write(f"unknown command {cmd}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
