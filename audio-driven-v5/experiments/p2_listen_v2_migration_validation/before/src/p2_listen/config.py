"""Explicit reuse root; no model loads or GPU side effects on import."""
import sys
from pathlib import Path

from . import VERSION
from .contracts import ROOT, path_at, read, positive, sha

DEFAULT = ROOT / "configs/p2_listen_event_base.json"


def load(path=DEFAULT):
    c = read(path)
    required = {"version", "reuse_root", "output", "extractor", "data", "events", "training", "generation"}
    if set(c) != required or c["version"] != VERSION:
        raise ValueError("Invalid P2 listening config")
    for section in ("extractor", "data", "events", "training", "generation"):
        if not isinstance(c[section], dict):
            raise ValueError(section)
    t = c["training"]
    for key in ("global_batch", "steps", "warmup_steps", "save_every", "connector_lr", "feature_lr", "event_lr", "clip_norm"):
        positive(t[key], key)
    for key in ("global_batch", "steps", "warmup_steps", "save_every"):
        if type(t[key]) is not int:
            raise ValueError(key + " must be integer")
    if c["events"]["types"] != ["listener_vocalization", "head_nod"] or c["generation"]["u_in_cfg"] != "both_branches":
        raise ValueError("Unsupported event/CFG protocol")
    for key in ("weight", "head_width", "pos_weight_cap"):
        positive(c["events"][key], key)
    if type(c["events"]["head_width"]) is not int:
        raise ValueError("head_width must be integer")
    if c["training"].get("weight_decay") != 0:
        raise ValueError("v1 paired protocol requires weight_decay=0")
    for key in ("feature_video_weight", "feature_audio_weight", "av_weight"):
        if c["extractor"].get(key, 1) != 1:
            raise ValueError("v1 base loss weights are fixed to 1")
    for key in ("steps", "video_cfg", "audio_cfg"):
        positive(c["generation"][key], key)
    if type(c["generation"]["steps"]) is not int:
        raise ValueError("sampling steps must be integer")
    if c["extractor"].get("vision_fps") != 8 or c["extractor"].get("audio_hz") != 8:
        raise ValueError("v1 time contract requires 8 FPS / 8 Hz")
    return c


def activate(c):
    root = path_at(c["reuse_root"])
    if not (root / "src/omni_av_p2/connector.py").is_file():
        raise FileNotFoundError("P2 reuse_root is unavailable")
    prior = sys.modules.get("omni_av_p2")
    if prior and not Path(prior.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Another P2 worktree is already imported")
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    return root


def extractor_config(c):
    activate(c)
    from omni_av_p2.config import Config
    return Config(**c["extractor"])


def source_identity(c):
    old = activate(c) / "src"
    files = list((ROOT / "src/p2_listen").glob("*.py")) + [ROOT / "src/p3_listen/prompts.py", ROOT / "src/p3_listen/features.py"]
    for directory in ("omni_av", "omni_av_p2", "omni_av_p3_staged"):
        files.extend((old / directory).glob("*.py"))
    return {str(p): sha(p) for p in sorted(files)}
