"""Explicit reuse root; no model loads or GPU side effects on import."""
import sys
from pathlib import Path

from . import VERSION
from .contracts import ROOT, EVENTS, LABEL_SCHEMA, class_subset, path_at, read, positive, sha

DEFAULT = ROOT / "configs/p2_listen_event_v2.json"


def feature_loss_enabled(c):
    """Legacy configs retain feature supervision; AV-only must opt out explicitly."""
    value = c["training"].get("feature_loss_enabled", True)
    if type(value) is not bool:
        raise ValueError("training.feature_loss_enabled must be boolean")
    return value


def training_cache_parts(c):
    return ["input", *(["teacher"] if feature_loss_enabled(c) else []), "text", "vae", "events"]


def text_lora_config(c):
    """Opt-in adapters; settings do not enter frozen feature/cache identities."""
    value = c["training"].get("text_lora", {"enabled": False})
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        raise ValueError("training.text_lora requires boolean enabled")
    required = {"enabled", "rank", "alpha", "learning_rate"} if value["enabled"] else {"enabled"}
    if set(value) != required:
        raise ValueError("Invalid text_lora fields")
    if not value["enabled"]:
        return None
    if type(value["rank"]) is not int or value["rank"] < 1:
        raise ValueError("text_lora rank must be a positive integer")
    for key in ("alpha", "learning_rate"):
        positive(value[key], "text_lora." + key)
    return {k: value[k] for k in ("rank", "alpha", "learning_rate")}


def performance_config(c):
    """Execution settings are explicit and included in checkpoint identity."""
    value = c["training"].get("performance", {})
    if not isinstance(value, dict) or set(value) - {"gradient_checkpointing", "gradient_bucket_mb"}:
        raise ValueError("Invalid training.performance fields")
    result = {"gradient_checkpointing": "all", "gradient_bucket_mb": 0, **value}
    if result["gradient_checkpointing"] not in ("all", "alternating", "three_quarters", "none"):
        raise ValueError("Invalid gradient_checkpointing policy")
    size = result["gradient_bucket_mb"]
    if type(size) is not int or size < 0:
        raise ValueError("gradient_bucket_mb must be a nonnegative integer; 0 selects legacy reductions")
    return result


def tensorboard_config(c):
    value = c["training"].get("tensorboard", {})
    if not isinstance(value, dict) or set(value) - {"enabled", "log_every", "flush_secs"}:
        raise ValueError("Invalid training.tensorboard fields")
    result = {"enabled": False, "log_every": 1, "flush_secs": 30, **value}
    if type(result["enabled"]) is not bool:
        raise ValueError("tensorboard.enabled must be boolean")
    for key in ("log_every", "flush_secs"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError("tensorboard." + key + " must be a positive integer")
    return result


def group_learning_rate(training, name):
    if name == "text_lora":
        return training["text_lora"]["learning_rate"]
    return training[{"connector": "connector_lr", "features": "feature_lr", "events": "event_lr"}[name]]


def load(path=DEFAULT):
    c = read(path)
    required = {"version", "reuse_root", "output", "extractor", "data", "events", "training", "generation"}
    if set(c) != required or c["version"] != VERSION:
        raise ValueError("Invalid P2 listening config")
    for section in ("extractor", "data", "events", "training", "generation"):
        if not isinstance(c[section], dict):
            raise ValueError(section)
    t = c["training"]
    text_lora_config(c)
    performance_config(c)
    tensorboard_config(c)
    feature_loss_weights(c["training"])
    for key in ("global_batch", "steps", "warmup_steps", "save_every", "connector_lr", "event_lr", "clip_norm"):
        positive(t[key], key)
    if feature_loss_enabled(c):
        positive(t["feature_lr"], "feature_lr")
    elif "feature_lr" in t:
        positive(t["feature_lr"], "feature_lr")
    for key in ("global_batch", "steps", "warmup_steps", "save_every"):
        if type(t[key]) is not int:
            raise ValueError(key + " must be integer")
    if c["events"]["types"] != list(EVENTS) or c["generation"]["u_in_cfg"] != "both_branches":
        raise ValueError("Unsupported event/CFG protocol")
    if c["events"].get("schema") != LABEL_SCHEMA:
        raise ValueError("Seven-class label schema required; v1 is not a resume source")
    class_subset(c["events"].get("K_train"))
    for key in ("mapping_version", "admission_policy"):
        if not isinstance(c["events"].get(key), str) or not c["events"][key].strip():
            raise ValueError(f"Explicit {key} required")
    for key in ("weight", "head_width", "pos_weight_cap"):
        positive(c["events"][key], key)
    if type(c["events"]["head_width"]) is not int:
        raise ValueError("head_width must be integer")
    if c["training"].get("weight_decay") != 0:
        raise ValueError("v2 paired protocol requires weight_decay=0")
    for key in ("feature_video_weight", "feature_audio_weight", "av_weight"):
        if c["extractor"].get(key, 1) != 1:
            raise ValueError("v2 base loss weights are fixed to 1")
    for key in ("steps", "video_cfg", "audio_cfg"):
        positive(c["generation"][key], key)
    if type(c["generation"]["steps"]) is not int:
        raise ValueError("sampling steps must be integer")
    if c["extractor"].get("vision_fps") != 8 or c["extractor"].get("audio_hz") != 8:
        raise ValueError("v2 time contract requires 8 FPS / 8 Hz")
    from listen_audio_postprocess.overlay import settings
    settings(c)  # Mandatory overlay; stale audio teacher is rejected before training.
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
    if text_lora_config(c) is not None:
        files.extend(old / "omni_av_p21" / name for name in ("__init__.py", "config.py", "lora.py"))
    # Record the actual implementations loaded by the overlay, not the legacy files.
    overlay = Path(__file__).resolve().parent
    original = ROOT / "src/p2_listen"
    files = [overlay / p.name if p.parent == original and (overlay / p.name).exists() else p for p in files]
    files.extend(p for p in overlay.glob("*.py") if p not in files)
    files.extend([overlay.parents[1] / "scripts/p2_listen.py", overlay.parents[1] / "scripts/p2_listen_base.py"])
    # Checkpoints and resume bind the actual audio composition implementation.
    files.extend((ROOT / "src/listen_audio_postprocess").glob("*.py"))
    files.extend((overlay.parents[1] / "compat").glob("*.py"))
    return {str(p): sha(p) for p in sorted(set(files))}


def feature_loss_weights(training):
    value=training.get("feature_loss_weights", {"video":1.,"audio":1.})
    if not isinstance(value,dict) or set(value)!={"video","audio"}:raise ValueError("Explicit video/audio feature_loss_weights required")
    for name,weight in value.items():
        if isinstance(weight,bool):raise ValueError("Feature weight must be numeric")
        positive(weight,"feature_loss_weights."+name)
    return dict(value)
