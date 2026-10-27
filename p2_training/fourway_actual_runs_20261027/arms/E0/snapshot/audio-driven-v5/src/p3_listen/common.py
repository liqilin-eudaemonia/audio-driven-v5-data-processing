import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/p3_listen.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_once(path, value):
    path = Path(path)
    if path.exists():
        # JSON normalizes dataclass tuples to lists; compare their serialized meaning.
        if digest(read(path)) != digest(value):
            raise ValueError(f"Existing contract differs: {path}; use a new output directory")
    else:
        write(path, value)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def config(path=DEFAULT_CONFIG):
    value = read(path)
    if value["version"] != "p3-listen-fixed-v1":
        raise ValueError("Unknown listening contract")
    return value


def source_contract(cfg):
    roots = [ROOT / "src", resolve(cfg["reuse_root"]) / "src"]
    return {str(p): sha(p) for root in roots for p in sorted(root.rglob("*.py"))}


def select_rows(manifest, split, limit=0):
    if split not in ("train", "dev", "test") or limit < 0:
        raise ValueError("Invalid split/limit")
    rows = [r for r in manifest["windows"] if r["split"] == split]
    return rows[:limit] if limit else rows
