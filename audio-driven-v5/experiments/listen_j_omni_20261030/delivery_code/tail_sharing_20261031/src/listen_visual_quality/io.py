"""Small, explicit JSON contracts; never modify input rows or original media."""
import hashlib
import json
import os
import uuid
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def safe_id(sid):
    if not isinstance(sid, str) or not sid or Path(sid).name != sid or sid in (".", "..") or "\\" in sid:
        raise ValueError("Unsafe sample ID")
    return sid


def input_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        def stream():
            with path.open() as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        rows = stream()
    else:
        value = read(path)
        rows = value["rows"] if isinstance(value, dict) else value
    seen = set()
    for row in rows:
        sid = safe_id(row["sample_id"])
        if sid in seen:
            raise ValueError("Duplicate sample ID: " + sid)
        seen.add(sid)
        if row["source"]["sample_id"] != sid:
            raise ValueError("Source/sample ID mismatch")
        yield row


def write_jsonl(path, rows):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
