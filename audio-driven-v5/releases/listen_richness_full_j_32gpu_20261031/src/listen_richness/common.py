import hashlib
import json
import math
from pathlib import Path

from listen_visual_quality.io import write, write_jsonl, safe_id, file_hash


def digest(value, *, ascii=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=ascii, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def union(intervals):
    """Half-open intervals; merge overlaps/touching only, never bridge a gap."""
    out = []
    for a, b in sorted(intervals):
        number(a); number(b)
        if b <= a:
            raise ValueError("Invalid interval")
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    p = (len(values) - 1) * q
    a = math.floor(p); b = math.ceil(p)
    return values[a] + (values[b] - values[a]) * (p - a)
