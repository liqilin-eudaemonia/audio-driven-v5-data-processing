#!/usr/bin/env python3
"""Verify archived and live workspace files for J nod + Omni natural language captioning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "manifest.json").read_text(encoding="utf-8"))
    archive_root = here / "code"

    workspace_root = Path("/private/liqi_workspace/project_code/audio-driven-v5")
    check_workspace = workspace_root.exists()

    for rel, expected_sha in manifest["archive_sha256"].items():
        archived_file = archive_root / rel
        assert archived_file.exists(), f"Missing archived file: {archived_file}"
        actual_arch = sha256_file(archived_file)
        assert actual_arch == expected_sha, f"Archive SHA mismatch for {rel}: {actual_arch} != {expected_sha}"

        if check_workspace:
            live_file = workspace_root / rel
            assert live_file.exists(), f"Missing workspace file: {live_file}"
            actual_live = sha256_file(live_file)
            assert actual_live == expected_sha, f"Workspace SHA mismatch for {rel}: {actual_live} != {expected_sha}"

    cfg = json.loads((archive_root / "experiments/listen_j_omni_20261030/config.json").read_text(encoding="utf-8"))
    for rel, expected_sha in cfg["code_hashes"].items():
        assert manifest["production_code_sha256"].get(rel) == expected_sha, f"Config hash mismatch for {rel}"
        assert sha256_file(archive_root / rel) == expected_sha, f"Production code archive mismatch for {rel}"

    print(
        f"{manifest['version']} verified: "
        f"{len(manifest['production_code_sha256'])} production code files, "
        f"{len(manifest['archive_sha256'])} total archived files "
        f"(workspace_checked={check_workspace})"
    )


if __name__ == "__main__":
    main()
