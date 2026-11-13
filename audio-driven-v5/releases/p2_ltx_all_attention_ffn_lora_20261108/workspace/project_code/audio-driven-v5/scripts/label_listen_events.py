#!/usr/bin/env python3
"""Offline multimodal event annotation; --help requires no model dependencies."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--manifest", required=True)
        p.add_argument("--mode", choices=("split", "omni", "visual"), default="split")
        if name == "run":
            p.add_argument("--output", required=True)
    p = sub.add_parser("export")
    p.add_argument("--summary", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reviews")
    p.add_argument("--allow-weak-train", action="store_true")
    for name in ("prepare-realtalk", "prepare-seamless"):
        p = sub.add_parser(name, help="Offline crop/512 resize preparation; no model requests")
        p.add_argument("--samples", required=True, help="RealTalk samples.json or Seamless items.json")
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--preview-ids", nargs="*", default=[])
    args = parser.parse_args()
    from listen_labeling.pipeline import export, preflight, run
    from p2_listen.contracts import read
    if args.command in ("prepare-realtalk", "prepare-seamless"):
        if args.command == "prepare-realtalk":
            from listen_labeling.realtalk import prepare_package
        else:
            from listen_labeling.seamless import prepare_package
        result = prepare_package(args.samples, args.output, read(args.config)["media"], args.preview_ids)
    elif args.command == "export":
        result = export(args.summary, args.output, args.reviews, args.allow_weak_train)
    elif args.command == "validate":
        result = preflight(args.manifest, read(args.config), args.mode)
    else:
        result = run(args.manifest, read(args.config), args.output, args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
