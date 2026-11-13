#!/usr/bin/env python3
"""Independent listening experiment CLI. Run with Omni-LTX-stage2/.venv/bin/python."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p3_listen.common import DEFAULT_CONFIG, config, resolve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "export", "cache", "train", "generate", "audit", "matrix"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", choices=("train", "dev", "test"), default="train")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--s0", action="store_true", help="Use fixed S0 selection; train maximum two steps unless --steps supplied")
    p.add_argument("--part", choices=("native", "text", "vae", "all"), default="all")
    p.add_argument("--stage", choices=("P", "C0", "A"), default="P")
    p.add_argument("--steps", type=int)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--sample-id")
    p.add_argument("--mode", choices=("pred", "oracle", "prefix", "text", "audio-swap"), default="pred")
    p.add_argument("--swap-audio", type=Path)
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--oracle-fit", action="store_true", help="Engineering C0 only: fully visible GT feature diagnostic")
    a = p.parse_args()
    cfg = config(a.config)
    sys.path.insert(1, str(resolve(cfg["reuse_root"]) / "src"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if a.command == "prepare":
        from p3_listen.dataset import prepare
        prepare(cfg)
    elif a.command in ("export", "audit"):
        from p3_listen.media import run
        run(cfg, a)
    elif a.command == "cache":
        from p3_listen.features import cache
        cache(cfg, a)
    elif a.command == "train":
        from p3_listen.training import train
        train(cfg, a)
    elif a.command == "matrix":
        from p3_listen.evaluation import matrix
        matrix(cfg, a)
    else:
        from p3_listen.generation import generate
        generate(cfg, a)


if __name__ == "__main__":
    main()
