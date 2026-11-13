"""Prepare/run listener response planning SFT; formal training is explicit."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    from p2_planning_sft.train import train
    train(args.config, args.resume, args.smoke, args.output, args.stop_after)
