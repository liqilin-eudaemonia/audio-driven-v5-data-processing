"""Run the unchanged metric CLI with an explicitly audited lossless GT cache."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT.parent/'OmniResponse')]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--decoded-gt-cache', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    from p2_testsets_eval.decoded_gt import DecodedGTCache
    from evaluation.video.paired_metrics import run
    cache = DecodedGTCache(args.decoded_gt_cache)
    output = Path(remaining[remaining.index('--output')+1])
    sys.argv = [sys.argv[0], *remaining]
    try:
        with cache.installed():
            run.main()
    finally:
        if output.is_dir():
            report = cache.report()
            (output/'decoded_gt_cache.json').write_text(json.dumps(report, indent=2)+'\n')
            contract = output/'contract.json'
            if contract.exists():
                c = json.loads(contract.read_text())
                c.setdefault('execution', {})['decoded_gt_cache'] = report
                contract.write_text(json.dumps(c, indent=2)+'\n')


if __name__ == '__main__':
    main()
