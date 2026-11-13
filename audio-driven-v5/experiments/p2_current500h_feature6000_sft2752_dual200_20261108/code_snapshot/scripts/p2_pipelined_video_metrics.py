"""Optional CPU read/pixel overlap for the unchanged paired video metric CLI."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT.parent/'OmniResponse')]


def option(args, name, default):
    return args[args.index(name)+1] if name in args else default


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--prefetch-depth', type=int, default=2)
    parser.add_argument('--decoded-gt-cache', type=Path)
    args, remaining = parser.parse_known_args()
    from evaluation.video.paired_metrics import run
    from p2_testsets_eval.metric_pipeline import MetricPipeline
    output = Path(option(remaining, '--output', ''))
    if not option(remaining, '--output', None):
        raise ValueError('--output is required')
    selected = set(option(remaining, '--metrics', ','.join(run.METRICS)).split(','))
    pipeline = MetricPipeline(run, args.prefetch_depth,
                              int(option(remaining, '--pixel-workers', '1')),
                              bool(selected & {'psnr', 'ssim'}))
    decoded_cache = None
    sys.argv = [sys.argv[0], *remaining]
    try:
        with ExitStack() as stack:
            if args.decoded_gt_cache is not None:
                from p2_testsets_eval.decoded_gt import DecodedGTCache
                decoded_cache = DecodedGTCache(args.decoded_gt_cache)
                stack.enter_context(decoded_cache.installed())
            stack.enter_context(pipeline.installed())
            run.main()
    finally:
        pipeline.close()
        if output.is_dir():
            execution = dict(cpu_pipeline=pipeline.report())
            (output/'cpu_pipeline.json').write_text(json.dumps(execution['cpu_pipeline'], indent=2)+'\n')
            if decoded_cache is not None:
                execution['decoded_gt_cache'] = decoded_cache.report()
                (output/'decoded_gt_cache.json').write_text(json.dumps(execution['decoded_gt_cache'], indent=2)+'\n')
            contract = output/'contract.json'
            if contract.exists():
                value = json.loads(contract.read_text())
                value.setdefault('execution', {}).update(execution)
                contract.write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__':
    main()
