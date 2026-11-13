#!/usr/bin/env python3
"""Prepare frozen annotation-guided Gemma/Qwen conditions, without training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Stream cleaned annotations; no model/GPU load')
    prep.add_argument('--config', required=True)
    prep.add_argument('--work', required=True)
    prep.add_argument('--sample-id', action='append')
    prep.add_argument('--max-text-tokens', type=int, choices=(1024, 2048, 4096, 8192), default=8192)
    prep.add_argument('--max-qwen-tokens', type=int, default=32768)
    for name in ('cache-text', 'cache-input-plan', 'verify'):
        p = sub.add_parser(name)
        p.add_argument('--work', required=True)
        p.add_argument('--sample-id', action='append')
        p.add_argument('--shard-index', type=int, default=0)
        p.add_argument('--shards', type=int, default=1)
        if name == 'verify':
            p.add_argument('--part', choices=('text', 'input_plan'), action='append')
        else:
            p.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.command == 'prepare':
        from p2_plan_cache.protocol import prepare
        result = prepare(args.config, args.work, args.sample_id, args.max_text_tokens, args.max_qwen_tokens)
    else:
        from p2_plan_cache.protocol import open_protocol
        from p2_listen.config import activate
        activate(open_protocol(args.work, verify_sources=False)['config'])
        kwargs = dict(sample_ids=args.sample_id, shard_index=args.shard_index, shards=args.shards)
        if args.command == 'verify':
            from p2_plan_cache.reader import verify
            result = verify(args.work, parts=args.part or ('text', 'input_plan'), **kwargs)
        else:
            from p2_plan_cache.runtime import cache
            result = cache(args.work, 'text' if args.command == 'cache-text' else 'input_plan', device=args.device, **kwargs)
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}, ensure_ascii=False, indent=2))
    return 0 if result['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
