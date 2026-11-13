#!/usr/bin/env python3
"""Prepare clip labels and train with the existing J audiovisual caches."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare');p.add_argument('--cache', required=True);p.add_argument('--output', required=True)
    p.add_argument('--processes', type=int, default=16);p.add_argument('--sample-ids', nargs='+')
    p.add_argument('--absence-weight', type=float, default=.1)
    p = sub.add_parser('emit');p.add_argument('--labels', required=True);p.add_argument('--output', required=True)
    p.add_argument('--min-positive', type=int, default=5000)
    for command in ('preflight', 'train'):
        p = sub.add_parser(command);p.add_argument('--config', required=True)
        if command=='preflight':p.add_argument('--processes',type=int,default=64)
        if command == 'train':
            p.add_argument('--steps', type=int);p.add_argument('--global-batch', type=int)
            group = p.add_mutually_exclusive_group();group.add_argument('--resume');group.add_argument('--parent')
            p.add_argument('--stop-after', type=int);p.add_argument('--audit-gradients', action='store_true')
    args = parser.parse_args()
    from p2_clip_events import backend
    if args.command == 'prepare':
        from p2_clip_events.labels import prepare
        result = prepare(args.cache, args.output, processes=args.processes,
                         sample_ids=args.sample_ids, absent_weight=args.absence_weight)
        result = dict(samples=result['samples'], classes={k:len(v) for k,v in result['classes'].items()}, output=args.output)
    elif args.command == 'emit':
        result = backend.emit(args.labels, args.output, min_positive=args.min_positive)
    else:
        from p2_matrix16.config import load
        c = load(args.config)
        if args.command == 'preflight':
            result = backend.preflight(c,processes=args.processes)
            result = dict(samples=result['samples'], output=c['output'])
        else:
            from p2_matrix16.training import train
            result = train(c, resume_path=args.resume, parent=args.parent, steps=args.steps,
                           global_batch=args.global_batch, stop_after=args.stop_after,
                           audit=args.audit_gradients, backend=backend)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
