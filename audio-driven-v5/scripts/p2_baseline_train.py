#!/usr/bin/env python3
"""Current500h native feature export and actual training validation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from baseline_training.data import WORK, prepare, read, export_media

p = argparse.ArgumentParser()
p.add_argument('stage', choices=['prepare', 'media', 'features', 'train', 'validate'])
p.add_argument('--method', choices=['glare', 'listenformer', 'vico', 'l2l', 'dim', 'dystream', 'avatarforcing'], default='glare')
p.add_argument('--work', type=Path, default=WORK)
p.add_argument('--ids', nargs='*')
p.add_argument('--steps', type=int, default=4)
p.add_argument('--resume', type=Path)
p.add_argument('--device', default='cuda:0')
p.add_argument('--all', action='store_true', help='export all284218 selected samples, resumably')
p.add_argument('--scope', choices=['diagnostic', 'full'], default='diagnostic')
p.add_argument('--statistics', type=Path, help='train-only statistics for full BFM training')
args = p.parse_args()
if args.stage == 'prepare':
    prepare(args.work)
elif args.stage == 'media':
    protocol = read(args.work / 'DATA_PROTOCOL.json')
    if args.all:
        from baseline_training.data import SelectedPool
        args.ids = SelectedPool().ids
    for sid in args.ids or protocol['diagnostic_train'] + protocol['diagnostic_validation']:
        export_media(sid, args.work)
elif args.stage == 'features':
    from baseline_training.features import export
    if args.all:
        from baseline_training.data import SelectedPool
        args.ids = SelectedPool().ids
    export(args.method, args.work, args.ids, args.device)
else:
    if args.method in ['l2l', 'dim']:
        from baseline_training.two_stage import run
    elif args.method in ['dystream', 'avatarforcing']:
        from baseline_training.streaming import run
    else:
        from baseline_training.training import run
    run(args.method, args.work, args.steps, args.resume, args.device, validate=args.stage == 'validate',
        full_pool=args.scope == 'full', statistics=args.statistics)
