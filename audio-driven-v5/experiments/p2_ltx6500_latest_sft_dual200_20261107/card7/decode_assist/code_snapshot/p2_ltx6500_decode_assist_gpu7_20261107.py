"""Assist the existing GPU7 decoder with unchanged native sample claims/export."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT.parent / 'OmniResponse')]
from p2_testsets_eval.protocol import read, save, sha


def run(work, slot):
    if 'js-infer-machine-ws3qe' not in os.uname().nodename:
        raise ValueError('Only the authorized node11 may run this helper')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '7':
        raise ValueError('Only physical GPU7 is authorized')
    p = read(work / 'protocol.json')
    cfg = read(p['cluster'])
    assert [(n['host'], n['gpus']) for n in cfg['nodes']] == [('172.29.189.119', [7])]
    assert all(sha(f) == h for f, h in p['execution_code'].items())
    assert p['shared_item_queue'] and p['total_samples'] == 400
    out = work / 'card7/decode_assist'
    out.mkdir(parents=True, exist_ok=True)
    with (out / f'{slot}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        free, total = torch.cuda.mem_get_info()
        if free < 24 * 1024 ** 3:
            raise RuntimeError('Insufficient free memory for another native decoder')
        import p2_ltx6500_latest_sft_dual200_20261107 as target
        from p2_testsets_eval import runtime as rt
        from p2_testsets_eval.shared_items import all_items
        items = all_items(work, [], p)
        save(out / f'worker{slot}.json', dict(pid=os.getpid(), physical_gpu=7,
             at=time.time(), source_sha256=sha(Path(__file__)), native_decode=True,
             progress_rank=70+slot, free_bytes_before=free))
        with target.base.runtime(work):
            # Original decoder, codecs, VAE tiling, trimming, FP precision and
            # sample-level flock/atomic commit are retained. Only scheduling changes.
            rt.decode(work, items, 70+slot, p, p['arms'])
        save(out / f'complete{slot}.json', dict(success=True, at=time.time(), physical_gpu=7))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--slot', type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()
    try:
        run(args.work.resolve(), args.slot)
    except Exception as exc:
        save(args.work / 'card7/decode_assist' / f'failure{args.slot}.json',
             dict(error=repr(exc), traceback=traceback.format_exc(), at=time.time()))
        raise
