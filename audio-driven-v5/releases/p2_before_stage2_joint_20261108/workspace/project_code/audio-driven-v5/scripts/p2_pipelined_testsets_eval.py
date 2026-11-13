"""Opt-in native evaluation queue with CPU read/pixel preparation overlap.

All original actions remain available. metrics_queue launches this same entry
for its workers; only the paired-video subprocess command is wrapped. Frozen
evaluation modules and already completed score files are left unchanged.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import runpy
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT.parent/'OmniResponse')]


def pipelined_command(command):
    if not isinstance(command, (list, tuple)):
        return command
    cmd = list(map(str, command))
    wrapper = str(ROOT/'scripts/p2_pipelined_video_metrics.py')
    if len(cmd) >= 3 and cmd[1:3] == ['-m', 'evaluation.video.paired_metrics.run']:
        return [cmd[0], wrapper, '--prefetch-depth', '2', *cmd[3:]]
    if len(cmd) >= 2 and Path(cmd[1]).resolve() == ROOT/'scripts/p2_cached_video_metrics.py':
        return [cmd[0], wrapper, '--prefetch-depth', '2', *cmd[2:]]
    return command


def main():
    import p2_testsets_eval.metrics as native
    original = native.subprocess.run
    def execute(command, *args, **kwargs):
        return original(pipelined_command(command), *args, **kwargs)
    if '--work' in sys.argv:
        from p2_testsets_eval.protocol import read, save, sha
        work = Path(sys.argv[sys.argv.index('--work')+1]).resolve()
        p = read(work/'protocol.json')
        save(work/'cpu_pipeline_execution'/f'{sys.argv[1]}_{__import__("os").getpid()}.json',
             dict(at=time.time(), wrapper=str(SELF), wrapper_sha256=sha(SELF),
                  pipeline_sha256=sha(ROOT/'src/p2_testsets_eval/metric_pipeline.py'),
                  video_cli_sha256=sha(ROOT/'scripts/p2_pipelined_video_metrics.py'),
                  prefetch_depth=2, native_batch_size=p['metric_batch_size'],
                  native_metric_and_sampling_formulas_unchanged=True))
    with ExitStack() as stack:
        stack.enter_context(patch.object(native, 'CLI', SELF))
        stack.enter_context(patch.object(native.subprocess, 'run', execute))
        runpy.run_path(str(ROOT/'scripts/p2_testsets_eval.py'), run_name='__main__')


if __name__ == '__main__':
    main()
