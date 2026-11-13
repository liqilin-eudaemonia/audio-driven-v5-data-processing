"""Opt-in pure AV control; leaves the running AV+Event sources unchanged."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from p2_listen.contracts import read, sha, atomic_json
from p2_matrix16.config import load
from p2_matrix16.model import MatrixModel
from p2_j_duration_execution import install_backend as duration_backend, install_sampler, Schedule

REFERENCE = ROOT/'experiments/p2_j_guided_av_event_s1_32gpu_resume200_20261101'
DEFAULT = ROOT/'configs/p2_j_av_only_20261101.json'
CLUSTER = ROOT/'configs/p2_j_av_only_32gpu_20261101.json'


def validate_control(c):
    if (c['id'], c['stage'], c['loss'], c.get('event_head_training')) != ('guided_av_s1', 1, 'av', False):
        raise ValueError('This entry requires guided S1 with event_head_training=false')
    return c


class AVOnlyModel(MatrixModel):
    def __init__(self, c, *args, **kwargs):
        validate_control(c)
        # Preserve initialization RNG consumption and checkpoint key compatibility.
        super().__init__(c, *args, **kwargs)
        self.event_heads.requires_grad_(False)

    def forward(self, bundle, pos_weights, generator, av_callback):
        if bundle.get('condition') is None:
            raise ValueError('Pure AV S1 requires frozen conditions')
        if bundle.get('teacher') is not None:
            raise ValueError('Pure AV must not load feature teachers')
        u = self.connector(bundle['condition'])
        av, components = av_callback(u, bundle['av'], bundle['text']['positive'])
        # Common loop keeps its metric schema, but no event MLP or BCE is run.
        zero = av.detach().new_zeros(())
        names = [name for values in self.spec['event_supervision']['classes'].values() for name in values]
        return dict(av=av, av_components=components, features={}, u=u,
                    event_sums={name: zero for name in names}, event_counts={name: zero for name in names})


def av_objective(values, c, counts, micro_count, world):
    validate_control(c)
    base = values['av'] * c['loss_weights']['av']
    return base / (micro_count * world), base, base.detach().new_zeros(())


def install_backend():
    """Install only inside this entry's process, as with the duration sampler."""
    from types import SimpleNamespace
    from p2_matrix16 import model, training
    native = duration_backend()
    def source_identity():
        return dict(native.source_identity(), **{str(Path(__file__).resolve()): sha(__file__)})
    backend = SimpleNamespace(**{key: getattr(native, key) for key in
        ('Dataset', 'data_config', 'data_identity', 'initialize_stage2')}, source_identity=source_identity)
    model.MatrixModel = AVOnlyModel
    training.objective = av_objective
    return backend


def preflight(c, backend):
    """Reuse the completed cache audit when every data/sampler setting matches."""
    reference = read(REFERENCE/'run.json')
    expected = copy.deepcopy(reference['experiment'])
    expected.update(id='guided_av_s1', loss='av', output=c['output'], event_head_training=False)
    expected['training']['gradient_checkpointing'] = c['training']['gradient_checkpointing']
    if c != expected:
        raise ValueError('Control differs beyond objective, output and recomputation policy')
    previous = read(REFERENCE/'preflight.json')
    sources = backend.source_identity()
    shared = {key: value for key, value in sources.items() if key != str(Path(__file__).resolve())}
    if shared != reference['source'] or previous['source'] != shared:
        raise ValueError('Shared training sources changed; cannot reuse original audit')
    identity = backend.data_identity(c)
    if identity != previous['data_identity'] or identity != reference['data_identity']:
        raise ValueError('Control data differs from audited AV+Event data')
    Schedule(c)
    result = dict(previous, source=sources, experiment_mode=['guided', 'av', 1],
                  reused_full_preflight=dict(path=str(REFERENCE/'preflight.json'),
                      sha256=sha(REFERENCE/'preflight.json'),
                      reason='Same complete caches, labels, data order and shared sources; pure AV objective only'))
    atomic_json(Path(c['output'])/'preflight.json', result, immutable=True)
    return dict(samples=result['samples'], output=c['output'], training_started=False)


def commands(cluster, config, resume=None, stop_after=None):
    from p2_j_s1_queue import commands as shared_commands
    rows = shared_commands(cluster, config, resume=resume, stop_after=stop_after)
    for row in rows:
        row[row.index(str(ROOT/'scripts/p2_clip_events.py'))] = str(Path(__file__).resolve())
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'commands', 'launch', 'train'])
    p.add_argument('--config', type=Path, default=DEFAULT)
    p.add_argument('--cluster', type=Path, default=CLUSTER)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--logs', type=Path)
    args = p.parse_args()
    c = validate_control(load(args.config, formal=True))
    if args.command == 'commands':
        print(json.dumps(commands(read(args.cluster), args.config, args.resume, args.stop_after), indent=2))
        return
    backend = install_backend()
    if args.command == 'preflight':
        result = preflight(c, backend)
    elif args.command == 'launch':
        from listen_cluster import launch
        if not args.logs:
            p.error('launch requires a new --logs directory')
        out = Path(c['output'])
        if (out/'STOP').exists():
            raise ValueError('Output STOP exists')
        if (out/'run.json').exists() and args.resume is None:
            raise ValueError('Run already exists; specify its own --resume checkpoint')
        if args.resume is not None and args.resume.resolve().parent != out.resolve():
            raise ValueError('Resume must be a checkpoint from this AV-only run')
        preflight(c, backend)
        launch(read(args.cluster), commands(read(args.cluster), args.config, args.resume, args.stop_after), args.logs)
        result = read(out/'status.json')
    else:
        from p2_matrix16.training import train
        install_sampler(c)
        result = train(c, resume_path=args.resume, stop_after=args.stop_after, backend=backend)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
