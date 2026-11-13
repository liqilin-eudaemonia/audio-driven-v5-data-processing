"""Prepare and explicitly train the full recipe with native attention/FFN LoRA."""
from pathlib import Path
import argparse, copy, fcntl, json, os, sys
ROOT = Path(__file__).resolve().parents[1]; SELF = Path(__file__).resolve()
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read, atomic_json
from p2_ltx_all_lora_noevent.config import expected, validate, load, DEFAULT_OUTPUT, CACHE_REFERENCE as MATCHED


def layout(cluster):
    if [n['name']for n in cluster['nodes']]!=['node12','node8','node4','node11']:raise ValueError('Expected four approved nodes')
    result=[];offset=0
    for n in cluster['nodes']:
        g=list(range(1,8))if n['name']=='node11'else list(range(8))
        if n['gpus']!=g:raise ValueError('Reserve node11 GPU0, use GPU1..7')
        result.append(dict(node=n['name'],gpus=g,rank_offset=offset,ranks=list(range(offset,offset+len(g)))));offset+=len(g)
    assert offset==31
    return result


def launcher():
    import p2_whisper_event_feature_train31 as native
    native.SELF = SELF; native.layout = layout
    return native


def commands(c, resume=None, stop_after=None):
    out = Path(c['output'])
    if resume and Path(resume).resolve().parent != out.resolve():
        raise ValueError('Resume only this independent expanded-LoRA experiment')
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('Invalid debug step budget')
    result = launcher().commands(read(out/'cluster.json'), out/'config.json', resume, out/'rank_logs')
    for command in result:
        command.insert(1, 'P2_ALL_LORA_NOEVENT_STOP_AFTER=' + (str(stop_after) if stop_after else ''))
    return result


def prepare(output):
    out = output.resolve()
    if out.exists():
        raise FileExistsError('Use a new independent output')
    c = validate(expected(out))
    from p2_ltx_all_lora_noevent.schedule import Schedule
    schedule = Schedule(c)
    if schedule.size != 284218 or len(schedule.pairs) != 9169*31:
        raise ValueError('Wrong final500h schedule')
    cluster = copy.deepcopy(read(ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108/cluster.json'))
    cluster.update(training_script=str(SELF), training_config=str(out/'config.json'), master_port=29868)
    out.mkdir(parents=True)
    atomic_json(out/'config.json', c, immutable=True)
    atomic_json(out/'cluster.json', cluster, immutable=True)
    atomic_json(out/'COMMANDS.json', commands(c), immutable=True)
    result = dict(success=True, training_started=False, samples=schedule.size,
                  world=31, steps=c['training']['steps'], save_every=c['training']['save_every'],
                  initialization='pretrained_only', lora_scope=c['ltx_lora_scope'],components=dict(av=True,whisper=True,feature=True,event=False),
                  rank=c['text_lora']['rank'], reserved_gpu='node11:0', new_cache_required=False)
    atomic_json(out/'PREPARED.json', result, immutable=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'preflight', 'commands', 'train', 'launch', 'node'])
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--config', type=Path); p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after', type=int); p.add_argument('--logs', type=Path)
    p.add_argument('--node-index', type=int); p.add_argument('--attempt', type=Path)
    a = p.parse_args()
    if a.command == 'prepare':
        result = prepare(a.output)
    else:
        config = (a.config or a.output/'config.json').resolve(); c = load(config); out = Path(c['output'])
        if a.command == 'node':
            return launcher().node(c, a.node_index, a.resume, a.attempt)
        if a.command == 'commands':
            result = commands(c, a.resume, a.stop_after)
        elif a.command == 'preflight':
            import torch
            torch.set_num_threads(2)
            from p2_ltx_all_lora_noevent.training import preflight
            v = preflight(c)
            result = dict(success=True, samples=v['samples'], canary_reads=len(v['canary_reads']))
        elif a.command == 'train':
            from p2_ltx_all_lora_noevent.training import train
            stop = a.stop_after
            if stop is None and os.environ.get('P2_ALL_LORA_NOEVENT_STOP_AFTER'):
                stop = int(os.environ['P2_ALL_LORA_NOEVENT_STOP_AFTER'])
            result = train(c, a.resume, stop)
        else:
            if a.logs is None:
                p.error('Explicit --logs required for launch')
            from p2_ltx_all_lora_noevent.training import check_preflight
            from p2_ltx_all_lora_noevent.config import source_identity
            from p2_ltx_all_lora_noevent.data import data_identity
            check_preflight(c, read(out/'preflight.json'), source_identity(), data_identity(c))
            if (out/'STOP').exists() or (out/'run.json').exists() and a.resume is None:
                raise ValueError('Preserve STOP; existing run requires strict resume')
            from listen_cluster import launch
            with (out/'launch.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                resources = launcher().selected_resources(read(out/'cluster.json'))
                atomic_json(out/'RESOURCE_PRECHECK.json', resources)
                if not resources['selected_all_idle']:
                    raise RuntimeError('Selected31 GPUs busy; preserve existing tasks')
                launch(read(out/'cluster.json'), commands(c, a.resume, a.stop_after), a.logs)
            result = read(out/'status.json')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
