"""Prepare speaker Whisper caches or explicitly train the Qwen+Whisper AV arm."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_whisper_av.config import DEFAULT, load
from p2_listen.contracts import read

CLUSTER = ROOT/'configs/p2_j_qwen_whisper_av_32gpu_20261104.json'


def commands(cluster,config,*,resume=None,stop_after=None):
    from p2_j_s1_queue import commands as shared_commands
    result = shared_commands(cluster,config,resume=resume,stop_after=stop_after)
    for row in result:
        row[row.index(str(ROOT/'scripts/p2_clip_events.py'))] = str(Path(__file__).resolve())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('cache-prepare','cache-extract','cache-finalize','preflight','commands','train','launch'))
    p.add_argument('--config',type=Path,default=DEFAULT)
    p.add_argument('--cluster',type=Path,default=CLUSTER)
    p.add_argument('--sample-ids',type=Path,help='JSON array for an explicitly separate pilot cache')
    p.add_argument('--rank',type=int,default=int(os.environ.get('RANK',0)))
    p.add_argument('--world-size',type=int,default=int(os.environ.get('WORLD_SIZE',1)))
    p.add_argument('--device',default=None)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--read-workers',type=int,default=2)
    p.add_argument('--processes',type=int,default=16)
    p.add_argument('--limit',type=int)
    p.add_argument('--resume',type=Path)
    p.add_argument('--stop-after',type=int)
    p.add_argument('--logs',type=Path)
    a = p.parse_args();c = load(a.config)
    if a.command == 'cache-prepare':
        from p2_whisper_av.cache import prepare
        result = prepare(c,sample_ids=None if a.sample_ids is None else read(a.sample_ids))
    elif a.command == 'cache-extract':
        import torch
        from p2_whisper_av.cache import extract
        device = a.device or ('cuda:'+os.environ.get('LOCAL_RANK','0') if torch.cuda.is_available() else 'cpu')
        if device.startswith('cuda'):
            torch.cuda.set_device(torch.device(device))
        torch.set_num_threads(2)
        result = extract(c,rank=a.rank,world=a.world_size,device=device,batch_size=a.batch_size,
                         read_workers=a.read_workers,limit=a.limit)
    elif a.command == 'cache-finalize':
        from p2_whisper_av.cache import finalize
        result = finalize(c)
    elif a.command == 'commands':
        result = commands(read(a.cluster),a.config,resume=a.resume,stop_after=a.stop_after)
    elif a.command == 'preflight':
        from p2_whisper_av.training import preflight
        result = preflight(c,processes=a.processes)
    elif a.command == 'train':
        from p2_whisper_av.training import train
        result = train(c,resume_path=a.resume,stop_after=a.stop_after)
    else:
        from listen_cluster import launch
        from p2_whisper_av.training import preflight
        if a.logs is None:
            p.error('launch requires a new --logs directory')
        if (Path(c['output'])/'STOP').exists():
            raise ValueError('Persistent training STOP exists')
        if (Path(c['output'])/'run.json').exists() and a.resume is None:
            raise ValueError('Run exists; resume its own complete checkpoint explicitly')
        if a.resume is not None and a.resume.resolve().parent != Path(c['output']).resolve():
            raise ValueError('Resume must belong to this Whisper experiment')
        preflight(c,processes=a.processes)
        launch(read(a.cluster),commands(read(a.cluster),a.config,resume=a.resume,stop_after=a.stop_after),a.logs)
        result = read(Path(c['output'])/'status.json')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
