"""Pack tensor storages per length before IPC; unpack views in original order."""
from collections import OrderedDict
import torch


def pack_values(values):
    groups=OrderedDict()
    for i,v in enumerate(values):groups.setdefault(len(v['motion']),[]).append((i,v))
    packed=[]
    for rows in groups.values():
        positions=[i for i,v in rows];items=[v for i,v in rows]
        tensors={k:torch.stack([v[k] for v in items]) for k in items[0] if k not in ('sample_id','statistics')}
        stats={k:torch.stack([v['statistics'][k] for v in items]) for k in items[0]['statistics']}
        packed.append((positions,[v['sample_id'] for v in items],tensors,stats))
    return len(values),packed


def unpack_values(chunk):
    count,groups=chunk;values=[None]*count
    for positions,ids,tensors,stats in groups:
        for j,i in enumerate(positions):
            values[i]=dict(sample_id=ids[j],statistics={k:v[j] for k,v in stats.items()},**{k:v[j] for k,v in tensors.items()})
    return values
