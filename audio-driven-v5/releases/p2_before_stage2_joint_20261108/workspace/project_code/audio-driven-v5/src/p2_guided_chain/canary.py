"""Run real-model smoke tests while other chunks continue consolidating/caching."""
from pathlib import Path
import math
from p2_listen.contracts import read,atomic_json,sha,digest
from .data import link
from .prepare import prepare


def make(root,export):
 root=Path(root);out=root/'canary';spec=read(root/'pipeline.json');source=Path(export['work']);p=read(source/'protocol.json')
 atomic_json(out/'pipeline.json',spec,immutable=True)
 plans=out/'plans';plans.mkdir(exist_ok=True)
 for name in ('samples','cache','locks'):link(plans/name,root/'plans'/name)
 atomic_json(plans/'protocol.json',p,immutable=True)
 rows=read(Path(p['config']['output'])/'manifests/observations.json');selected=[o for o in rows if o['sample_id'] in p['entries']]
 view=out/'view';atomic_json(view/'observations.json',selected,immutable=True)
 atomic_json(view/'COMPLETE.json',dict(samples=len(selected),rejected=0,source_samples=len(rows),steps_per_stage=math.ceil(len(selected)/24),observation_sha256=sha(view/'observations.json'),plan_protocol_sha256=digest(p),canary_only=True),immutable=True)
 prepare(out);return out
