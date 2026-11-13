import json
from pathlib import Path
from .common import ROOT, read, sha, digest, atomic_json, code_identity
from .formatting import render


def prepare(observations, annotations, output, max_text_tokens=4096):
    if max_text_tokens not in (1024,2048,4096): raise ValueError("Invalid maximum text length")
    from p2_listen.contracts import observation
    observations, annotations, output = map(lambda p: Path(p).resolve(), (observations, annotations, output))
    obs = [observation(o, observations.parent) for o in read(observations)]
    ids = {o['sample_id'] for o in obs}
    if not obs or len(ids) != len(obs):
        raise ValueError('Expected nonempty unique observations')
    records = {}
    with annotations.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            if r['sample_id'] not in ids:
                continue
            if r['sample_id'] in records:
                raise ValueError('Duplicate selected annotation sample')
            records[r['sample_id']] = r
    if set(records) != ids:
        raise ValueError('Missing integrated annotation samples: '+str(sorted(ids-set(records))))
    outputs = []
    for o in obs:
        r = records[o['sample_id']]
        if abs(r['duration_s'] - o['requested_duration_s']) > 1e-6:
            raise ValueError('Annotation and observation durations differ')
        outputs.append(render(r))
    p = dict(schema='p2-english-guided-prompt-protocol-v1', evaluation_role='target_annotation_guided_diagnostic',
             target_annotation_access=True, formal_source_only_evaluation=False, observations=obs,
             guidance={r['sample_id']:digest(r) for r in outputs}, code=code_identity(),
             sources={str(observations):sha(observations), str(annotations):sha(annotations)},
             max_text_tokens=max_text_tokens,
             token_policy='pad to 1024/2048/4096 within configured limit; exact original token sequence checked; reject overflow')
    # Validate existing contract before writing any per-sample files.
    if (output/'protocol.json').exists() and read(output/'protocol.json') != p:
        raise ValueError('Output protocol already differs; choose a new directory')
    for r in outputs:
        atomic_json(output/'prompts'/(r['sample_id']+'.json'), r, immutable=True)
        (output/'prompts'/(r['sample_id']+'.txt')).write_text(r['prompt'])
    atomic_json(output/'protocol.json', p, immutable=True)
    return dict(samples=len(outputs), output=str(output), evaluation_role=p['evaluation_role'])
