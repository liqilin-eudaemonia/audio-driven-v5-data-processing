from pathlib import Path
from p2_listen.contracts import ROOT, read, sha, digest, atomic_json


def code_identity():
    files = list(Path(__file__).parent.glob('*.py')) + [ROOT/'scripts/p2_guided_prompt_en.py', ROOT/'scripts/p2_eval_generate.py', ROOT/'src/p3_listen/prompts.py']
    return {str(p): sha(p) for p in sorted(files)}


def checked_code(expected):
    if expected != code_identity():
        raise ValueError('Guidance implementation changed; prepare a new version')


def prompt_record(work, sid):
    from .formatting import validate_guidance
    from p2_listen.contracts import safe_id
    return validate_guidance(read(Path(work)/'prompts'/(safe_id(sid)+'.json')))


def load_protocol(work):
    p = read(Path(work)/'protocol.json'); checked_code(p['code'])
    for name, h in p['sources'].items():
        if sha(name) != h:
            raise ValueError('Frozen source changed: '+name)
    return p
