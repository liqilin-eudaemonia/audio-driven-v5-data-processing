"""Prove identical Gemma sources after moving them into the private LTX copy."""
import copy
from pathlib import Path

from p2_listen.contracts import read
from .bootstrap import CORE

GEMMA_FILES = {'text_encoders/gemma/'+name for name in
               ('embeddings_processor.py','embeddings_connector.py','tokenizer.py')}


def canonical_dependencies(value, upstream=None):
    """Relocate only three byte-identical files; never relax an asset/hash check."""
    upstream = read(CORE/'UPSTREAM.json') if upstream is None else upstream
    result = copy.deepcopy(value);code = {}
    for name,h in result['code'].items():
        path = Path(name)
        if path.is_relative_to(CORE/'ltx_core'):
            rel = str(path.relative_to(CORE/'ltx_core'))
            if rel not in GEMMA_FILES or upstream['files'].get(rel) != h:
                raise ValueError('Changed private Gemma source cannot reuse baseline text caches')
            name = str(Path(upstream['origin'])/rel)
        if name in code:
            raise ValueError('Duplicate dependency after Gemma relocation')
        code[name] = h
    result['code'] = code
    return result


def cache_reader_class():
    from p2_j_cache.runtime import CacheReader
    class CompatibleCacheReader(CacheReader):
        def __init__(self, work):
            from p2_j_cache.data import open_manifest
            from p2_listen.config import activate
            from p2_plan_cache.runtime import dependencies
            from p2_listen.features import weights_identity
            self.work=Path(work);self.manifest=open_manifest(work);activate(self.manifest['config'])
            paths=sorted((self.work/'assets').glob('gpu*.json'))
            if not paths:
                raise ValueError('No completed baseline worker assets')
            self.assets=read(paths[0])
            if any(read(p)!=self.assets for p in paths[1:]):
                raise ValueError('Baseline workers used different model assets')
            c=self.manifest['config'];q=dependencies(c,'input_plan')
            t=canonical_dependencies(dependencies(c,'text'))
            current=dict(input=q,input_plan=q,text_guided=t,text=t,vae=dict(ltx=weights_identity(c,'ltx')))
            if current != self.assets:
                raise ValueError('Baseline cache dependency changed beyond byte-identical Gemma relocation')
    return CompatibleCacheReader
