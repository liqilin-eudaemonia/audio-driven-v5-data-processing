"""Isolated text substitution around the unchanged checkpoint-native generator."""
from contextlib import contextmanager
from pathlib import Path
import importlib.util
import sys
from .common import ROOT, read, sha, digest, atomic_json, load_protocol, prompt_record


def checked_guidance(work, p, sid):
    g = prompt_record(work, sid)
    if p['guidance'].get(sid) != digest(g):
        raise ValueError('Guidance no longer matches frozen protocol')
    return g


def text_identity(c, p, g):
    from p2_listen.features import identity
    from p2_listen.config import activate
    return dict(schema='p2-guided-text-cache-v1', guidance_sha256=digest(g),
                prompt_sha256=g['prompt_sha256'], native_text=identity(c, 'text'),
                max_text_tokens=p['max_text_tokens'], code=p['code'], native_encoder_sha256=sha(activate(c)/'src/omni_av/ltx_runtime.py'))


def cache_text(work, run, sample_ids=None, device='cuda:0'):
    from p2_listen.config import extractor_config
    from p2_listen.features import load_text, unload
    from p2_listen.cache import load_part, save_part
    work = Path(work); p = load_protocol(work); c = read(Path(run)/'run.json')['config']
    selected = sample_ids if sample_ids is not None else list(p['guidance'])
    if not selected or len(set(selected)) != len(selected) or any(s not in p['guidance'] for s in selected):
        raise ValueError('Invalid selected guidance IDs')
    pending = []; results = []
    for sid in selected:
        g = checked_guidance(work, p, sid); ident = text_identity(c, p, g)
        if load_part(work/'text_cache'/sid, ident, missing_ok=True) is None:
            pending.append((sid, g, ident))
        else:
            results.append(dict(sample_id=sid, status='reused'))
    if pending:
        cfg = extractor_config(c)
        from omni_av.ltx_runtime import NativeTextConditioner
        negative = load_text(c)['negative']; encoder = NativeTextConditioner(cfg, device)
        try:
            for sid, g, ident in pending:
                from .encoding import encode_guidance
                try:
                    encoded, encoding = encode_guidance(encoder, g['prompt'], p['max_text_tokens'])
                except OverflowError as exc:
                    error = dict(sample_id=sid, status='rejected_context_overflow', error=str(exc), limit=p['max_text_tokens'],
                                 prompt_sha256=g['prompt_sha256'], truncated=False)
                    atomic_json(work/'cache_errors'/(sid+'.json'), error); results.append(error); continue
                save_part(work/'text_cache'/sid, ident, dict(positive=encoded, negative=negative,
                          prompt_sha256=g['prompt_sha256'], **encoding))
                results.append(dict(sample_id=sid, status='cached', **encoding))
        finally:
            encoder.close(); unload()
    return dict(results=results, success=all(r['status'] in ('cached','reused') for r in results))


@contextmanager
def historical_sources(run, restoration_path=None):
    """Restore only exact checkpoint-hashed bytes, never disable source checking."""
    import p2_listen.config as config
    trained = read(Path(run)/'run.json'); expected = trained['source']; candidates = {}
    if restoration_path:
        data = read(restoration_path)
        # Accept either the existing four-arm map or a single run's map.
        groups = [data] if all(isinstance(v, dict) and 'path' in v for v in data.values()) else list(data.values())
        for group in groups:
            for original, item in group.items():
                if original in expected and item['sha256'] == expected[original]:
                    if sha(item['path']) != item['sha256']:
                        raise ValueError('Historical source bytes changed')
                    candidates[original] = item['path']
    mapping = {}
    for original, h in expected.items():
        if Path(original).is_file() and sha(original) == h:
            continue
        if original not in candidates:
            raise ValueError('Checkpoint source needs exact historical restoration: '+original)
        mapping[original] = candidates[original]
    old_sha = config.sha; saved_modules = {}
    try:
        config.sha = lambda path: old_sha(mapping.get(str(path), path))
        for original, restored in mapping.items():
            if original.endswith('/src/p2_listen/training_data.py'):
                name = 'p2_listen.training_data'; saved_modules[name] = sys.modules.get(name)
                spec = importlib.util.spec_from_file_location(name, restored); module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module; spec.loader.exec_module(module)
        if config.source_identity(trained['config']) != expected:
            raise ValueError('Checkpoint source identity differs after exact restoration')
        yield mapping
    finally:
        config.sha = old_sha
        for name, old in saved_modules.items():
            if old is None: sys.modules.pop(name, None)
            else: sys.modules[name] = old


@contextmanager
def substituted_text(generation, loader):
    original = generation.load_text
    generation.load_text = loader
    try:
        yield
    finally:
        generation.load_text = original


def generate(work, run, checkpoint, sample_id, output, *, seed=137, device='cuda:0', restoration=None, variant='guided'):
    import fcntl
    from p2_listen.contracts import observation
    from p2_listen.cache import load_part
    from p2_listen.features import load_text
    from p2_glare_eval.generation import check_idle_device
    from p2_glare_eval.common import complete, artifact_valid
    if variant not in ('guided', 'fixed') or type(seed) is not int or seed < 0:
        raise ValueError('Invalid prompt variant/seed')
    work, run, checkpoint, output = map(lambda p:Path(p).resolve(), (work, run, checkpoint, output))
    p = load_protocol(work); g = checked_guidance(work, p, sample_id); trained = read(run/'run.json'); c = trained['config']
    obs = next((o for o in p['observations'] if o['sample_id'] == sample_id), None)
    if obs is None: raise ValueError('Sample is outside the frozen observations')
    obs = observation(obs)
    if checkpoint.parent != run: raise ValueError('Checkpoint must belong to selected run')
    if read(checkpoint.with_suffix('.complete.json'))['sha256'] != sha(checkpoint):
        raise ValueError('Checkpoint differs from complete marker')
    ident = dict(protocol_sha256=digest(p), sample_id=sample_id, sample_seed=seed, variant=variant,
                 guidance_sha256=digest(g) if variant=='guided' else None, checkpoint_sha256=sha(checkpoint))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if artifact_valid(output, ident): return dict(status='reused', output=str(output))
        if output.exists(): raise ValueError('Incomplete/different output exists; inspect before retry')
        # Check all selected source bytes before import of the native generator.
        with historical_sources(run, restoration) as restored:
            from p2_listen import generation as native
            cached = load_part(work/'text_cache'/sample_id, text_identity(c,p,g)) if variant=='guided' else None
            if cached is not None and (cached['prompt_sha256'] != g['prompt_sha256'] or cached['positive']['prompt'] != g['prompt']):
                raise ValueError('Text payload does not match selected prompt')
            native_text = load_text(c)  # Also verify unchanged fixed/negative cache.
            if cached is not None:
                # Negative embeddings must be byte-identical to original CFG branch.
                import torch
                if cached['negative'].keys() != native_text['negative'].keys():
                    raise ValueError('Negative text fields differ')
                for key, value in cached['negative'].items():
                    other = native_text['negative'][key]
                    if isinstance(value, torch.Tensor): equal = torch.equal(value, other)
                    else: equal = value == other
                    if not equal: raise ValueError('Negative CFG text changed')
            from p2_listen.cache import tree_to
            def loader(config, device='cpu'):
                if config != c: raise ValueError('Unexpected runtime config')
                return tree_to(cached if cached is not None else native_text, device)
            release = dict(config_sha256=digest(c), checkpoint_sha256=[ident['checkpoint_sha256']],
                           sample_ids=[sample_id], seeds=[seed], guidance_sha256=ident['guidance_sha256'],
                           evaluation_role='target_annotation_guided_diagnostic' if variant=='guided' else 'fixed_prompt_control')
            release_path = work/'releases'/(digest(ident)+'.json'); atomic_json(release_path,release,immutable=True)
            check_idle_device(device)
            with substituted_text(native, loader):
                metadata = native.generate(c,obs,output,checkpoint=checkpoint,seed=seed,device=device,release=release_path)
            # Override the native source-only metadata explicitly: labels provide
            # target information, even though no target video/audio was read.
            metadata.update(evaluation_role=release['evaluation_role'], prompt_variant=variant,
                            target_annotation_access=variant=='guided', gt_media_access=False,
                            gt_access=variant=='guided', prompt_sha256=digest(loader(c)['positive']['prompt']),
                            guidance_sha256=ident['guidance_sha256'], guidance_protocol_sha256=digest(p),
                            historical_source_restoration=restored)
            atomic_json(output/'completed.json',metadata)
            (output/'prompt.txt').write_text(loader(c)['positive']['prompt'])
            complete(output,ident,[output/n for n in ('generated.mp4','listener.wav','completed.json','prompt.txt')],
                     target_annotation_access=variant=='guided',gt_media_access=False)
            return metadata
