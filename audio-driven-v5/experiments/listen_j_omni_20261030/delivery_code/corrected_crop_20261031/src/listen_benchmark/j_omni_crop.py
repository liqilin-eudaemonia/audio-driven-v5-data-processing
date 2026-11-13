"""Apply the reviewed bounded crop after immutable, original-input quality checks.

Completed captions are never rewritten. Corrected nod measurements have their own
cache, and captions record both their actual input and their original J source.
"""
from functools import lru_cache
from pathlib import Path
import os
import subprocess
import json
import select
import threading

from .j_annotation import read, write, locked, result_path
from listen_visual_quality.io import canonical_hash, file_hash

REVISION = 'quality-bounded-crop-v1-20261030'
_services = {}
_services_lock = threading.Lock()


def _measure(request, settings, cfg, resources, cache):
    """One persistent CPU service per preparer, bounded by host-wide slots."""
    key=(settings['runtime_python'],str(resources))
    with _services_lock:
        service=_services.get(key)
        if service is None or service['process'].poll() is not None:
            log=cache/'runtime_logs'/os.uname().nodename/f'{os.getpid()}.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GLOG_minloglevel='2')
            with log.open('a') as output:
                process=subprocess.Popen([settings['runtime_python'],
                    str(Path(cfg['project_root'])/'scripts/listen_j_corrected_nod.py'),'--serve'],
                    env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=output,text=True,bufsize=1)
            service=dict(process=process,lock=threading.Lock(),log=str(log));_services[key]=service
    with service['lock']:
        process=service['process']
        process.stdin.write(json.dumps(request)+'\n');process.stdin.flush()
        if not select.select([process.stdout],[],[],300)[0]:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
            raise TimeoutError('Corrected nod service timed out; see '+service['log'])
        line=process.stdout.readline()
        if not line:raise RuntimeError('Corrected nod service exited; see '+service['log'])
        reply=json.loads(line)
        if not reply.get('ok') or reply.get('output')!=request['output']:
            raise RuntimeError('Corrected nod service failed: '+str(reply)+'; see '+service['log'])


@lru_cache(maxsize=64)
def _receipt(path, size, mtime):
    return read(path)


def effective_row(row, cfg, decision=None):
    """Validate the existing measurement before applying its approved proposal."""
    contract = cfg['bounded_crop_input']
    if contract['revision'] != REVISION:
        raise ValueError('Unknown crop revision')
    gate = cfg['visual_quality_gate']
    if decision is None:
        path = Path(gate['work'])/'measure/completed'/f"{row['source']['pair']}.json"
        st = path.stat(); receipt = _receipt(str(path), st.st_size, st.st_mtime_ns)
        if receipt['contract_sha256'] != gate['contract_sha256']:
            raise ValueError('Crop measurement policy mismatch')
        decisions = [d for d in receipt['decisions'] if d['sample_id'] == row['sample_id']]
        if len(decisions) != 1:
            raise ValueError('Missing or duplicate crop decision')
        d = decisions[0]
    else:
        # Upstream nod owns an early measurement receipt before the source is
        # admitted to the downstream Omni queue. Its contract is identical.
        d = decision
        if d['sample_id'] != row['sample_id'] or d['contract_sha256'] != gate['contract_sha256']:
            raise ValueError('Early crop decision identity/contract mismatch')
    if d['decision'] != 'retain' or d['source_sha256'] != canonical_hash(row['source']):
        raise ValueError('Only matching retained inputs can receive corrected crops')
    p = Path(d['measurement_file'])
    if file_hash(p) != d['measurement_sha256']:
        raise ValueError('Crop measurement changed')
    measurement = read(p)
    if measurement['contract_sha256'] != gate['contract_sha256']:
        raise ValueError('Crop measurement contract differs')
    from .j_quality_pipeline import cropped_row
    applied = cropped_row(row, measurement, d, p)
    before = row['source']['video_transform']['crop']
    after = applied['source']['video_transform']['crop']
    provenance = dict(revision=REVISION, original_crop=before, actual_crop=after,
        coordinates_changed=before != after, proposal_status=measurement['crop']['status'],
        original_source_sha256=canonical_hash(row['source']),
        effective_source_sha256=canonical_hash(applied['source']),
        measurement_file=str(p), measurement_sha256=d['measurement_sha256'],
        quality_contract_sha256=gate['contract_sha256'])
    return applied, provenance


def corrected_nod(original, applied, old_work, work, resources, cfg):
    """Reuse identical pixels; remeasure changed crops with the frozen CPU detector."""
    from .j_omni import nod_evidence
    if original['source']['video_transform'] == applied['source']['video_transform']:
        evidence = nod_evidence(original, old_work)
        return dict(evidence, crop_reuse='identical_video_transform')
    upstream = result_path(old_work, 'nod', applied['sample_id'])
    if upstream.exists() and read(upstream).get('source') == applied['source']:
        return dict(nod_evidence(applied, old_work), crop_reuse='upstream_corrected_crop')
    settings = cfg['bounded_crop_input']; cache = Path(settings['nod_cache'])
    sid = applied['sample_id']; target = result_path(cache, 'nod', sid)
    with locked(str(target)+'.lock'):
        if not target.exists():
            from listen_visual_native8_nod import decode_slot
            request_path = result_path(cache, 'requests', sid)
            write(request_path, applied)
            # This limit is separate from source decoding and is shared per host.
            with decode_slot(Path(resources)/'corrected_nod_slots', settings['nod_slots_per_host']):
                _measure(dict(row=str(request_path),output=str(target),resources=str(resources)),
                         settings,cfg,resources,cache)
    return dict(nod_evidence(applied, cache), crop_reuse='remeasured_corrected_crop')


def result_sources(row, prepared):
    if 'effective_source' not in prepared:
        return dict(source=row['source'])
    return dict(source=prepared['effective_source'], original_source=row['source'],
                crop_provenance=prepared['evidence']['crop_application'])


def attempts_root(work, cfg):
    # Pending old attempts remain intact and can never masquerade as new inputs.
    return Path(work)/('attempts_corrected_crop_v1' if cfg.get('bounded_crop_input') else 'attempts')


def caption_crop_info(result):
    """Read legacy provenance without mutating immutable completed captions."""
    if 'crop_provenance' in result:
        return dict(result['crop_provenance'], label_origin='corrected_crop_pipeline')
    return dict(revision=result['source']['crop_proposal'].get('revision'),
                actual_crop=result['source']['video_transform']['crop'],
                label_origin='legacy_original_crop_preserved',
                final_crop_applied=False)
