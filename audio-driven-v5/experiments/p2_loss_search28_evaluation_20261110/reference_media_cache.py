"""Reuse exactly identical reference media/detections across candidate scorings.

Native inference still validates detector/model/version and cache identities.
Generated views retain their own media; identical-content deduplication is allowed
only after the native preparer has hashed decoded pixels and audio independently.
"""
from pathlib import Path
import copy
import fcntl
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src')]
from p2_listen.contracts import read,sha,atomic_json
from p2_event_presence.contracts import digest,rows,save_rows


def media_files(media):
    result={}
    for key,hashkey in [('visual25','visual25_sha256'),('audio','audio_file_sha256')]:
        if media.get(key):result[media[key]]=media[hashkey]
    for window in media.get('windows',[]):
        if window.get('frames'):result[window['frames']]=window['frames_sha256']
    return result


def verified(entry,contract):
    if entry['policy_identity']!=contract['policy_identity']or entry['source']!=contract['source']:
        raise ValueError('Reference cache belongs to another detector policy/source')
    media=entry['media']
    for path,expected in {**media_files(media),**entry['artifact_hashes']}.items():
        if sha(path)!=expected:raise ValueError('Cached reference artifact changed: '+path)
    detection=read(Path(media['folder'])/'detection.json')
    cue=Path(media['folder'])/'candidates.json'
    expected=digest([media,entry['detector_identity'],sha(cue)if cue.exists()else None])
    if detection['cache_identity']!=expected or detection['detector_identity']!=entry['detector_identity']or detection['media_identity']!=media['identity']:
        raise ValueError('Cached detection/media identity mismatch')
    return media


def bind(events,cache):
    """Repoint only content-verified media, never manufacture labels or scores."""
    events=Path(events);cache=Path(cache);contract=read(events/'contract.json')
    prepared=read(events/'PREPARED.json');manifest=events/'prepared_pairs.jsonl'
    if prepared.get('policy_identity')!=contract['policy_identity']or prepared.get('source')!=contract['source']:
        raise ValueError('Prepared media policy/source differs from event contract')
    if sha(manifest)!=prepared['prepared_pairs_sha256']:raise ValueError('Prepared event manifest changed')
    records=rows(manifest);replacements={};hits=[]
    for row in records:
        key=digest([row['dataset'],row['id']]);path=cache/(key+'.json')
        if not path.exists():continue
        entry=read(path);original=row['prepared_media']['reference']
        if(entry['dataset'],entry['id'])!=(row['dataset'],row['id']):raise ValueError('Wrong reference cache key')
        media=verified(entry,contract)
        if original['identity']!=media['identity']or original['media_id']!=media['media_id']:
            raise ValueError('Reference pixels/audio changed; do not reuse detection')
        replacements[media['media_id']]=media;hits.append(dict(id=row['id'],entry_sha256=sha(path)))
    generated_exact_content_hits=0
    for row in records:
        for side,original in row['prepared_media'].items():
            if original['media_id']not in replacements:continue
            media=replacements[original['media_id']]
            if original['identity']!=media['identity']:raise ValueError('Opaque media hash collision')
            if side=='generated':generated_exact_content_hits+=1
            # Native unique-media iteration keys by content ID. Rebind both sides
            # if independently prepared generated media is bit-identical; leaving
            # two folders for one ID would break the native completion hashes.
            row['prepared_media'][side]=copy.deepcopy(media)
    original_hash=sha(manifest)
    if hits:
        save_rows(manifest,records)
        prepared['prepared_pairs_sha256']=sha(manifest)
        atomic_json(events/'PREPARED.json',prepared)
    result=dict(reference_cache_hits=len(hits),reference_samples=len(records),hits=hits,
        generated_identical_content_hits=generated_exact_content_hits,
        original_prepared_sha256=original_hash,prepared_sha256=sha(manifest),
        pixels_and_own_audio_verified=True,native_detector_identity_check_still_required=True)
    atomic_json(events/'REFERENCE_MEDIA_REUSE.json',result)
    return result


def remember(events,cache):
    """Called only for the baseline after completed native detection/scoring."""
    events=Path(events);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    contract=read(events/'contract.json');prepared=read(events/'PREPARED.json')
    if prepared.get('policy_identity')!=contract['policy_identity']or prepared.get('source')!=contract['source']:
        raise ValueError('Prepared reference policy/source changed')
    if sha(events/'prepared_pairs.jsonl')!=prepared['prepared_pairs_sha256']:raise ValueError('Prepared manifest changed')
    complete=read(events/'DETECTION_COMPLETE.json');det=read(events/'detector_contract.json')
    if complete['detector_identity']!=det['identity_hash']:raise ValueError('Detector completion mismatch')
    records=rows(events/'prepared_pairs.jsonl');added=0
    with(cache/'cache.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for row in records:
            media=row['prepared_media']['reference'];folder=Path(media['folder']);detection=folder/'detection.json'
            if sha(detection)!=complete['detections'][media['media_id']]:raise ValueError('Reference detection changed after completion')
            artifacts={str(detection):sha(detection)}
            cue=folder/'candidates.json'
            if cue.exists():artifacts[str(cue)]=sha(cue)
            entry=dict(dataset=row['dataset'],id=row['id'],media=media,policy_identity=contract['policy_identity'],
                source=contract['source'],detector_identity=det['identity_hash'],artifact_hashes=artifacts)
            verified(entry,contract)
            target=cache/(digest([row['dataset'],row['id']])+'.json')
            if target.exists():
                if read(target)!=entry:raise ValueError('Frozen reference cache entry changed')
            else:atomic_json(target,entry,immutable=True);added+=1
    result=dict(reference_samples=len(records),new_entries=added,cache=str(cache),detector_identity=det['identity_hash'])
    atomic_json(events/'REFERENCE_MEDIA_CACHED.json',result)
    return result
