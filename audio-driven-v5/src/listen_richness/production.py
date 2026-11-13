"""Frozen v3 extraction and streaming statistics adapters for production.

The original caption and every raw response remain evidence. Validation errors
never silently turn a sample into a zero-behavior example.
"""
import json
from pathlib import Path
import sys

from .common import digest
from .extraction import SYSTEM_V3, MODEL, sentences, compile_response

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_listen_richness_pilot import aggregate, VERSION as TABLE_VERSION

VERSION='richness-full-j-v3-v1'


def contract():
    return dict(version='caption-observable-evidence-v3-selection-only',
                system_sha256=digest(SYSTEM_V3),model=MODEL,temperature=0,max_tokens=8192,input='caption_only')


def messages(caption):
    return [dict(role='system',content=SYSTEM_V3),dict(role='user',content=json.dumps(
        dict(caption=caption,sentences=[dict(id=s['id'],text=s['text']) for s in sentences(caption)]),ensure_ascii=False))]


def tokenized(snapshot, tokenizer):
    ids=tokenizer.apply_chat_template(messages(snapshot['caption']),tokenize=True,
                                      add_generation_prompt=True,enable_thinking=False)
    if len(ids)+contract()['max_tokens']>12288:raise ValueError('Context overflow; no silent truncation')
    return dict(sample_id=snapshot['sample_id'],snapshot_sha256=digest(snapshot),snapshot=snapshot,
                prompt_token_ids=ids,prompt_tokens=len(ids),sentence_count=len(sentences(snapshot['caption'])))


def validation_error(raw,caption,finish_reason):
    try:
        if finish_reason!='stop':raise ValueError('Incomplete model generation: '+str(finish_reason))
        payload=json.loads(raw)
        if payload.get('relations')!=[]:raise ValueError('v3 relations must be empty')
        compile_response(payload,caption)
        return None
    except Exception as exc:return str(exc)


def usable_behavior(b):
    """Shape-only guard; keep supported v3 records even if strict references fail."""
    if not isinstance(b,dict):return False
    if not all(k in b for k in ('id','behavior','modality','polarity','form','source_ids','repeat')):return False
    if not isinstance(b['id'],str) or not isinstance(b['behavior'],str) or not b['behavior'].strip():return False
    if b['modality'] not in ('visual','audio') or b['polarity'] not in ('positive','negative','uncertain') or b['form'] not in ('baseline','state','change'):return False
    if not isinstance(b['source_ids'],list) or not all(isinstance(x,str) for x in b['source_ids']):return False
    r=b['repeat']
    return (isinstance(r,dict) and r.get('kind') in ('explicit','multiple_unspecified','not_stated') and
            isinstance(r.get('source_ids'),list) and all(isinstance(x,str) for x in r['source_ids']) and
            'count' in r and (r['count'] is None or type(r['count']) is int))


def statistic_record(request,attempts,extraction_file):
    s=request['snapshot'];chosen=None;payload=None
    # Prefer a complete, strictly valid response; otherwise recover the newest
    # parseable response, retaining all failures and unsupported local entries.
    candidates=[a for a in attempts if not a.get('error')]+list(reversed(attempts))
    for a in candidates:
        try:p=json.loads(a['raw'])
        except (ValueError,TypeError):continue
        if isinstance(p,dict) and isinstance(p.get('behaviors'),list):chosen=a;payload=p;break
    invalid=[]
    if payload is not None:
        valid=[]
        for b in payload['behaviors']:
            (valid if usable_behavior(b) else invalid).append(b)
        row=aggregate(s,dict(behaviors=valid,relations=[]))
        status='computed' if chosen.get('error') is None and not invalid else 'recovered_with_review'
        if invalid and not valid:status='unresolved_extraction'
    else:
        row=aggregate(s,dict(behaviors=[],relations=[]));status='unresolved_extraction'
    if status=='unresolved_extraction':
        row['counted_type_count']=None;row['raw_positive_type_count']=None
        row['text_stat_status']='unresolved_extraction_not_zero'
    row.update(schema=VERSION,statistics_version=TABLE_VERSION,record_status=status,
               original_validation_error=chosen.get('error') if chosen else 'No parseable response',
               unusable_behaviors=invalid,extraction_attempt=chosen.get('attempt') if chosen else None,
               extraction_file=str(extraction_file),snapshot_sha256=request['snapshot_sha256'],
               caption_file_sha256=s['caption_file_sha256'],source=s['source'],quality=s['quality'],
               source_review_flags=s['exclusion_reasons'],crop_provenance=s.get('crop_provenance'))
    return row
