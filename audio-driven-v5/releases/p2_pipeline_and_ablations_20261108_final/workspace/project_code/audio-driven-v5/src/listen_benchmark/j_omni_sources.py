"""Resolve frozen source ownership independently from GPU execution nodes."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .j_annotation import read


@lru_cache(maxsize=4)
def load_plan(path, sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('Source ownership manifest changed')
    plan = json.loads(raw)
    if plan['schema'] != 'j-omni-source-ownership-v1':
        raise ValueError('Unexpected source ownership schema')
    return plan


def source_plan(config):
    spec = config.get('source_ownership')
    return load_plan(spec['path'], spec['sha256']) if spec else None


def source_nodes(config):
    plan = source_plan(config)
    return plan['nodes'] if plan else config['nodes']


def journal_for(node, queue_state):
    old = Path(node['old_work'])
    if not node.get('imported_completed_journal'):
        return old/'ready.jsonl', queue_state['offset']
    # A filtered historical import is eligible only after its real nod queue
    # has drained; it never fabricates a detection/completion receipt.
    if (queue_state['inflight'] or queue_state['failed'] or
            queue_state['offset'] < (old/'ready.jsonl').stat().st_size):
        raise ValueError('Imported historical nod queue is not complete')
    journal = Path(node['imported_completed_journal'])
    raw = journal.read_bytes()
    if hashlib.sha256(raw).hexdigest() != node['journal_sha256']:
        raise ValueError('Imported source journal changed')
    return journal, len(raw)


def resolve_nod_receipt(config, old, item):
    old = Path(old)
    jid = item['job_id']
    done = old/'nod/completed'/f'{jid}.json'
    if not done.exists():
        return None
    receipt = read(done)
    if receipt.get('status') == 'delegated_branch_not_measurement':
        plan = source_plan(config)
        expected = plan and plan.get('delegated_nod_owners', {}).get(jid)
        if not expected or receipt.get('owner_work') != expected:
            raise ValueError('Unverified delegated nod owner: '+jid)
        old = Path(expected)
        done = old/'nod/completed'/f'{jid}.json'
        if not done.exists():
            return None
        receipt = read(done)
    if (receipt.get('job_id') != jid or receipt.get('clips') != item['count'] or
            not receipt.get('source_result_sha256')):
        raise ValueError('Nod receipt mismatch: '+jid)
    return old, done, receipt


def validate_item(node, item):
    allowed = node.get('allowed_jobs')
    if allowed is not None and allowed.get(item['job_id']) != item['count']:
        raise ValueError('Source outside frozen ownership scope: '+item['job_id'])
