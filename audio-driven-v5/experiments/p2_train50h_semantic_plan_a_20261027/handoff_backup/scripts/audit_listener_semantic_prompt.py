#!/usr/bin/env python3
"""Replay accepted proposals on CPU and audit immutable audio/source evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.semantic_prompt import compile_plan, digest, materialize, render_prompt, PLAN_PROMPT, REVIEW_PROMPT, source_payload, review_payload


def audit(inputs, run):
    jobs = [json.loads(line) for line in inputs.open() if line.strip()]
    protocol = json.loads((run / 'protocol.json').read_text())
    if digest(jobs) != protocol['input_sha256']:
        raise ValueError('Input identity changed')
    if 'controller_sha256' in protocol:
        controller=run/'code/scripts/run_listener_semantic_prompt_cluster24.py'
        if hashlib.sha256(controller.read_bytes()).hexdigest()!=protocol['controller_sha256']:
            raise ValueError('Frozen cluster controller changed')
    for name, expected in protocol['code'].items():
        if hashlib.sha256((run / 'code' / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen code changed: ' + name)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Use the matching code snapshot to replay: ' + name)
    groups = {}; rows = []; missing = []
    for job in jobs:
        path = run / 'results' / (job['job_id'] + '.json')
        if not path.exists():
            missing.append(job['job_id']); continue
        result = json.loads(path.read_text())
        if result['provenance']['protocol_sha256'] != digest(protocol):
            raise ValueError('Result protocol mismatch')
        attempt = result['provenance']['attempt']
        proposal = json.loads((run / 'attempts' / f'{job["job_id"]}.proposal.{attempt}.json').read_text())
        review = json.loads((run / 'attempts' / f'{job["job_id"]}.review.{attempt}.json').read_text())
        for raw in (proposal, review):
            if raw['finish_reason'] != 'stop' or raw['protocol_sha256'] != digest(protocol):
                raise ValueError('Invalid attempt identity/completion')
            if 'worker_protocol_sha256' in raw:
                worker=json.loads(Path(raw['worker_protocol_path']).read_text())
                if digest(worker)!=raw['worker_protocol_sha256'] or worker['code']!=protocol['code']:
                    raise ValueError('Aggregated worker identity mismatch')
        answer=compile_plan(job['record'], json.loads(proposal['raw']))
        base=PLAN_PROMPT+json.dumps(source_payload(job['record']),ensure_ascii=False)
        if not proposal['prompt'].startswith(base):raise ValueError('Proposal request differs from original source')
        suffix=proposal['prompt'][len(base):]
        if suffix and not suffix.startswith('\nPrevious attempt failed validation. Correct it without dropping genuine actions:\n'):
            raise ValueError('Unexpected proposal request suffix')
        expected_review=REVIEW_PROMPT+json.dumps(review_payload(job['record'],answer),ensure_ascii=False)
        if review['prompt']!=expected_review:raise ValueError('Review belongs to a different proposal')
        rebuilt = materialize(job['record'], answer,
                              json.loads(review['raw']), result['provenance'])
        if rebuilt != result:
            raise ValueError('Replay differs: ' + job['job_id'])
        if render_prompt(result) != (run / 'prompts' / (job['job_id'] + '.txt')).read_text():
            raise ValueError('Prompt differs: ' + job['job_id'])
        original_audio = sorted((e for e in job['record']['events'] if e['modality'] == 'audio'), key=lambda e: e['event_id'])
        derived_audio = sorted((e for e in result['events'] if e['modality'] == 'audio'), key=lambda e: e['event_id'])
        if original_audio != derived_audio or job['record'] != result['source_record']:
            raise ValueError('Original audio or source changed')
        before = [e for e in job['record']['events'] if e['modality'] == 'visual']
        after = [e for e in result['events'] if e['modality'] == 'visual']
        stats = groups.setdefault(job['provenance']['role'], Counter())
        stats.update(samples=1, input_visual=len(before), output_visual=len(after), audio_unchanged=len(original_audio),
                     input_visual_characters=sum(len(e['description']) for e in before),
                     output_visual_characters=sum(len(e['description']) for e in after),
                     samples_with_conflicts=int(bool(result['conflicts'])))
        dispositions = Counter(d['disposition'] for d in result['source_decisions'])
        stats.update({'decision_' + key: value for key, value in dispositions.items()})
        rows.append(dict(job_id=job['job_id'], before=len(before), after=len(after), audio=len(original_audio),
                         conflicts=len(result['conflicts']), dispositions=dict(dispositions)))
    return dict(status='passed' if not missing else 'partial', requested=len(jobs), verified=len(rows),
                missing_results=missing, groups=groups, samples=rows,
                checks=['frozen_code_identity', 'exact_proposal_replay', 'exact_prompt_replay',
                        'original_source_unchanged', 'all_audio_fields_unchanged', 'source_timing_and_quote_validation',
                        'exact_visible_prefix', 'explicit_action_omission_guard', 'review_bound_to_actual_proposal'],
                semantic_correctness_proven=False, media_rechecked=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.inputs, args.run)
    (args.run / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('samples', 'missing_results')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
