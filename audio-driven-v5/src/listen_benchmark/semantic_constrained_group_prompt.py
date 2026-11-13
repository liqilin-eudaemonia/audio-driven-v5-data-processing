"""Expose existing omission constraints before the model chooses representatives."""
from copy import deepcopy
from . import semantic_prompt as base
from . import semantic_fixed_group_prompt as fixed

VERSION='listen-semantic-constrained-groups-v3-extractive'
digest=base.digest
compile_plan=fixed.compile_plan
validate_proposal=base.validate_proposal
validate_review=base.validate_review
review_schema=base.review_schema
render_prompt=base.render_prompt

PLAN_PROMPT=fixed.PLAN_PROMPT.removesuffix('INPUT:\n')+'''
Each source now lists allowed_representatives: targets permitted by the existing overlap and explicit-action omission checks. Plan within these choices, not just overlapping_source_ids. The output keeper list MUST begin with mandatory_keeper_ids in exactly that order. These are sources whose only valid representative is themselves. They cannot be dropped under the existing rules.
After mandatory keepers, choose additional optional keepers as needed. Every active source needs at least one of its allowed_representatives in keepers. uncovered_by_mandatory identifies sources requiring additional choices; do not forget those sources. Never assign a source to a keeper outside its allowed_representatives. These restrictions are conservative omission alarms, not a new behavior taxonomy.
INPUT:
'''
REVIEW_PROMPT=fixed.REVIEW_PROMPT.removesuffix('INPUT:\n')+'''
Under this extractive policy, a redundant_output issue requires a specific alternative kept original that covers the WHOLE source detail for the same overlapping occurrence. Shared smile wording alone is not enough to remove an instruction that also contains a different head, eye or hand detail. Do not demand paraphrasing, assembling clauses from several originals, or changing time intervals: those edits are not allowed. Partial shared wording can be necessary to preserve all original details. Continue rejecting actual fully redundant outputs and real omissions.
INPUT:
'''


def source_payload(record):
    payload=base.source_payload(record)
    mandatory=[]
    for source in payload['visual_sources']:
        quotes=source['quote_options'];needed=set().union(*(base.action_cues(q) for q in quotes))
        allowed=([source['source_event_id']]+[other['source_event_id'] for other in payload['visual_sources']
            if other['source_event_id'] in source['overlapping_source_ids'] and other['quote_options']
            and all(needed<=base.action_cues(q) for q in other['quote_options'])]) if quotes else []
        if not base.action_cues(source['description']):allowed.append(None)
        source['allowed_representatives']=[x for x in allowed if x is not None]
        source['inactive_allowed']=None in allowed
        source['must_keep']=allowed==[source['source_event_id']]
        if source['must_keep']:mandatory.append(source['source_event_id'])
    payload['mandatory_keeper_ids']=mandatory
    payload['uncovered_by_mandatory']=[s['source_event_id'] for s in payload['visual_sources']
        if not s['inactive_allowed'] and not set(s['allowed_representatives']).intersection(mandatory)]
    return payload


def plan_schema(record):
    schema=fixed.plan_schema(record);payload=source_payload(record)
    mandatory=payload['mandatory_keeper_ids'];possible=[s['source_event_id'] for s in payload['visual_sources'] if s['quote_options']]
    optional=[sid for sid in possible if sid not in mandatory]
    schema['properties']['keepers']={'type':'array','prefixItems':[{'type':'string','const':sid} for sid in mandatory],
        'items':{'type':'string','enum':optional} if optional else False,
        'minItems':len(mandatory),'maxItems':len(possible)}
    return schema


def review_payload(record,answer):
    return fixed.review_payload(record,answer)


def materialize(record,answer,review,provenance):
    result=fixed.materialize(record,answer,review,provenance)
    result['schema']=VERSION
    result['policy']['existing_allowed_targets_visible_to_model']=True
    return result
