"""Constrain response syntax during decoding, never repair model labels after generation."""
from .schema import CLASSES

def response_format(branch,duration,stage):
    number=dict(type='number',minimum=0,maximum=duration)
    spans=dict(type='array',items=dict(type='array',items=number,minItems=2,maxItems=2))
    keys=(['visual','audio'] if branch=='omni' else [branch]) if stage=='discovery' else CLASSES[branch]
    support=dict(type='object',properties={k:spans for k in keys},required=keys,additionalProperties=False)
    props=dict(start=number,end=number,owner=dict(type='string',enum=['listener','uncertain'] if stage=='discovery' else ['listener']),boundary_censored=dict(type='boolean'))
    if stage=='discovery':
        props.update(modality=dict(type='string',enum=keys),behavior=dict(type='string'),evidence=dict(type='string'),uncertainty=dict(type='string'))
        required=list(props);props.update(confidence=dict(type='number',minimum=0,maximum=1),interpretation=dict(type='string'))
    else:
        props.update(type=dict(type='string',enum=keys),description=dict(type='string'),confidence=dict(type='number',minimum=0,maximum=1))
        required=list(props);props.update(count=dict(type='integer',minimum=1),subtype=dict(type='string'))
    schema=dict(type='object',properties=dict(reviewed=support,unknown=support,
        events=dict(type='array',items=dict(type='object',properties=props,required=required,additionalProperties=False)),notes=dict(type='string')),
        required=['reviewed','unknown','events','notes'],additionalProperties=False)
    return dict(type='json_schema',json_schema=dict(name='listener_'+branch+'_'+stage,strict=True,schema=schema))
