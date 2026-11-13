"""Reject only prefixes that cannot extend to a strictly admissible raw reply."""
import math
import re
import regex

NUMBER=r'-?\d+(?:\.\d+)?'
SCORE=r'\d+(?:\.\d+)?'
TRIPLE=rf'{NUMBER}\s*,\s*{NUMBER}\s*,\s*{SCORE}'
# Deliberately a superset: whitespace may include a newline inside a row.
# This can defer a bad reply, but never rejects a legal reply because of whitespace.
PREFIX=regex.compile(rf'\A\s*(?:NULL|{TRIPLE}(?:\s+{TRIPLE})*)?\s*\Z',regex.I)
TIMES=re.compile(rf'^\s*({NUMBER})\s*,\s*({NUMBER})\s*,')


def impossible_prefix(text,duration=2.4):
    if '\ufffd' in text:return None # Incomplete UTF8 byte tokens may decode to replacement characters.
    try:match=PREFIX.fullmatch(text,partial=True,timeout=.005)
    except TimeoutError:return None
    if match is None:return 'cannot_extend_to_numeric_rows_or_exact_NULL'
    previous_end=-math.inf
    for line in text.splitlines():
        m=TIMES.match(line)
        if not m:continue
        a,b=map(float,m.groups())
        if not all(math.isfinite(v) for v in (a,b)) or not 0<=a<b<=duration+1e-6:
            return 'completed_time_fields_out_of_bounds'
        if a<previous_end-1e-8:return 'completed_time_fields_overlap_or_unsorted'
        previous_end=b
    return None


def install(runtime):
    import torch
    from transformers import LogitsProcessor,LogitsProcessorList
    model=runtime.analyzer.model;original=model.generate
    class RejectProcessor(LogitsProcessor):
        def __init__(self,prompt_length):self.prompt_length=prompt_length;self.reasons={};self.prefixes={}
        def __call__(self,input_ids,scores):
            suffix=input_ids[:,self.prompt_length:]
            text=runtime.analyzer.processor.batch_decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            eos=runtime.eos_ids[0]
            for i,raw in enumerate(text):
                reason=self.reasons.get(i) or impossible_prefix(raw)
                if reason:
                    if i not in self.reasons:self.reasons[i]=reason;self.prefixes[i]=raw
                    scores[i,:]=-float('inf');scores[i,eos]=0.
            return scores
    def generate(**kwargs):
        checker=RejectProcessor(kwargs['input_ids'].shape[1])
        previous=list(kwargs.get('logits_processor',[]));kwargs=dict(kwargs,logits_processor=LogitsProcessorList(previous+[checker]))
        result=original(**kwargs)
        for i,row in enumerate(runtime.rows):
            if i in checker.reasons:
                row.update(early_reject=True,early_reject_reason=checker.reasons[i],raw_prefix=checker.prefixes[i],
                           forced_eos=True,full_official_reply_available=False)
        return result
    model.generate=generate
    return lambda:setattr(model,'generate',original)
