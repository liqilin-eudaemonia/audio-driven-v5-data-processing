"""Partition only failed shards using the native first-request length predicate."""
import json
from functools import lru_cache
from listen_benchmark.semantic_prompt import PLAN_PROMPT,source_payload


@lru_cache(maxsize=1)
def tokenizer(model):
 from transformers import AutoTokenizer
 return AutoTokenizer.from_pretrained(model,local_files_only=True)


def initial_overflow(job,model):
 tok=tokenizer(model);source=source_payload(job['record']);prompt=PLAN_PROMPT+json.dumps(source,ensure_ascii=False)
 try:text=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 except TypeError:text=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
 budget=max(8192,min(24000,1024+220*len(source['visual_sources'])))
 return len(tok.encode(text))+budget>32768


def groups(jobs,model):
 long=[];safe=[]
 for job in jobs:(long if initial_overflow(job,model) else safe).append(job)
 if long:
  # Every isolated source STILL runs through the original native worker;
  # this CPU test does not publish labels or reject any training member.
  return ([safe] if safe else [])+[[job] for job in long]
 middle=len(jobs)//2
 return [jobs[:middle],jobs[middle:]]
