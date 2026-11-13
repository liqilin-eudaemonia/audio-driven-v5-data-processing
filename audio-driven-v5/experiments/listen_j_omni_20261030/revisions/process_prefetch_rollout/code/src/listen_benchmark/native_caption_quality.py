"""Conservative output checks; keep raw text, never repair it by cutting sentences."""
from collections import Counter
import re


def generation_failures(text, finish_reason):
    flags=[]
    if finish_reason!='stop':flags.append('generation_not_finished')
    if not text.strip():flags.append('empty_caption')
    # Long verbatim loops, not legitimate repeated short words such as "yeah".
    words=re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
    grams=Counter(tuple(words[i:i+16]) for i in range(len(words)-15))
    sentences=[tuple(re.findall(r"[a-z]+",s.lower())) for s in re.split(r'[.!?]+\s*',text)]
    repeated_sentence=any(n>=3 for s,n in Counter(s for s in sentences if len(s)>=8).items())
    if (grams and max(grams.values())>=4) or repeated_sentence:
        flags.append('repeated_generation')
    return flags
