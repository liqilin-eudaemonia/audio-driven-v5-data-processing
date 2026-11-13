"""Lexical census and reproducible caption sample. Does not create training labels."""
import sqlite3,json,re,random,hashlib
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
DB=ROOT/'experiments/p2_ltx_event_current_sources_v2_20261106/sources.sqlite'
OUT=Path(__file__).resolve().parent
PATTERNS={
 'explicit_order_word':r'\b(?:then|afterwards?|subsequently|followed by|before|after|following)\b',
 'cooccurrence_word':r'\b(?:while|simultaneously|at the same time|concurrently)\b',
 'phase_word':r'\b(?:initially|at first|at the beginning|towards? the end|near the end|later|eventually|finally)\b',
 'persistent_state_word':r'\b(?:throughout|maintains?|remains?|continues?|holding|sustained)\b',
 'repeated_action_word':r'\b(?:repeatedly|several times|multiple times|a few times|occasionally|intermittently|twice|again|series of)\b',
 'numeric_time':r'\b\d+(?:\.\d+)?\s*[- ]?(?:seconds?|s)\b|\b\d{1,2}:\d{2}\b',
}
REACTIONS={
 'nod':r'\bnod(?:s|ding|ded)?\b',
 'shake':r'\b(?:head[- ]shak\w*|shak\w* (?:\w+ ){0,3}head|side[- ]to[- ]side)\b',
 'smile':r'\bsmil\w*\b',
 'laugh':r'\b(?:laugh\w*|chuckl\w*|giggl\w*)\b',
 'brow':r'\b(?:brows?|eyebrows?|frown\w*|furrow\w*|surpris\w*)\b',
 'voice':r'\b(?:says?|said|speaks?|speaking|utters?|utterance|vocaliz\w*|speech|humming)\b',
}

def chunk(bounds):
 db=sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True);c=Counter()
 for sid,caption in db.execute("SELECT sid,json_extract(value,'$.caption') FROM samples WHERE ordinal>=? AND ordinal<?",bounds):
  c['samples']+=1
  flags={k:bool(re.search(p,caption,re.I)) for k,p in PATTERNS.items()}
  c.update(k for k,v in flags.items() if v)
  if flags['explicit_order_word']:
   c['order_and_cooccurrence']+=flags['cooccurrence_word']
   c['order_and_repetition']+=flags['repeated_action_word']
   # Mentions include negations and uncertain descriptions. This is NOT an event count.
   c['order_and_two_or_more_reaction_word_families']+=sum(bool(re.search(p,caption,re.I)) for p in REACTIONS.values())>=2
 db.close();return c

def main():
 db=sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True);n=db.execute('SELECT count(*) FROM samples').fetchone()[0]
 counts=Counter()
 with ProcessPoolExecutor(max_workers=8) as pool:
  for c in pool.map(chunk,[(i,min(i+8192,n)) for i in range(0,n,8192)]):counts.update(c)
 sample=[]
 for ordinal in random.Random(20261106).sample(range(n),12):
  sid,value=db.execute('SELECT sid,value FROM samples WHERE ordinal=?',(ordinal,)).fetchone();r=json.loads(value)
  cap=json.loads(Path(r['statistics']['caption_file']).read_text())
  if cap['caption']!=r['caption']:raise ValueError('Frozen caption mismatch')
  e=cap.get('input_evidence',{})
  sample.append(dict(ordinal=ordinal,sample_id=sid,duration_s=r['source']['duration_s'],caption=r['caption'],caption_file=r['statistics']['caption_file'],
   nod=e.get('nod_supplement'),reaction=e.get('reaction_supplement'),vocal=e.get('vocal_supplement')))
 result=dict(complete=True,scope='Current frozen 284218-sample train+dev 500h',census=dict(counts),percentages={k:round(100*v/n,3) for k,v in counts.items()},
  caveat='Lexical indicators only: matches may describe static states, negated events, quoted speech or repeated instances. These are NOT counts of usable order labels.',patterns=PATTERNS,reaction_word_families=REACTIONS,random_seed=20261106,sample=sample)
 (OUT/'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('sample','patterns','reaction_word_families')},ensure_ascii=False))
 for s in sample:
  if s['ordinal'] in (6084,59338,170430,46316):
   print(json.dumps(dict(ordinal=s['ordinal'],sample_id=s['sample_id'],nod=(s['nod'] or {}).get('events'),reaction=(s['reaction'] or {}).get('cues'),vocal=s['vocal']),ensure_ascii=False))

if __name__=='__main__':main()
