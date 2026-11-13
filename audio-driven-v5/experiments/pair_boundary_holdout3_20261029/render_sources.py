import ctypes
# Avoid host transparent-hugepage allocation stalls; numerical code is unchanged.
assert ctypes.CDLL(None,use_errno=True).prctl(41,1,0,0,0)==0
import json,sys
from pathlib import Path
from types import SimpleNamespace
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from review_listen_full_source import render_source
from p3_listen.common import sha
PREVIEW=ROOT/'data_previews/seamless_listen_full_source_recovery'
def read(p):return json.loads(p.read_text())
for item in read(EXP/'selection.json')['items']:
 pair=item['pair'];d=read(EXP/(pair+'.json'))
 norm=read(ROOT/'data_corpora/seamless_listen_local_full/normalized_pairs'/f'{pair}.json')
 assert all(sha(Path(t['path']))==t['sha256'] for t in norm['tracks'])
 ctx=SimpleNamespace(limit=d['limit'],sources=d['sources'],geometry=[dict(rotation_degrees=0)]*2)
 panels,media=render_source(ctx,norm,PREVIEW/pair)
 (EXP/(pair+'.media.json')).write_text(json.dumps(dict(panels=panels,media=media),indent=2)+'\n')
 print('RENDERED',pair,flush=True)
