"""Real adjacent-pair equality and paired input-read timing, bounded to3groups."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import prepare_glare_nonprosody_cache as base
from glare_shared_decode import decode_group
from p2_j_cache.data import decode_frames

WORK=Path(__file__).resolve().parent
p=base.load_protocol(base.DEFAULT)
db=sqlite3.connect(base.DEFAULT/'jobs.sqlite');db.row_factory=sqlite3.Row
jobs={r['ordinal']:dict(r) for r in db.execute('SELECT * FROM jobs')};db.close()
db=sqlite3.connect(WORK/'groups.sqlite')
selected=[json.loads(r[0]) for r in db.execute('SELECT ordinals FROM groups WHERE pairs=2 LIMIT 3')];db.close()
results=[]
for ordinals in selected:
    rows=[base.load_row(jobs[o],p) for o in ordinals]
    with tempfile.TemporaryDirectory(dir='/dev/shm',prefix='glare-shared-verify-') as tmp:
        tick=time.time();group=decode_group(rows,tmp);shared_seconds=time.time()-tick
        for row,item in zip(rows,group['items']):
            tick=time.time();reference,pts=decode_frames(row['source']);single=time.time()-tick
            actual=np.load(Path(item['folder'])/'lia_rgb.npy')
            assert np.array_equal(reference,actual),row['sample_id']
            assert pts==item['pts']
            value=dict(sample_id=row['sample_id'],pixel_exact=True,pts_exact=True,
                       frames=len(reference),single_decode_s=single,shared_group_s=shared_seconds)
            results.append(value);print(json.dumps(value),flush=True)
base.save(WORK/'SHARED_DECODE_ACCEPTED.json',dict(passed=True,real_groups=3,real_pairs=6,
    original_opens=3,instead_of=6,samples=results,
    shared_decode_code_sha256=base.sha(ROOT/'scripts/glare_shared_decode.py')))
