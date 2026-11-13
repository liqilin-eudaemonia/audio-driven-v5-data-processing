import sys,time,sqlite3,json
from pathlib import Path
r=Path("/private/liqi_workspace/project_code/audio-driven-v5");sys.path.insert(0,str(r/'src'))
from listen_benchmark.j_annotation import acquire,write
from listen_richness.prepare_cluster import database
w=r/'experiments/listen_richness_full_j_20261031';lock=acquire(w/'prepare.lock',True)
db=sqlite3.connect(w/'inputs.sqlite',timeout=120)
query='SELECT status,COUNT(*) FROM coverage GROUP BY status'
t=time.monotonic();before=dict(db.execute(query));old_seconds=time.monotonic()-t;old_plan=list(db.execute('EXPLAIN QUERY PLAN '+query));db.close()
t=time.monotonic();db=database(w);index_seconds=time.monotonic()-t
t=time.monotonic();after=dict(db.execute(query));new_seconds=time.monotonic()-t;new_plan=list(db.execute('EXPLAIN QUERY PLAN '+query));assert before==after
write(w/'goal_monitor_20261031/input_index_validation.json',dict(at=time.time(),before=before,after=after,before_seconds=old_seconds,after_seconds=new_seconds,index_build_seconds=index_seconds,before_plan=old_plan,after_plan=new_plan,passed=True))
db.close();lock.close()
