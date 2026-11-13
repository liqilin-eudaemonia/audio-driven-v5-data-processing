import sys,json,sqlite3,cProfile,pstats,io,tempfile,concurrent.futures
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(root/'src'))
from listen_richness.prepare_cluster import source_records,database,pack_source,flush
from listen_richness.production import contract
from transformers import AutoTokenizer
w=root/'experiments/listen_richness_full_j_20261031';u=root/'experiments/listen_j_omni_20261030'
with sqlite3.connect(f'file:{w}/inputs.sqlite?mode=ro',uri=True) as db:ids=[r[0] for r in db.execute("SELECT jid FROM sources WHERE status='prepared' ORDER BY jid LIMIT 64")]
with concurrent.futures.ThreadPoolExecutor(8) as pool:rs=list(pool.map(lambda jid:source_records(u,jid),ids))
tok=AutoTokenizer.from_pretrained(contract()['model'],local_files_only=True)
with tempfile.TemporaryDirectory(prefix='richness-pack-') as td:
 db=database(td);pr=cProfile.Profile();pr.enable()
 with db:
  for r in rs:pack_source(db,r,tok,commit=False)
 flush(db,td,512,True)
 pr.disable();buf=io.StringIO();pstats.Stats(pr,stream=buf).sort_stats('cumtime').print_stats(30);print(buf.getvalue());(w/'prepare_pack_profile.txt').write_text(buf.getvalue());db.close()
