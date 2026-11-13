import subprocess,sys,json
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');out=root/'data_corpora/seamless_listen_qc3_full'
with (out/'logs/report.log').open('ab',buffering=0) as log:
 p=subprocess.Popen([sys.executable,'-u',str(root/'scripts/report_listen_qc3_bulk.py'),'--watch'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'report_pid.json').write_text(json.dumps(dict(pid=p.pid,host='172.29.15.91'))+'\n');print(p.pid)
