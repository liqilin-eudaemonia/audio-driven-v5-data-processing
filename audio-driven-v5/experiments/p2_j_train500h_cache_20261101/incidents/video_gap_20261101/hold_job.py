import fcntl,time,json,os
from pathlib import Path
p=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_j_train500h_cache_20261101/locks/features/006276.lock')
f=p.open("a")
fcntl.flock(f,fcntl.LOCK_EX)
Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_j_train500h_cache_20261101/incidents/video_gap_20261101/held.json').write_text(json.dumps({"pid":os.getpid(),"job":"006276","sample_id":'naturalistic_V03_S1696_I00000194_local_c7e89057dc23',"temporary_hold":True,"time":time.time()}))
while True:time.sleep(30)
