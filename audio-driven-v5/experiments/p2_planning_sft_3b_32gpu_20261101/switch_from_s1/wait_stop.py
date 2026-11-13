
import hashlib,json,time
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5')
s1=root/'experiments/p2_j_guided_av_event_s1_32gpu_duration512_20261101'
w=root/'experiments/p2_planning_sft_3b_32gpu_20261101/switch_from_s1'
marker=s1/'step_002300.complete.json'
while not marker.exists():
    time.sleep(0.2)
# Marker is written only after checkpoint is fully saved. Request native graceful stop.
(s1/'STOP').write_text('User requested pause after step 2300 checkpoint and switch four nodes to planning LoRA SFT.\n')
record=dict(requested_checkpoint=2300,stop_requested_at=time.time(),checkpoint=str(s1/'step_002300.pt'))
(w/'stop_requested.json').write_text(json.dumps(record,indent=2)+'\n')
h=hashlib.sha256()
with (s1/'step_002300.pt').open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
expected=json.loads(marker.read_text())
assert expected['step']==2300 and h.hexdigest()==expected['sha256']
record.update(checkpoint_sha256=h.hexdigest(),checkpoint_verified=True)
(w/'checkpoint_verified.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
