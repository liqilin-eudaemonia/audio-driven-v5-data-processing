import sys,pathlib,json,time
root=pathlib.Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from regenerate_j_upright_captions_v4 import prepare_task,WORK,write
rows={r['sample_id']:r for r in map(json.loads,(WORK/'regenerate_candidates.jsonl').open())}
ids=['naturalistic_V03_S0980_I00000392_local_f21299a74717']
# Pick every distinct actual failure from the previous workers, then verify its full original-archive cue path.
import re
members=set()
for p in (WORK/'regeneration_logs').glob('gpu-*.log'):
 for x in re.findall(r"filename '([^']+\.json)' not found",p.read_text(errors='replace')):members.add(pathlib.Path(x).stem)
selected=[r for r in rows.values() if r['listener_id'] in members]
assert selected and len(members)>=2
checks=[]
for row in selected:
 t=time.monotonic();r,parent,prepared,restored=prepare_task(row);assert prepared is not None
 checks.append(dict(sample_id=row['sample_id'],vocal_cues=len(prepared['vocal_cues']),vocal_evidence_equal_to_parent=prepared['vocal_cues']==parent['input_evidence']['vocal_supplement'],audio_sha256=prepared['evidence']['audio_encoder_sha256'],elapsed_s=time.monotonic()-t))
 print(checks[-1],flush=True)
write(WORK/'VOCAL_ORIGINAL_ARCHIVE_PREFLIGHT.json',dict(status='passed',source_members=sorted(members),checks=checks))
