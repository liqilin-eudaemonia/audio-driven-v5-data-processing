import sys,json,pathlib,os,time
root=pathlib.Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from regenerate_j_upright_captions_v3 import prepare_task,WORK,write
rows=[json.loads(x) for x in (WORK/'regenerate_candidates.jsonl').open()]
selected=[next(r for r in rows if not r.get('_restore_orientation')),next(r for r in rows if r['sample_id']=='naturalistic_V03_S0984_I00000046_local_63463d08404f')]
checks=[]
for row in selected:
 t=time.monotonic();r,parent,p,restoration=prepare_task(row);assert p is not None
 checks.append(dict(sample_id=row['sample_id'],restored=bool(restoration),video_frames_sha256=p['evidence']['video_frames_sha256'],audio_encoder_sha256=p['evidence']['audio_encoder_sha256'],nod_events=len(p['nod_events']),reaction_cues=len(p['reaction_cues']),elapsed_s=time.monotonic()-t))
 print(checks[-1],flush=True)
write(WORK/'input_preflight.json',dict(status='passed',checks=checks))
