"""Wake the agent on a failed recovery, even before retry limits are reached."""
import json

def pending_error(work):
 result=work/'RECOVERY_RESULT.json'
 resumed=json.loads(result.read_text())if result.exists()else{}
 for name in ['NEEDS_AGENT.json','GUARDIAN_ERROR.json','RECOVERY_ERROR.json']:
  path=work/name
  if not path.exists():continue
  error=json.loads(path.read_text())
  if name=='RECOVERY_ERROR.json'and resumed.get('restarted')and resumed.get('at',0)>error.get('at',0):continue
  return dict(file=name,detail=error)
 return None
