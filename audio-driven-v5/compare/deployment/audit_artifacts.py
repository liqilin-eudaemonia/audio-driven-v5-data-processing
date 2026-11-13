from pathlib import Path
import json,hashlib,subprocess,time
base=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while c:=f.read(8*1024*1024):h.update(c)
 return h.hexdigest()
weights=[]
for repo in ['DyStream','learning2listen']:
 for pattern in ['**/*.pth','**/*.ckpt','**/*preprocess_core.npz']:
  for p in (base/repo).glob(pattern):weights.append({'path':str(p.relative_to(base)),'bytes':p.stat().st_size,'sha256':sha(p)})
commits={r:subprocess.check_output(['git','-C',str(base/r),'rev-parse','HEAD'],text=True).strip() for r in ['DyStream','learning2listen','Dyadic-Interaction-Modeling','vico_challenge_baseline']}
(base/'deployment/weights_manifest.json').write_text(json.dumps({'created':time.time(),'commits':commits,'weights':weights},indent=2))
print('hashed',len(weights),'weights/statistics files',flush=True)
