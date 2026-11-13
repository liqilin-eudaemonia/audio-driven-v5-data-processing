"""Read-only checkpoint search in public forks; save bounded evidence."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,json,re,time
base=Path(__file__).resolve().parent/'logs'
s=requests.Session()
def get(url):
 r=s.get(url,timeout=25);r.raise_for_status();return r.json()
def inspect(repo):
 name=repo['full_name'];branch=repo['default_branch'];record={'repo':name,'branch':branch,'pushed_at':repo['pushed_at']}
 try:
  data=get(f'https://api.github.com/repos/{name}/git/trees/{branch}?recursive=1')
  record['tree_truncated']=data.get('truncated');record['candidate_files']=[{'path':x['path'],'size':x.get('size')} for x in data.get('tree',[]) if re.search(r'\.(pth|ckpt|pt|bin|safetensors|onnx|tar|zip)$',x['path'],re.I)]
  r=s.get(f'https://raw.githubusercontent.com/{name}/{branch}/README.md',timeout=20);record['readme_http_status']=r.status_code
  if r.ok:
   record['readme_download_links']=sorted(set(re.findall(r'https?://[^\s<>\)\]"\x27]+',r.text)))
 except Exception as e:record['error']=str(e)
 return record
result={'checked_at':time.time(),'repositories':{}}
for upstream in ['Boese0601/Dyadic-Interaction-Modeling','dc3ea9f/vico_challenge_baseline']:
 try:
  forks=get(f'https://api.github.com/repos/{upstream}/forks?per_page=100')
  print(upstream,'forks',len(forks),flush=True)
  with ThreadPoolExecutor(max_workers=5) as pool:records=list(pool.map(inspect,forks))
  result['repositories'][upstream]=records
  for x in records:print(x['repo'],x.get('candidate_files'),x.get('error'),flush=True)
 except Exception as e:result['repositories'][upstream]={'error':str(e)}
(base/'public_forks_checkpoint_audit.json').write_text(json.dumps(result,indent=2))
