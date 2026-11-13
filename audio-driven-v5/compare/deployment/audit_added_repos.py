from pathlib import Path
import requests,json,time
from concurrent.futures import ThreadPoolExecutor
base=Path(__file__).resolve().parent/'logs'
repos=['hzwer/MM2022-ViCoPerceptualHeadGeneration','awakening-ai/OmniResponse','lingjivoo/ReactFace','Dorniwang/SpeakerVid-5M-Code']
def run(repo):
 out={'repo':repo,'checked':time.time()}
 for endpoint in ['releases','issues?state=all&per_page=100','branches']:
  r=requests.get(f'https://api.github.com/repos/{repo}/{endpoint}',timeout=25);out[endpoint]=r.json();out[endpoint+'_http']=r.status_code
  if endpoint.startswith('issues') and isinstance(r.json(),list):
   for issue in r.json():
    if any(s in ((issue.get('title') or '')+' '+(issue.get('body') or '')).lower() for s in ['weight','checkpoint','pretrain','model','inference']):
     cr=requests.get(issue['comments_url'],timeout=20);issue['download_audit_comments']=cr.json()
 (base/(repo.split('/')[-1]+'_public_audit.json')).write_text(json.dumps(out,indent=2))
 print(repo,'releases',len(out['releases']) if isinstance(out['releases'],list) else out['releases'],flush=True)
 for issue in out.get('issues?state=all&per_page=100',[]):
  if isinstance(issue,dict) and 'download_audit_comments' in issue:print(issue['html_url'],issue['title'],str(issue.get('body'))[:1000],[(c.get('user',{}).get('login'),c.get('body')) for c in issue['download_audit_comments']] if isinstance(issue['download_audit_comments'],list) else '',flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,repos))
for term in ['ReactFace','OmniResponse','SpeakerVid','VidChat']:
 r=requests.get('https://huggingface.co/api/models',params={'search':term,'limit':50},timeout=25)
 print('HF',term,[x['id'] for x in r.json()],flush=True)
 (base/(term+'_HF_model_search.json')).write_text(r.text)
