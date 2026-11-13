"""Read-only official source snapshots, two concurrent HTTP requests at most."""
import concurrent.futures,datetime,hashlib,json,urllib.request,urllib.error
from pathlib import Path
BASE=Path(__file__).resolve().parent/'web'
REPOS=['lzk901372/glare','lingjivoo/ReactFace','liushenme/ListenFormer','nuo1wang/DiffListener','Boese0601/Dyadic-Interaction-Modeling','grisoon/INFP','evonneng/learning2listen','dc3ea9f/vico_challenge_baseline']
URLS={r.replace('/','__')+'__'+kind:'https://api.github.com/repos/'+r+suffix for r in REPOS for kind,suffix in [('repo',''),('readme','/readme'),('issues','/issues?state=all&per_page=100')]}
URLS.update({'ReactFace_author_pdf':'https://wcxie.github.io/Weicheng-Xie/pdf/TVCG-2024-Luo.pdf','DiTaiListener_project':'https://havent-invented.github.io/DiTaiListener/','ARIG_project':'https://jinyugy21.github.io/ARIG/'})
def fetch(item):
 name,url=item;req=urllib.request.Request(url,headers={'User-Agent':'research-protocol-audit','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
 try:
  with urllib.request.urlopen(req,timeout=40) as resp: body=resp.read();status=resp.status;headers=dict(resp.headers)
 except urllib.error.HTTPError as e:body=e.read();status=e.code;headers=dict(e.headers)
 except Exception as e:return {'name':name,'url':url,'error':str(e),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 ext='.pdf' if name.endswith('_pdf') else '.json' if 'api.github.com' in url else '.html';p=BASE/(name+ext);p.write_bytes(body)
 return {'name':name,'url':url,'status':status,'file':str(p),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'link_header':headers.get('Link'),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if __name__=='__main__':
 BASE.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(fetch,URLS.items()))
 (BASE/'INDEX.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
 for r in results:print(r['name'],r.get('status',r.get('error')),r.get('bytes'))
