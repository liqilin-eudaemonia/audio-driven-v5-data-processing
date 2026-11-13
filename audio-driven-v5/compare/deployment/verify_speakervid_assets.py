from pathlib import Path
import hashlib,json,time
base=Path(__file__).resolve().parents[1];tree=json.loads((base/'deployment/logs/speakervid_model_tree.json').read_text());records=[]
for item in tree:
 if item['type']!='file' or not item['path'].startswith(('assets/','ckpt/','weights/')):continue
 p=base/'SpeakerVid-5M-Code'/item['path'];r={'path':item['path'],'expected_bytes':item['size'],'exists':p.exists()}
 if p.exists():
  r['bytes']=p.stat().st_size
  if 'lfs' in item:
   h=hashlib.sha256()
   with p.open('rb') as f:
    while c:=f.read(8*1024*1024):h.update(c)
   r['sha256']=h.hexdigest();r['expected_sha256']=item['lfs']['oid'];r['ok']=r['bytes']==r['expected_bytes'] and r['sha256']==r['expected_sha256']
  else:r['ok']=r['bytes']==r['expected_bytes']
 else:r['ok']=False
 records.append(r)
result={'checked':time.time(),'all_valid':all(r['ok'] for r in records),'files':records}
(base/'deployment/logs/speakervid_assets_verified.json').write_text(json.dumps(result,indent=2));print('verified',sum(r['ok'] for r in records),'of',len(records),flush=True)
