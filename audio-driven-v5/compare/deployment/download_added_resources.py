from pathlib import Path
import gdown,json,time,tarfile
base=Path(__file__).resolve().parents[1];report=[]
tasks=[('ReactFace PIRender','https://drive.google.com/drive/folders/1Ys1u0jxVBxrmQZrcrQbm8tagOPNxrTUA',base/'ReactFace/external/PIRender'),('ReactFace references','https://drive.google.com/drive/folders/1uVOOJzY3p2XjDESwH4FCjGO8epO7miK4',base/'ReactFace/external/FaceVerse')]
for name,url,dest in tasks:
 dest.mkdir(parents=True,exist_ok=True)
 try:
  files=gdown.download_folder(url,output=str(dest)+'/',quiet=False)
  report.append({'resource':name,'url':url,'files':files,'status':'downloaded' if files else 'failed'})
 except Exception as e:report.append({'resource':name,'url':url,'error':str(e),'status':'failed'});print(name,str(e),flush=True)
p=base/'MM2022-ViCoPerceptualHeadGeneration/u2net_human_seg.pth'
try:
 if not p.exists():gdown.download(id='1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P',output=str(p),quiet=False)
 report.append({'resource':'MM2022 U2Net','file':str(p),'status':'downloaded' if p.exists() else 'failed'})
except Exception as e:report.append({'resource':'MM2022 U2Net','status':'failed','error':str(e)})
(base/'deployment/logs/added_resource_downloads.json').write_text(json.dumps(report,indent=2))
