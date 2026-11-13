from pathlib import Path
import importlib.util,torch,cv2,numpy as np,json
base=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sv_fvd',base/'SpeakerVid-5M-Code/metrics/express/lib/metric/fvd.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
assert torch.cuda.device_count()==1
cap=cv2.VideoCapture(str(base/'deployment/dystream_smoke.mp4'));frames=[]
while len(frames)<32:
 ok,f=cap.read()
 if not ok:break
 frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
cap.release()
video=torch.from_numpy(np.stack(frames)).permute(3,0,1,2).float()/255
model=mod.load_i3d_pretrained(torch.device('cuda'))
feat=mod.get_fvd_feats(video.unsqueeze(0),model,torch.device('cuda'),bs=1)
assert np.isfinite(feat).all()
report={'gpu':torch.cuda.get_device_name(),'shape':list(feat.shape),'finite':True,'note':'Official FVD feature extraction only; no generator is released in this repository.'}
(base/'deployment/logs/speakervid_fvd_verified.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
