"""Re-score archived DualTalk motion and verify ListenFormer reproduction evidence."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_literature_metrics import evaluate

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def motion(p):
 a=np.load(p)['motion'];assert a.ndim==2 and a.shape[1]==56 and np.isfinite(a).all()
 return np.concatenate([a[:,50:],a[:,:50]],axis=1)
B=ROOT/'experiments/p2_vico_additional_baselines_20261107';D=ROOT/'experiments/p2_compare_vico43_20261106/learning2listen'
files={};g=[];p=[];s=[]
comp=read(B/'dualtalk/COMPLETE.json');assert comp['success'] and comp['outputs']==43
for rec in comp['records']:
 path=Path(rec['path']);assert sha(path)==rec['sha256'];files[str(path)]=sha(path)
 uid=rec['id'];gp=D/'features/gt'/(uid+'.npz');sp=D/'features/speaker'/(uid+'.npz')
 assert sha(sp)==rec['speaker_feature_sha256']
 pp=motion(path);gg=motion(gp);ss=motion(sp);assert pp.shape==gg.shape==ss.shape
 p.append(pp[1:]);g.append(gg[1:]);s.append(ss[:-1])
 for f in [gp,sp]:files[str(f)]=sha(f)
row=evaluate(g,p,s);row['released_flatten_variance']=row['metrics']['Var'].copy()
for key,arrays in [('Var',p),('Var_GT',g)]:
 a=np.concatenate(arrays);row['metrics'][key]={k:float(np.var(a[:,sl],axis=0,ddof=0).mean()) for k,sl in [('exp',slice(6,56)),('pose',slice(0,6))]}
row['units']='DECA transfer proxy: model native exp50+jaw3+neck3 reordered as pose6+exp50; neck/global geometry NOT verified; NOT EMOCA'
row['published_checkpoint_is_vico_table4_checkpoint']=None
row['strict_paper_reproduction']=False
save('DUALTALK_PROXY.json',row)
print('DualTalk proxy rescored',row['metrics'],flush=True)

LF=B/'listenformer'
for name,expected in read(LF/'COMPLETE.json')['files'].items():
 assert sha(name)==expected,name;files[name]=expected
for rec in read(LF/'RENDER_COMPLETE.json')['records']:
 assert sha(rec['path'])==rec['sha256'];files[rec['path']]=rec['sha256']
lfrows=[];old=read(LF/'MOTION_RESULTS.json')['bfm_native']['metrics']['FD']
for rec in read(LF/'GENERATION_COMPLETE.json')['records']:
 assert sha(rec['path'])==rec['sha256'];files[rec['path']]=rec['sha256']
 a=np.load(rec['path'])['motion'];gp=ROOT/'experiments/p2_vico_motion8000_20261106/features/gt'/(rec['id']+'.npz');b=np.load(gp)['motion'];files[str(gp)]=sha(gp)
 assert a.shape==b.shape and a.shape[1]==70
 lfrows.append({'id':rec['id'],'frames':len(a),**{k:float(np.abs(a[:,sl]-b[:,sl]).mean()) for k,sl in [('exp',slice(0,64)),('angle',slice(64,67)),('trans',slice(67,70))]}})
new={k:float(np.mean([r[k] for r in lfrows])) for k in ['exp','angle','trans']}
assert max(abs(new[k]-old[k]) for k in new)<1e-7
image=read(LF/'RESULTS.json')['metrics']
paper={'psnr':18.89,'ssim':.62,'fid':24.52,'csim':.63,'cpbd':.17}
comparison=[]
for key,reported in paper.items():
 actual=image[key];comparison.append({'method':'ListenFormer local retraining','metric':key,'reported':reported,'local':actual,'difference':actual-reported,'relative_difference_percent':100*(actual-reported)/reported,'source':'ListenFormer local PDF Table2 page6','evidence':'sealed image score; base metrics separately rerun this round','same_official_checkpoint':False})
for k,value in {'angle':7.35,'exp':13.36,'trans':5.84}.items():
 actual=new[k]*100;comparison.append({'method':'ListenFormer local retraining','metric':'BFM_L1x100_'+k,'reported':value,'local':actual,'difference':actual-value,'relative_difference_percent':100*(actual-value)/value,'source':'ListenFormer local PDF Table1 page6 test Average','evidence':'new CPU recomputation, clip-equal native driver readout; paper aggregation not confirmed','same_official_checkpoint':False})
save('LISTENFORMER_COMPARISON.json',{'rows':comparison,'native_l1':new,'per_clip':lfrows,'image_source':str(LF/'RESULTS.json'),'complete_epochs':200,'selection':'validation selected epoch1 / 869 steps','official_pretrained_share_status':'cancelled, Baidu errno -21','strict_paper_reproduction':False})
with (OUT/'listenformer_vs_report.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
save('ADDITIONAL_AUDIT.json',{'passed':True,'files':files,'dualtalk_prediction_files':43,'listenformer_video_files':43,'listenformer_prediction_files':43,'native_l1_recompute_max_abs':max(abs(new[k]-old[k]) for k in new),'avatar_forcing_evaluated':False})
print('ListenFormer source hashes and native L1 verified',new,flush=True)
