"""Official released Conan inputs/weights/stats; one forward, zero updates."""
import os,sys,json,types,hashlib
from pathlib import Path
import numpy as np,torch
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');REPO=R/'compare/learning2listen/src';O=R/'experiments/l2l_literature_protocol_audit_20261106/original_paper_recheck_20261107'
sys.path.insert(0,str(R/'compare/deployment'))
from l2l_reliable_core import backend,save,sha
backend();os.chdir(REPO);sys.path.insert(0,str(REPO))
from vqgan.vqmodules.gan_models import setup_vq_transformer,calc_vq_loss
from utils.load_utils import load_test_data
cfg=json.loads((REPO/'configs/vq/delta_v6.json').read_text());vcfg=json.loads((REPO/cfg['l_vqconfig']).read_text())
path=REPO/'vqgan/models/l2_32_smoothSS_er2er_best.pth';expected=sha(path)
model,_,_=setup_vq_transformer(types.SimpleNamespace(),vcfg,load_path=str(path),test=True);model.eval()
x,y,a,files,stats=load_test_data(cfg,'er2er',cfg['tag'],0,vqconfigs={'l_vqconfig':vcfg,'s_vqconfig':None},smooth=True,speaker='conan',num_out=32)
with torch.no_grad():
 target=torch.from_numpy(y[:,:32].copy()).cuda();prediction,q=model(target,None);loss=calc_vq_loss(prediction,target,q)
 raw_pred=prediction.cpu().numpy()*stats['body_std_Y']+stats['body_mean_Y'];raw_gt=y[:,:32]*stats['body_std_Y']+stats['body_mean_Y']
 reconstructed,decoded=model.module.sample_step(target)
record=torch.load(path,map_location='cpu',weights_only=False)
result=dict(passed=bool(torch.isfinite(loss)),scope='Released original Conan test, first32 sequences/first32 frames; not ViCo test43; not full paper-table reproduction',
 optimizer_updates=0,checkpoint_sha256=expected,checkpoint_epoch=record.get('epoch'),checkpoint_optimizer_steps=record.get('optimizer',{}).get('n_steps'),
 normalized_reconstruction=float(loss-q.mean()),quantization_loss=float(q.mean()),normalized_total=float(loss),
 
 token_roundtrip_max_error=float((reconstructed-decoded).abs().max()),
 vq_config=vcfg,source_files=[str(REPO/'data/conan/test'/n) for n in ['p0_list_faces_clean_deca.npy','p1_speak_faces_clean_deca.npy','p1_speak_audio_clean_deca.npy','p1_speak_files_clean_deca.npy']])
assert sha(path)==expected
save(O/'OFFICIAL_ANCHOR_CHECK.json',result);print(json.dumps(result))
