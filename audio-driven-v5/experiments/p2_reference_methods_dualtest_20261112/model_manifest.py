from runtime import *
base=ROOT/'experiments/p2_vico_additional_baselines_20261107';models={}
models['avatarforcing']=read(ROOT/'experiments/p2_vico43_unified_20261107/avatarforcing/protocol.json')['assets']
models['dystream']=read(ROOT/'experiments/p2_compare_seamless200_20261106/dystream/protocol.json')['weights']
l2l=read(ROOT/'experiments/p2_compare_vico43_20261106/learning2listen/protocol.json')['models']
for n,r in l2l.items():models['l2l_'+n]=r['files']
for name,files in {
'listenformer':[base/'listenformer/best.pt',base/'features/train_stats.npz',base/'assets/renderer/face/epoch_00190_iteration_000400000_checkpoint.pt'],
'dualtalk':[base/'assets/dualtalk/DualTalk.pth',base/'assets/wav2vec/config.json'],
'feature_extractors':[Path('/public/0_DATA/2_Avatar/ViCo/epoch_20.pth'),ROOT/'compare/deployment/assets/deca_model.tar'],
'gpavatar':[ROOT/'experiments/p2_l2l_public_renderer_20261110/assets/one_model.ckpt']}.items():models[name]={str(p):sha(p) for p in files}
for name,files in models.items():
 for path,h in files.items():assert sha(path)==h,(name,path)
save(WORK/'MODEL_MANIFEST.json',dict(passed=True,models=models,listenformer_source='local full200epoch ViCo training',other_generators='official pretrained',at=time.time()))
print('Verified model assets:',len(models))
