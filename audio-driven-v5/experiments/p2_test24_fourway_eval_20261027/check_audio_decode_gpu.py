import sys,json,torch,soundfile as sf,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.config import extractor_config
from p2_listen.contracts import read,atomic_json
c=read(OUT/'experiments.json')['E0']['config'];cfg=extractor_config(c)
from ltx_trainer.model_loader import load_audio_vae_decoder,load_vocoder
row=max(read(OUT/'training_audio_supervision_audit.json')['E0']['rows'],key=lambda x:x['target_rms'])
z=torch.load(Path(row['cache'])/'payload.pt',map_location='cpu',weights_only=True)['av']['audio_latents']['latents'];device=sys.argv[1] if len(sys.argv)>1 else 'cpu';torch.set_num_threads(4)
result=dict(sample=row,device=device,tests={})
for dtype in [torch.float32,torch.bfloat16]:
 name=str(dtype)
 try:
  with torch.no_grad():
   decoder=load_audio_vae_decoder(cfg.ltx_path,device,dtype).requires_grad_(False).eval();mel=decoder(z.to(device,dtype));del decoder
   vocoder=load_vocoder(cfg.ltx_path,device,dtype).requires_grad_(False).eval();wave=vocoder(mel).squeeze(0).float().cpu();sr=int(vocoder.output_sampling_rate);kind=type(vocoder).__name__;del vocoder
  sf.write(OUT/f'target_reconstructed_gpu_{name}.wav',wave.T.numpy(),sr,subtype='FLOAT');result['tests'][name]=dict(rms=float(wave.square().mean().sqrt()),peak=float(wave.abs().max()),vocoder=kind,mel_min=float(mel.min()),mel_max=float(mel.max()),wave_samples=wave.shape[-1]);del mel
 except Exception as e:result['tests'][name]=dict(error=repr(e))
 atomic_json(OUT/'audio_decode_gpu_audit.json',result);print(name,result['tests'][name],flush=True)
