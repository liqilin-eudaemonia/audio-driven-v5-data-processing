"""Staged trial inference: source-only planning/H, Gemma, Whisper, LTX, then decoding."""
from pathlib import Path
import io,json,math,os,subprocess,sys
import numpy as np
import torch
from p2_listen.contracts import ROOT,read,sha,atomic_json
from p2_listen.cache import atomic_torch,cpu,tree_to


def folder(c,step):return Path(c['output'])/'evaluation'/f'step_{step:06d}'


def rows(c):
 if sha(c['validation_manifest'])!=c['validation_manifest_sha256']:raise ValueError('Validation changed')
 values=read(c['validation_manifest'])
 if len(values)!=64 or len({v['sample_id']for v in values})!=64 or any(v['split']!='dev'for v in values):raise ValueError('Invalid frozen validation')
 return values


def source_inputs(row,out):
 """Only the returned image/audio/duration are passed to the planner."""
 import soundfile as sf
 from scipy.signal import resample_poly
 from j_upright_response_sft_data import load_example
 example=load_example(row);inputs=example['inputs'];wave=np.asarray(inputs['speaker_audio'],dtype=np.float32)
 sr=inputs['speaker_sample_rate']
 if sr!=16000:
  g=math.gcd(sr,16000);wave=resample_poly(wave,16000//g,sr//g).astype(np.float32)
 count=round(row['duration_s']*16000)
 if not count<=len(wave)<=count+1:raise ValueError('Validation speaker time mismatch')
 wave=wave[:count];out.mkdir(parents=True,exist_ok=True)
 inputs['listener_first_frame'].save(out/'first_frame.png')
 sf.write(out/'speaker.wav',wave,16000,subtype='FLOAT')
 atomic_json(out/'source_audit.json',dict(example['audit'],speaker_sha256=sha(out/'speaker.wav'),first_frame_sha256=sha(out/'first_frame.png'),target_description_in_model_input=False))
 return dict(sample_id=row['sample_id'],image=inputs['listener_first_frame'],audio=wave,duration=row['duration_s'])


def observation(row,out):
 return dict(sample_id=row['sample_id'],speaker_audio=str(out/'speaker.wav'),listener_image=str(out/'first_frame.png'),
  requested_duration_s=row['duration_s'],geometry=dict(fps=25,width=512,height=512))


def load_state(c,step):
 from .checkpoint import checked
 from .config import source_identity
 saved=checked(Path(c['output'])/f'step_{step:06d}.pt')
 if saved['identity']['experiment']!=c or saved['identity']['source']!=source_identity()or saved['step']!=step:raise ValueError('Evaluation checkpoint identity mismatch')
 return saved


def qwen(c,step,device):
 from .qwen import load_online
 from .data import data_config
 from p2_listen.config import extractor_config
 from p2_planning_sft.data import PlanningCollator
 from p2_plan_cache.encoding import extract_planned_input
 from p2_j_cache.data import caption_prompt
 out=folder(c,step);cfg=extractor_config(data_config(c['generator']))
 model,peft=load_online(cfg,Path(c['assets']['adapter_config']['path']).parent,device)
 saved=load_state(c,step)
 if model.parameter_names!=saved['identity']['qwen_parameter_names']:raise ValueError('Adapter parameter order changed')
 state={k.removeprefix('qwen.'):v for k,v in saved['model'].items()if k.startswith('qwen.')}
 model.load_state_dict(state,strict=True);del state,saved
 model.eval().requires_grad_(False);model.runtime.thinker.eval()
 collator=PlanningCollator(cfg.omni_path,max_length=32768,max_pixels=c['evaluation']['planner_max_pixels']);collator.processor=model.processor
 eos=model.processor.tokenizer.convert_tokens_to_ids('<|im_end|>');records=[]
 for row in rows(c):
  dest=out/'conditions'/row['sample_id'];raw=source_inputs(row,dest)
  batch=collator.source_inputs([raw]);batch=tree_to(dict(batch),device)
  batch={k:v.to(dtype=torch.bfloat16)if torch.is_tensor(v)and v.is_floating_point()else v for k,v in batch.items()}
  model.runtime.thinker.rope_deltas=None
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   generated=model.runtime.thinker.generate(**batch,max_new_tokens=c['evaluation']['max_new_tokens'],do_sample=False,use_cache=True,
    use_audio_in_video=False,eos_token_id=eos,pad_token_id=model.processor.tokenizer.pad_token_id)
  ids=generated[0,batch['input_ids'].shape[1]:].tolist();valid=eos in ids
  tail=ids[:ids.index(eos)]if valid else ids
  valid=valid and bool(tail)and not(set(tail)&set(model.processor.tokenizer.all_special_ids))
  text=model.processor.tokenizer.decode(tail,skip_special_tokens=False).strip()
  valid=bool(valid and text and '<|'not in text and '|>'not in text)
  atomic_json(dest/'plan.json',dict(text=text,token_ids=ids,valid=valid,target_description_access=False,checkpoint_step=step))
  # Validation reference text is used only in this separate teacher-forced CE.
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):total,tokens=model.planning_loss(batch,row['target']['text'])
  records.append(dict(sample_id=row['sample_id'],planning_sum=float(total),planning_tokens=tokens,format_error=not valid))
  if valid:
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    payload=extract_planned_input(data_config(c['generator']),observation(row,dest),caption_prompt(text),model.runtime,model.processor,32768)
   atomic_torch(dest/'hidden.pt',payload)
 atomic_json(out/'planning_metrics.json',records)
 if any(r['format_error']for r in records):raise ValueError('Invalid/truncated plans retained; evaluation incomplete, no silent repair')


def gemma(c,step,device):
 from omni_av.ltx_runtime import NativeTextConditioner
 from p2_listen.config import extractor_config
 from .data import data_config
 from p2_plan_cache.encoding import encode_text
 from p3_listen.prompts import FixedPrompt
 cfg=extractor_config(data_config(c['generator']));encoder=NativeTextConditioner(cfg,device)
 out=folder(c,step)
 try:
  with torch.inference_mode():atomic_torch(out/'negative.pt',cpu(encoder.encode(FixedPrompt(''))))
  for row in rows(c):
   dest=out/'conditions'/row['sample_id'];payload=torch.load(dest/'hidden.pt',map_location='cpu',weights_only=True)
   with torch.inference_mode():value,_=encode_text(encoder,payload['prompt'],8192)
   atomic_torch(dest/'text.pt',cpu(value))
 finally:encoder.close()


def whisper(c,step,device):
 if not c['generator']['components']['whisper']:return
 import soundfile as sf
 from dataclasses import asdict
 from p2_whisper_av.features import FrozenWhisper
 encoder=FrozenWhisper(c['generator']['whisper']['model_path'],device)
 for row in rows(c):
  dest=folder(c,step)/'conditions'/row['sample_id'];wave,sr=sf.read(dest/'speaker.wav',dtype='float32')
  if sr!=16000:raise ValueError('Whisper/Qwen speaker mismatch')
  feature=encoder.encode_batch([wave],batch_size=1)[0];feature.validate()
  atomic_torch(dest/'whisper.pt',cpu(asdict(feature)))


def render(c,step,device):
 from p2_listen.config import extractor_config
 from .data import data_config
 from .generator import Model
 from omni_av_p2.ltx import load_renderer,unpack_latents
 from omni_av_p2.inference import reference_latents
 from omni_av_p2.connector import ConditionInputs
 from omni_av.ltx_runtime import with_text
 from p2_listen.features import make_observation
 from p2_listen.layout import geometry
 from p2_current_component_ablation.inference import sample
 from p2_testsets_eval.runtime import cached_loaders
 from p2_whisper_av.features import SpeakerFeatures
 cfg=extractor_config(data_config(c['generator']));renderer,strategy,sampler=load_renderer(cfg,device)
 gen=Model(c['generator'],cfg,renderer).to(device);saved=load_state(c,step)
 gen.load_state_dict({k.removeprefix('generator.'):v for k,v in saved['model'].items()if k.startswith('generator.')},strict=True);del saved
 gen.eval().requires_grad_(False);renderer.eval().requires_grad_(False);renderer.set_gradient_checkpointing(False)
 out=folder(c,step);negative=tree_to(torch.load(out/'negative.pt',map_location='cpu',weights_only=True),device)
 with cached_loaders(['load_video_vae_encoder']),torch.inference_mode():
  for row in rows(c):
   dest=out/'conditions'/row['sample_id'];obs=observation(row,dest)
   payload=torch.load(dest/'hidden.pt',map_location='cpu',weights_only=True)
   inputs=ConditionInputs(**tree_to(payload['condition'],device));text=tree_to(torch.load(dest/'text.pt',map_location='cpu',weights_only=True),device)
   if payload['prompt']!=text['prompt']:raise ValueError('Generated plan/Gemma mismatch')
   with torch.autocast('cuda',dtype=torch.bfloat16):u=gen.connector(inputs)
   speaker=SpeakerFeatures(**tree_to(torch.load(dest/'whisper.pt',map_location='cpu',weights_only=True),device))if c['generator']['components']['whisper']else None
   av=reference_latents(cfg,make_observation(obs),geometry(obs),device)
   torch.manual_seed(c['evaluation']['seed']);prepared=strategy.prepare_training_inputs(with_text(av,text),sampler)
   torch.manual_seed(c['evaluation']['seed'])
   sampled=sample(gen,prepared,u,speaker,negative,**c['evaluation']['sampling'])
   video,audio=unpack_latents(sampled,av);atomic_torch(out/'latents'/row['sample_id']/'payload.pt',cpu(dict(video=video,audio=audio)))


def decode(c,step,device):
 import soundfile as sf
 from omni_av_p3_staged.media import decode_and_save
 from .data import data_config
 from p2_listen.config import extractor_config
 from p2_testsets_eval.runtime import cached_loaders
 cfg=extractor_config(data_config(c['generator']));out=folder(c,step)
 with cached_loaders(['load_video_vae_decoder','load_audio_vae_decoder','load_vocoder']),torch.inference_mode():
  for row in rows(c):
   sid=row['sample_id'];dest=out/'generated'/sid;dest.mkdir(parents=True,exist_ok=True)
   z=torch.load(out/'latents'/sid/'payload.pt',map_location='cpu',weights_only=True)
   decode_and_save(cfg,z['video'],z['audio'],25,dest/'padded.mp4',device,raw_audio_path=dest/'raw_audio.pt')
   subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads','2','-i',str(dest/'padded.mp4'),'-map','0:v:0','-map','0:a:0',
    '-frames:v',str(math.ceil(row['duration_s']*25-1e-7)),'-af',f"atrim=duration={row['duration_s']:.9f}",'-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-c:a','aac',str(dest/'generated.mp4')],check=True)
   raw=torch.load(dest/'raw_audio.pt',map_location='cpu',weights_only=True);wave=raw['wave'];count=round(row['duration_s']*raw['rate'])
   if wave.shape[-1]<count:wave=torch.nn.functional.pad(wave,(0,count-wave.shape[-1]))
   sf.write(dest/'listener.wav',wave[...,:count].T.numpy(),raw['rate'],subtype='FLOAT')


def references(c,step,device):
 """GT is decoded only after generated media; never fed to inference stages."""
 import soundfile as sf
 from j_upright_response_sft_data import resolve
 from p2_j_cache.data import decode_frames
 from listen_benchmark.j_omni import read_audio
 sys.path.insert(0,str(ROOT.parent/'OmniResponse'))
 from evaluation.listener_metrics.media import write_video
 out=folder(c,step);pairs=[]
 for row in rows(c):
  sid=row['sample_id'];dest=out/'references'/sid;dest.mkdir(parents=True,exist_ok=True)
  item=resolve(row);frames,pts=decode_frames(item['source']);count=math.ceil(row['duration_s']*25-1e-7)
  if len(frames)<count:raise ValueError('Incomplete GT video')
  write_video(frames[:count],dest/'gt.mkv')
  caption=read(row['target']['source_file']);wave,sr,audit=read_audio(caption['audio_recipe'])
  if audit['processed_waveform_sha256']!=caption['input_evidence']['audio_identity']['processed_waveform_sha256']:raise ValueError('GT clean audio changed')
  sf.write(dest/'gt.wav',wave,sr,subtype='FLOAT');generated=out/'generated'/sid
  pairs.append(dict(id=sid,dataset='stage2_dev',identity='listener',generated=str(generated/'generated.mp4'),generated_audio=str(generated/'listener.wav'),
   gt=str(dest/'gt.mkv'),gt_audio=str(dest/'gt.wav'),pair_duration_s=row['duration_s'],duration_s=row['duration_s'],listening_intervals=[[0.,row['duration_s']]]))
 (out/'pairs.jsonl').write_text(''.join(json.dumps(v)+'\n'for v in pairs))


PHASES={'qwen':qwen,'gemma':gemma,'whisper':whisper,'render':render,'decode':decode,'references':references}
