import sys,pathlib,time,json,statistics,torch
R=pathlib.Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from glare_resident_inference import ResidentGLARE
out=pathlib.Path(__file__).parent
ref=R/'experiments/glare_inference_20261111/reference.png'
audio=R/'experiments/glare_inference_20261111/speaker.wav'
torch.set_num_threads(2)
t=time.perf_counter();engine=ResidentGLARE(R/'experiments/glare_paper_settings_node11_gpu7_20261111/best.pt',out/'cache');load=time.perf_counter()-t
first=engine.generate(ref,audio,decode_batch=1);first_info=first[-1];base=first[3];del first
results=[]
for batch in [1,4,8,16,8,4,1]:
 samples=[];maximum=0.;mean=0.
 for _ in range(2):
  image,rgb,padded,raw,info=engine.generate(ref,audio,decode_batch=batch)
  diff=(base-raw).abs();maximum=max(maximum,float(diff.max()));mean=max(mean,float(diff.mean()))
  samples.append(info['generation_seconds'])
  del raw,diff
 row=dict(batch=batch,seconds=samples,max_raw_difference=maximum,mean_raw_difference=mean,cache_hits=info['audio_cache_hit'] and info['reference_cache_hit'])
 results.append(row);print(json.dumps(row),flush=True)
valid=[r for r in results if r['max_raw_difference']<1e-3 and r['mean_raw_difference']<1e-5]
best=min(valid,key=lambda r:statistics.median(r['seconds']))['batch']
exports=[]
for overlap in [False,True]:
 jobs=[dict(reference=str(ref),audio=str(audio),output=str(out/f'overlap{int(overlap)}_{i}.mp4')) for i in range(3)]
 started=time.perf_counter();engine.run_jobs(jobs,best,overlap);elapsed=time.perf_counter()-started
 exports.append(dict(overlap=overlap,seconds=elapsed,jobs=3));print(json.dumps(exports[-1]),flush=True)
result=dict(model_load_seconds=load,first_generate=first_info,decode_results=results,recommended_decode_batch=best,export_results=exports,step=engine.step)
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
