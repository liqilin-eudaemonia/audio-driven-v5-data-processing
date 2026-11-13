"""Independent final artifact, frame-grid, source-only and aggregation audit."""
from pathlib import Path
import json,hashlib,subprocess,concurrent.futures,csv,math,time
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'experiments/p2_vico256_combined8000_20261106';ARM='base_event_feature_whisper_8000'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=load(W/'protocol.json');result=load(W/'RESULTS.json');complete=load(W/'COMPLETE.json');items=load(W/'shards/00.json');spec=p['datasets']['vico'];assert complete['protocol_sha256']==sha(W/'protocol.json');assert sha(spec['annotation'])==spec['annotation_sha256'];assert sha(W/'INPUT_PREFLIGHT.json')==p['input_preflight_sha256']
 for f,h in p['execution_code'].items():assert sha(f)==h,f
 with Path(spec['annotation']).open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==len(items)==43 and {r['uuid'] for r in rows}=={x['id'] for x in items}
 def check(x):
  ident=x['id'];g=x['geometry'];media=W/'scoring'/ARM/(ident+'.listener.mp4');receipt=load(W/'conversion'/f'{ident}.verified.json');conv=load(W/'conversion'/f'{ident}.json');meta=load(W/'generated'/ARM/x['key']/'metadata.json');audit=load(W/'conditions'/x['key']/'input_audit.json');plan=load(W/'conditions'/x['key']/'qwen/plan.json')
  assert sha(media)==receipt['score_sha256']==conv['file_sha256']
  assert meta['step']==8000 and meta['seed']==137 and meta['checkpoint_sha256']==p['arms'][0]['original_sha256']
  assert meta['generation']=={'steps':30,'video_cfg':3,'audio_cfg':7} and meta['target_description_access'] is False and meta['listener_future_input'] is False
  assert plan['adapter'] is None and plan['target_description_access'] is False
  assert audit['source_only'] is True and audit['decoded_listener_frames']==1 and audit['listener_future_media_in_model_input'] is False and audit['target_description_in_model_input'] is False
  assert meta['export']['width']==meta['export']['height']==512 and meta['export']['fps']==25 and meta['export']['frames']==g['frames']
  expected=[min(g['frames']-1,math.floor(i*25/30+.5)) for i in range(g['gt_frames'])];assert conv['source_frame_indices']==expected and conv['max_timestamp_error_s']<=.02000001
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames','-of','json',str(media)]))['streams'][0]
  assert (probe['width'],probe['height'])==(256,256) and probe['avg_frame_rate']=='30/1' and int(probe['nb_read_frames'])==g['gt_frames'],(ident,probe)
  return dict(id=ident,decoded_scoring_frames=int(probe['nb_read_frames']),sha256=receipt['score_sha256'],generation_frames=g['frames'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(check,items))
 assert sum(x['decoded_scoring_frames'] for x in verified)==result['frames']==16288
 frames={x['id']:x['decoded_scoring_frames'] for x in verified};cpbd=[load(W/'metrics'/f'cpbd_{i:02d}.json') for i in range(43)];weighted=sum(r['values']['cpbd']*frames[rows[i]['uuid']] for i,r in enumerate(cpbd))/16288;assert abs(weighted-result['metrics']['cpbd'])<1e-12
 for kind in ['base_metrics','fid','csim']:
  receipt=load(W/'metrics'/f'{kind}.json')
  for k,v in receipt['values'].items():assert result['metrics'][k]==v and math.isfinite(v)
 paths=[ROOT/'evaluation/vico_challenge_baseline/evaluations'/x for x in ['compute_base_metrics.py','compute_fid.py','compute_cpbd.py','arcface_torch/compute_csim.py']]
 final=dict(passed=True,at=time.time(),clips=43,decoded_scoring_frames=16288,source_only=True,checkpoint_step=8000,protocol_sha256=sha(W/'protocol.json'),result_sha256=sha(W/'RESULTS.json'),metrics=result['metrics'],official_script_sha256={str(x):sha(x) for x in paths},checks=['exact original test43 membership','all scoring files SHA256','all scoring videos independently decoded by ffprobe','native512/25 and scoring256/30 frame grids','nearest-timestamp mapping <=20ms','same checkpoint8000/seed137/Euler30/CFG3,7','listener first frame only, no GT future/media/description','all official receipt values','CPBD exact frame-weighted aggregation'],verified=verified)
 (W/'FINAL_AUDIT.json').write_text(json.dumps(final,indent=2));print(json.dumps({k:v for k,v in final.items() if k not in ['verified','official_script_sha256','checks']},indent=2))
if __name__=='__main__':main()
