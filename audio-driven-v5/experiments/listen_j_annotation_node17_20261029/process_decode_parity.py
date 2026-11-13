"""Real spawn-from-loader-thread parity check using original source samples."""
from pathlib import Path
import os,sys,json,tempfile,hashlib,time
from concurrent.futures import ThreadPoolExecutor
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path.insert(0,str(R/'src'))

def main():
 from listen_benchmark.process_decode import ProcessMediaDecoder
 from listen_benchmark.visual_native8 import prepare_media
 from listen_benchmark.frame_crop import crop_original_frame
 rows=json.loads((W/'jobs/naturalistic_V03_S0130_I00000372.json').read_text())['rows'][100:116]
 expected=json.loads((W/'monitor_snapshots/decode_executor_probe.json').read_text())['arms'][0]['results'];expected={x['sample_id']:x for x in expected}
 outputs=[];started=time.monotonic()
 with tempfile.TemporaryDirectory(prefix='listen-process-decode-parity-',dir='/dev/shm') as folder:
  decoder=ProcessMediaDecoder(folder,W/'resources',True)
  try:
   with ThreadPoolExecutor(1) as loader:
    result=loader.submit(lambda:list(decoder.map(rows,8,112))).result()
   for row,media,metrics in result:
    assert media is not None,metrics
    signature=hashlib.sha256(json.dumps(media[8]['meta'],sort_keys=True).encode()).hexdigest()
    assert signature==expected[row['sample_id']]['signature']
    assert all(im.size==(512,512) for im in media[8]['images'])
    assert metrics['decoder_cuda_visible_devices']=='' and metrics['decoder_pid']!=os.getpid()
    outputs.append({'sample_id':row['sample_id'],'signature':signature,'metrics':metrics})
   baseline=prepare_media(rows[0]['source'],frame_crop=crop_original_frame)[8]
   assert [im.tobytes() for im in baseline['images']]==[im.tobytes() for im in result[0][1][8]['images']]
  finally:decoder.close()
 report={'at':time.time(),'wall_s':time.monotonic()-started,'samples':len(outputs),'source_and_frame_metadata_equal':True,'first_sample_pixels_equal':True,'spawned_from_background_thread':True,'cache_removed':True,'outputs':outputs}
 (W/'monitor_snapshots/process_decode_parity.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='outputs'}),flush=True)

if __name__=='__main__':main()
