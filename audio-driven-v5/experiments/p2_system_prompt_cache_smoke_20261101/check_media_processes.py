"""Check that process-isolated decoding preserves all 24 reviewed inputs."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import multiprocessing
from pathlib import Path
import sys
import time

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json
from p2_j_cache.data import read_media


def main():
    old=ROOT/'experiments/p2_system_prompt_cache_smoke_20261101/j24_v2'
    work=ROOT/'experiments/p2_system_prompt_cache_smoke_20261101/process_media_check'
    manifest=read(old/'manifest.json');begin=time.time();rows=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={}
        for sid in manifest['sample_ids']:
            item=read(old/'samples'/(sid+'.json'))
            futures[pool.submit(read_media,item,work)]=item
        for future in as_completed(futures):
            item=futures[future];value=future.result();review=item['review']
            assert hashlib.sha256(value['pixels'].tobytes()).hexdigest()==review['video']['preencode_rgb_sha256']
            for key in ('listener','speaker'):
                assert hashlib.sha256(value[key].astype('<f4',copy=False).tobytes()).hexdigest()==review[key+'_audio']['raw_sha256']
            assert value['audit']['source_frame_pts_s']==review['video']['source_frame_pts_s']
            rows.append(dict(sample_id=item['sample_id'],exact=True));del value
    result=dict(success=True,samples=len(rows),processes=8,pixels_audio_pts_bit_exact=True,elapsed_s=time.time()-begin,results=rows)
    atomic_json(work/'result.json',result)
    print({k:v for k,v in result.items() if k!='results'},flush=True)


if __name__=='__main__':main()
