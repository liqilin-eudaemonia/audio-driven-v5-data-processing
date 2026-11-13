#!/usr/bin/env python3
"""Whole-clip parity and positive GLARE regression; CPU-only, no production write."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import hashlib
from pathlib import Path
import sys
import time

WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))


def main():
    os.nice(10)
    from listen_benchmark.visual_native8 import prepare_media
    from listen_benchmark.frame_crop import crop_original_frame
    from listen_benchmark.j_annotation import read,write,result_path
    from listen_j_annotation_nod import annotate
    from run_glare_nodding_pilot import module,FACE,NOD,MODEL,PROFILES,intervals
    cases=[]
    for pair in ['naturalistic_V03_S1487_I00000105','naturalistic_V02_S5078_I00000281']:
        row=read(WORK/'jobs'/f'{pair}.json')['rows'][0]
        tick=time.monotonic();old=prepare_media(row['source']);old_s=time.monotonic()-tick
        tick=time.monotonic();new=prepare_media(row['source'],frame_crop=crop_original_frame);new_s=time.monotonic()-tick
        assert old[8]['duration_s']==new[8]['duration_s'] and old[8]['meta']==new[8]['meta']
        assert len(old[8]['images'])==len(new[8]['images'])
        assert all(a.tobytes()==b.tobytes() for a,b in zip(old[8]['images'],new[8]['images']))
        cases.append({'sample_id':row['sample_id'],'frames':len(old[8]['images']),
                      'source_size':row['source']['video_transform']['source_size'],
                      'old_decode_s':old_s,'new_decode_s':new_s,'all_pixels_pts_jpeg_metadata_exact':True})
    row=read(WORK/'jobs/naturalistic_V03_S1487_I00000105.json')['rows'][0]
    old=read(result_path(WORK,'nod',row['sample_id']))
    new=annotate(row,module('face_fast_crop',FACE),module('nod_fast_crop',NOD),PROFILES['recall'],
                 MODEL,intervals,WORK,fast_crop=True)
    keys=['status','events','scores','pose','source','frames','parameters','model_sha256','media']
    for key in keys:assert old[key]==new[key],key
    write(WORK/'fast_crop_full_parity.json',{'at':time.time(),'visual_clips':cases,
        'nod_positive':{'sample_id':row['sample_id'],'compared_keys':keys,'exact_match':True,
                        'new_performance':new['performance']},'production_changed':False})
    print('Whole-clip pixels/PTS/JPEG metadata exact; nod media, pose, scores and events exact',flush=True)


if __name__=='__main__':main()
