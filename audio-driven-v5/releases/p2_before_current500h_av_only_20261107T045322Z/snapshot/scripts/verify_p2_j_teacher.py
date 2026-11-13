"""Small real-media equivalence and per-GPU batched teacher benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from p2_listen.contracts import atomic_json
from p2_j_teacher.core import Reader,load_model,extract_batch,persist
from p2_j_teacher.media import initialize_processor,prepare


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--batch',type=int,required=True);p.add_argument('--producers',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(2);reader=Reader(a.work);runtime,processor,size,grid=load_model(reader)
    ids=reader.protocol['sample_ids'];overrides=reader.source.manifest.get('video_sampling_overrides',{})
    report=dict(gpu=a.gpu,batch=a.batch,producers=a.producers,samples=len(ids),success=False)
    out=Path(a.work)/f'benchmark-gpu{a.gpu}.json';started=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.producers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_processor,initargs=(reader.protocol['model'],)) as pool:
        futures=[pool.submit(prepare,reader.item(s),overrides.get(s),size,grid) for s in ids]
        prepared=[f.result(timeout=180) for f in futures]
        report['prepare_s']=time.monotonic()-started
        if a.gpu==0:
            refs=[pool.submit(prepare,reader.item(s),overrides.get(s),size,grid,True) for s in ids]
            for new,f in zip(prepared,refs):
                ref=f.result(timeout=180)
                assert new['audit']==ref['audit'],new['sample_id']
                for mod in ('video','audio'):
                    assert new[mod].keys()==ref[mod].keys()
                    for k in new[mod]:assert torch.equal(new[mod][k],ref[mod][k]),(new['sample_id'],mod,k)
            report['full25fps_reference_equal']=len(ids)
            atomic_json(out.with_suffix('.progress.json'),report)
    extract_batch(runtime,prepared[:1]);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    t=time.monotonic();values=[]
    for i in range(0,len(prepared),a.batch):values+=extract_batch(runtime,prepared[i:i+a.batch])
    torch.cuda.synchronize();report['inference_s']=time.monotonic()-t
    report['samples_per_gpu_second']=len(ids)/report['inference_s'];report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    if a.gpu==0:
        serial=[extract_batch(runtime,[v])[0] for v in prepared]
        differences=[]
        for new,ref in zip(values,serial):
            row={}
            for mod in ('video','audio'):
                assert torch.equal(new[mod+'_valid'],ref[mod+'_valid'])
                x,y=new[mod].float(),ref[mod].float();ratio=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-8))
                row[mod]=dict(relative_l2=ratio,max_abs=float((x-y).abs().max()),exact=torch.equal(new[mod],ref[mod]))
            differences.append(row)
        report['batch_vs_single']=differences
        atomic_json(out.with_suffix('.progress.json'),report)
        assert all(row[mod]['relative_l2']<.005 for row in differences for mod in ('video','audio')),differences
        for sid,value in zip(ids,values):persist(reader,reader.item(sid),value)
        for sid in ids:
            bundle=reader.training_sample(sid,'guided')
            assert bundle['teacher']['video'].shape[:2]==bundle['layout']['video_valid'].shape
        report['guided_cache_teacher_read_ok']=len(ids)
    report['success']=True;atomic_json(out,report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
