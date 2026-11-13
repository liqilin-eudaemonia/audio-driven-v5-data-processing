"""CPU-only maximal-duration token budget check; no generated labels or scores."""
from pathlib import Path
import os
import sys
import json
import time

WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json
from p2_event_presence.contracts import rows
from p2_event_presence.inference import request
from p2_event_presence.prompts import prompt


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    cfg=read(ROOT/'configs/p2_event_presence_v1.json')
    processor=AutoProcessor.from_pretrained(cfg['model'],trust_remote_code=True,local_files_only=True)
    media=rows(WORK/'events/prepared_pairs.jsonl')[0]['prepared_media']['generated']
    original=np.load(media['windows'][0]['frames'],allow_pickle=False)
    # Repeat a real frame solely to exercise the longest input geometry on CPU.
    video=np.repeat(original[:1],96,axis=0)
    results=[]
    for count in (0,24):
        hints=dict(frame_coverage=1.,cues=[dict(type='nodding',start=i*.4,end=i*.4+.3)for i in range(count)])
        text=prompt('visual',12.,hints,None)
        req=request(processor,'visual',video,text,'context-budget-'+str(count),fps=8)
        output=processor(text=req['prompt'],videos=[video],return_tensors='pt',**req['mm_processor_kwargs'])
        tokens=int(output['input_ids'].shape[-1]);reserve=cfg['max_output_tokens']*2
        results.append(dict(frames=96,height=512,width=512,cues=count,input_tokens=tokens,
            retry_output_reserve=reserve,required_context=tokens+reserve,
            configured_context=cfg['max_model_len'],fits=tokens+reserve<=cfg['max_model_len']))
        del output
    assert not torch.cuda.is_initialized()
    result=dict(at=time.time(),no_gpu=True,geometry_only_no_semantic_evaluation=True,cases=results,
                all_fit=all(r['fits']for r in results))
    atomic_json(WORK/'CONTEXT_BUDGET.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
