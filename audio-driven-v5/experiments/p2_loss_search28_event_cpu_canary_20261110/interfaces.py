"""Check the installed detector processor/structured-output API without CUDA."""
from pathlib import Path
import importlib.metadata
import json
import os
import sys
import time

WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json
from p2_event_presence.contracts import rows
from p2_event_presence.inference import request,model_identity
from p2_event_presence.prompts import prompt,schema,ASR_PROMPT,ASR_SCHEMA


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    import soundfile as sf
    from transformers import AutoProcessor
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    torch.set_num_threads(2)
    cfg=read(ROOT/'configs/p2_event_presence_v1.json')
    processor=AutoProcessor.from_pretrained(cfg['model'],trust_remote_code=True,local_files_only=True)
    record=rows(WORK/'events/prepared_pairs.jsonl')[0];media=record['prepared_media']['generated']
    data=np.load(media['windows'][0]['frames'],allow_pickle=False)
    wave,sr=sf.read(media['audio'],dtype='float32');assert sr==16000
    cases=[('visual',data,prompt('visual',media['duration_s'],dict(cues=[],frame_coverage=1.),None),schema('visual')),
           ('audio',wave,ASR_PROMPT,ASR_SCHEMA),
           ('audio',wave,prompt('audio',media['duration_s'],None,dict(transcript='',nonlexical_syllables=[],normalized_candidate_texts=[])),schema('audio'))]
    details=[]
    for index,(mod,array,text,structure)in enumerate(cases):
        req=request(processor,mod,array,text,'cpu-interface-'+str(index),fps=8)
        sampling=SamplingParams(temperature=0.,seed=137,max_tokens=768,
            structured_outputs=StructuredOutputsParams(json=structure))
        assert sampling.structured_outputs.json==structure
        assert isinstance(req['prompt'],str)and req['prompt']
        assert req['mm_processor_kwargs']['use_audio_in_video']is False
        details.append(dict(modality=mod,shape=list(array.shape),text_tokens=len(processor.tokenizer.encode(req['prompt'])),
                            structured_output_supported=True,video_audio_separated=True))
    assert not torch.cuda.is_initialized()
    result=dict(success=True,at=time.time(),no_gpu=True,no_generation=True,
        processor=type(processor).__name__,versions={k:importlib.metadata.version(k)for k in ['torch','vllm','transformers','numpy']},
        detector_assets=model_identity(cfg['model']),requests=details)
    atomic_json(WORK/'INTERFACES_ACCEPTED.json',result)
    print(json.dumps({k:v for k,v in result.items()if k!='detector_assets'}),flush=True)


if __name__=='__main__':main()
