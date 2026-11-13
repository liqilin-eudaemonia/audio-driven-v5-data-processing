"""Experimental ASR + word alignment. Model output is not human ground truth."""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dependency-path')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.dependency_path:sys.path.insert(0,a.dependency_path)
    import torch
    from qwen_asr import Qwen3ASRModel,Qwen3ForcedAligner
    from listen_benchmark.media import prepare
    asr=Qwen3ASRModel.from_pretrained('/private/ckpts/Qwen3-ASR-1.7B',dtype=torch.bfloat16,device_map='cuda:0',max_inference_batch_size=8,max_new_tokens=256)
    aligner=Qwen3ForcedAligner.from_pretrained('/private/ckpts/Qwen3-ForcedAligner-0.6B',dtype=torch.bfloat16,device_map='cuda:0')
    for row in json.loads(Path(a.manifest).read_text()):
        start=time.monotonic();result={'sample_id':row['sample_id'],'human_ground_truth':False,'words':[]}
        try:
            media=prepare(row,'audio',{'fps':8,'max_frames':128,'max_edge':768,'max_audio_seconds':16})
            transcript=asr.transcribe(audio=(media['audio'],16000),language=None)[0]
            result.update(text=transcript.text,language=transcript.language,input_evidence=media['meta'])
            if transcript.text.strip():
                aligned=aligner.align(audio=(media['audio'],16000),text=transcript.text,language=transcript.language)[0]
                result['words']=[dataclasses.asdict(x) for x in aligned]
            result['status']='complete'
        except Exception as e:
            result.update(status='failed',error=repr(e))
        result['seconds']=time.monotonic()-start
        (out/(row['sample_id']+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='input_evidence'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
