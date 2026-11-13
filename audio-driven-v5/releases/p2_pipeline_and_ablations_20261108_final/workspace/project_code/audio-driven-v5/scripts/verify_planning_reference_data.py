"""Bounded real-media + assistant-label verification; no model/GPU training."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from p2_listen.contracts import read,atomic_json,sha
from p2_planning_sft.reference_data import datasets,ReferencePlanningDataset
from p2_planning_sft.data import PlanningCollator


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();config=read(a.config);torch.set_num_threads(2);begin=time.monotonic()
    train,diagnostic,full=datasets(config)
    counts=dict(train=len(train),diagnostic_dev=len(diagnostic),full_dev=len(full))
    collator=PlanningCollator(config['model'],config['max_length'],config.get('max_pixels',50176))
    tokenizer=collator.get_processor().tokenizer;results=[]
    for split,n in [('train',8),('dev',4)]:
        dataset=ReferencePlanningDataset(config['data']['seal'],split,limit=n,seed=config['seed'])
        # Exercise process-local SQLite/file handles in real parallel media reads.
        loader=DataLoader(dataset,batch_size=None,num_workers=2)
        for item in loader:
            item['audio']=np.asarray(item['audio'],dtype=np.float32)
            assert item['image'].size==(512,512)
            assert len(item['audio'])==round(item['duration']*16000)
            assert not item['audit']['future_listener_media_in_inputs']
            batch=collator([item]);labels=batch['labels'][0];supervised=labels[labels!=-100].tolist()
            expected=tokenizer.encode(item['target'],add_special_tokens=False)+[tokenizer.convert_tokens_to_ids('<|im_end|>')]
            assert supervised==expected
            assert (batch['labels'][batch['attention_mask']==0]==-100).all()
            source=collator.source_inputs([item]);prefix=source['input_ids'][0][source['attention_mask'][0].bool()]
            assert (labels[:len(prefix)]==-100).all()
            assert torch.equal(batch['input_ids'][0,:len(prefix)],prefix)
            results.append(dict(sample_id=item['sample_id'],split=split,duration=item['duration'],audio_samples=len(item['audio']),
                image_size=list(item['image'].size),input_tokens=int(batch['attention_mask'].sum()),target_tokens=len(supervised),
                audit=item['audit'],target_matches=True,prefix_mask_correct=True))
            print(split,item['sample_id'],'OK',flush=True)
        dataset.close()
    report=dict(success=True,counts=counts,config_sha256=sha(a.config),seal_sha256=sha(config['data']['seal']),
        samples=results,wall_seconds=time.monotonic()-begin,scope='full manifest/index checks in seal; 12 real media/processor checks with two CPU workers; no GPU backward or throughput test')
    atomic_json(a.output,report)
    print(counts,flush=True)


if __name__=='__main__':main()
