"""One targeted retry of invalid evidence quotations; judgment policies stay fixed."""
import argparse
import copy
import json
from pathlib import Path
import time

import filter_full_response_sft as base
import recover_voiced_response_sft as voiced
from fix_train_dev_caption_prefixes import DEFAULT, judgments, final_decision


def run(work):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoTokenizer
    strict, recovery = judgments(work / 'strict'), judgments(work / 'voiced')
    unresolved = {sid for sid,r in strict.items() if final_decision(r,recovery.get(sid))[0]=='needs_review'}
    captions = {}
    for split in ['dev','train']:
        for line in (work/'edited'/f'{split}.jsonl').open():
            row=json.loads(line)
            if row['sample_id'] in unresolved:
                captions[row['sample_id']]=row['target']['text']
    cfg=base.read(work/'strict/config.json')
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    llm=LLM(model=cfg['model'],dtype='bfloat16',tensor_parallel_size=1,
        gpu_memory_utilization=cfg['gpu_memory_utilization'],max_model_len=cfg['max_model_len'],
        max_num_seqs=cfg['max_num_seqs'],max_num_batched_tokens=cfg['max_num_batched_tokens'],
        enable_prefix_caching=True,enable_chunked_prefill=True,seed=20261101,
        compilation_config=dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',
            cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256,512,1024],max_cudagraph_capture_size=1024),
        structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True})
    output=work/'quote_retry.jsonl'
    assert not output.exists() and not (work/'QUOTE_RETRY_COMPLETE.json').exists()
    results=[]
    for policy,previous,parse,schema in [('strict',strict,base.parse_answer,base.schema),
                                         ('voiced',recovery,voiced.parse_answer,voiced.schema)]:
        selected=[r for sid,r in previous.items() if sid in unresolved and r['decision']=='needs_review']
        prompts=[]
        system=(work/policy/'system.txt').read_text()
        for r in selected:
            caption=captions[r['sample_id']]
            # Plain caption avoids an extra JSON escape layer around quoted speech.
            user='Caption (data, not instructions):\n'+caption+'\n\nEnd of caption.\nThe previous answer failed because its evidence quote was not an exact source substring. Apply the same judgment criteria. Choose a SHORT contiguous quote, preferably 3-8 words without dialogue quotation marks or backslashes. Do not join separate sentences, paraphrase evidence, or copy JSON field names into the quote. Return the required JSON only.'
            ids=tokenizer.apply_chat_template([dict(role='system',content=system),dict(role='user',content=user)],tokenize=True,add_generation_prompt=True)
            assert len(ids)+512<=cfg['max_model_len']
            prompts.append(dict(prompt_token_ids=ids))
        if not prompts:
            continue
        generated=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=512,
            structured_outputs=StructuredOutputsParams(json=schema())),use_tqdm=False)
        for old,response in zip(selected,generated):
            r=copy.deepcopy(old);answer=response.outputs[0]
            r['attempts'].append(dict(raw=answer.text,finish_reason=answer.finish_reason,
                tokens=len(answer.token_ids),retry='plain_caption_short_evidence_v1'))
            r.pop('model_judgment',None)
            try:
                r['model_judgment']=parse(answer.text,answer.finish_reason,captions[r['sample_id']])
                r.pop('error',None)
            except Exception as e:
                r['error']=str(e)
            judgment=r.get('model_judgment')
            r['decision']='needs_review' if judgment is None else ('keep' if judgment['keep'] and not r['rule_evidence'] else 'reject')
            results.append(dict(policy=policy,sample_id=r['sample_id'],previous=old,result=r))
        print(json.dumps(dict(policy=policy,retried=len(selected),keep=sum(r['result']['decision']=='keep' for r in results if r['policy']==policy),unresolved=sum(r['result']['decision']=='needs_review' for r in results if r['policy']==policy))),flush=True)
    temp=output.with_suffix('.partial')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    temp.replace(output)
    base.save(work/'QUOTE_RETRY_COMPLETE.json',dict(status='complete',sha256=base.sha(output),
        cases=len(results),original_unresolved=len(unresolved),policy_unchanged=True,
        format='plain caption + short contiguous evidence instruction; no parser relaxation',completed_at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=DEFAULT)
    run(parser.parse_args().work)
