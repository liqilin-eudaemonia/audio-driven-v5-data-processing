"""Append alignment-only retries; never rewrite the original ASR answers."""
import argparse,dataclasses,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run).resolve()
    cfg=json.loads((run/'config.json').read_text());sys.path.insert(0,cfg['asr_dependency_path'])
    import torch
    from qwen_asr import Qwen3ForcedAligner
    from listen_benchmark.full_raw import prepare
    from listen_benchmark.windows import slice_media
    from p2_listen.contracts import atomic_json
    out=run/'raw/asr_alignment_repairs';out.mkdir(parents=True,exist_ok=True)
    pending={}
    for path in (run/'raw/asr/results').glob('*.jsonl'):
        with path.open() as f:
            for line in f:
                try:r=json.loads(line)
                except ValueError:continue
                if r.get('record_kind')=='result' and r.get('status')=='alignment_failed':pending[r['request_id']]=(path,r)
    model=Qwen3ForcedAligner.from_pretrained(cfg['models']['aligner'],dtype=torch.bfloat16,device_map='cuda:0')
    count=0;fail=0
    with (out/'results.jsonl').open('a',buffering=1) as f:
        for rid,(path,r) in pending.items():
            try:
                media=prepare(r['source'],'asr')
                if r['view']!='full':media=slice_media(media,r['window_start_s'],r['window_end_s'],8)
                words=model.align(audio=(media['audio'],16000),text=r['raw_response']['text'],language=r['raw_response']['language'])[0]
                record=dict(request_id=rid,status='complete',raw_response=dict(r['raw_response'],words=[dataclasses.asdict(w) for w in words]))
                count+=1
            except Exception as exc:record=dict(request_id=rid,status='alignment_failed',error=repr(exc));fail+=1
            record.update(source_raw_file=str(path),source_recorded_at=r['recorded_at'],recorded_at=time.time(),
                          retry_kind='alignment_only_original_transcript_unchanged',human_verified=False,training_admitted=False)
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
            atomic_json(out/'status.json',dict(total=len(pending),complete=count,failed=fail,done=False))
    atomic_json(out/'status.json',dict(total=len(pending),complete=count,failed=fail,done=True))

if __name__=='__main__':main()
