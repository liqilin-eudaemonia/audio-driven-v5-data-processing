"""Screen completed recovery preparations, one GPU process per stable shard."""
import argparse,collections,time,traceback
import evaluate_listen_v8_recovery as e

def main(rank,world):
    e.setup()
    contract=e.read(e.OUT/'contract.json')
    for name,digest in contract['code_sha256'].items():assert e.sha(e.ROOT/name)==digest
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ast=e.baseline.full.legacy.EventModel(e.baseline.full.legacy.MODEL,batch_size=16)
    panns=e.baseline.full.legacy.PannsModel(batch_size=16)
    rejects=e.read(e.BASE/'human_rejections.json')['rows'];positives=e.read(e.BASE/'human_retained.json')['rows']
    feedback=e.read(e.BASE/'qc4_review_feedback.json')['rows'];counts=collections.Counter()
    for pair in e.read(e.OUT/'selection.json')['pairs'][rank::world]:
        path=e.OUT/'prepared'/(pair+'.json');started=time.monotonic()
        while not path.exists():
            if (e.OUT/'prepare_summary.json').exists():break
            if time.monotonic()-started>1800:raise TimeoutError('Preparation did not finish: '+pair)
            time.sleep(1)
        if not path.exists():counts['missing_preparation']+=1;continue
        item=e.read(path);assert e.sha(e.BASE/'results'/path.name)==item['old_result_sha256']
        rows=[]
        try:
            for row in item['novel_clips']:
                r=e.baseline.assess(e.baseline.BASE_SCREEN(row,item['normalized'],ast,panns,rejects,positives),feedback)
                r['recovery']=e.recovery.analysis(r);rows.append(r)
            e.write(e.OUT/'results'/path.name,dict(pair=pair,split=item['split'],clips=rows,complete=True))
            counts['pairs']+=1;counts['clips']+=len(rows)
        except Exception as error:
            counts['errors']+=1;e.write(e.OUT/'errors'/f'screen_{path.name}',dict(pair=pair,error=repr(error),traceback=traceback.format_exc()))
        e.write(e.OUT/'progress'/f'rank_{rank}.json',dict(utc=e.baseline.full.now(),state='running',counts=dict(counts)))
    e.write(e.OUT/'progress'/f'rank_{rank}.json',dict(utc=e.baseline.full.now(),state='finished',counts=dict(counts)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,default=8);a=p.parse_args();main(a.rank,a.world)
