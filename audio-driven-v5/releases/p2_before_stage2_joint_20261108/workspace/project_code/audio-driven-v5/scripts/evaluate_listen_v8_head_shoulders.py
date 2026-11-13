"""512-pair crop pilot; same sample prefix and frozen downstream audio checks."""
import collections,concurrent.futures,json
import evaluate_listen_v8_recovery as e
import listen_v8_head_shoulders as geometry
CROP=e.ROOT/'experiments/p3_listen_v8_head_shoulders_20261023'

def init_worker():
    e.OUT=CROP;e.setup();e.baseline.full.planner.rules.full_turn_crop=geometry.crop
    e.CATALOG={r['pair']:r for r in e.read(e.BASE/'catalog.json')['rows']}

def main():
    init_worker();pairs=e.read(e.ROOT/'experiments/p3_listen_v8_yield_audit_20261023/sample_pairs.json')[:512]
    e.write(CROP/'selection.json',dict(pairs=pairs,scope='First 512 pairs of fixed random audit sample; crop trial, not selected for favorable yield'))
    sources=[e.SCRIPT,e.ROOT/'scripts/listen_v8_recovery.py',e.ROOT/'scripts/listen_v8_head_shoulders.py',e.ROOT/'scripts/evaluate_listen_v8_head_shoulders.py']
    e.write(CROP/'contract.json',dict(revision=geometry.REVISION,baseline_contract_sha256=e.sha(e.BASE/'contract.json'),
        code_sha256={str(p.relative_to(e.ROOT)):e.sha(p) for p in sources},selection_sha256=e.sha(CROP/'selection.json'),
        native_face_short_p10_minimum=160,output_face_short_p10_minimum=128,native_crop_minimum=512,training_admitted=False))
    counts=collections.Counter();errors=[]
    with concurrent.futures.ProcessPoolExecutor(8,initializer=init_worker) as pool:
        for value,error in pool.map(e.prepare_attempt,pairs):
            counts['pairs']+=1
            if error:errors.append(error)
            else:counts.update({k:value[k] for k in ('old','new','novel')})
            if counts['pairs']%32==0:
                e.write(CROP/'prepare_progress.json',dict(counts=dict(counts),errors=len(errors)));print(json.dumps(dict(counts)),flush=True)
    e.write(CROP/'prepare_summary.json',dict(counts=dict(counts),errors=errors))

if __name__=='__main__':main()
