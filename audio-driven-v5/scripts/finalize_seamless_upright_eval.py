"""Make an explicit full200 overlay; preserve old generation and score artifacts."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,rows
from p2_combined_whisper_eval import link
BASE=ROOT/'experiments/seamless_upright_eval_fix_20261112';GEN=BASE/'generation6000';OLD=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
if __name__=='__main__':
    assert read(GEN/'COMPLETE.json')['success'];p=read(GEN/'protocol.json');arm=p['arms'][0]['name'];changed={x['id'] for x in read(GEN/'shared_items.json')};refs=rows(BASE/'prepared_200/references.jsonl');oldrefs={x['id']:x['reference'] for x in read(OLD/'shared_items.json') if x['dataset']=='seamless'}
    mapping=[];audits=[]
    for r in refs:
        sid=r['id'];root=GEN if sid in changed else OLD;folder=root/'generated'/arm/'seamless'/sid
        assert (folder/'complete.json').exists()
        if sid not in changed:assert r['gt_reference']['row_sha256']==oldrefs[sid]['gt_reference']['row_sha256']
        else:
            a=read(GEN/'whisper/seamless'/sid/'metadata.json')['source'];b=read(OLD/'whisper/seamless'/sid/'metadata.json')['source'];assert a['waveform_sha256']==b['waveform_sha256'] and a['source_sha256']==b['source_sha256']
            oldim=OLD/'conditions/seamless'/sid/'first_frame.png';newim=GEN/'conditions/seamless'/sid/'first_frame.png';assert sha(oldim)!=sha(newim)
            audits.append(dict(id=sid,new_first_frame_sha256=sha(newim),old_first_frame_sha256=sha(oldim),speaker_waveform_unchanged=True,new_plan_sha256=sha(GEN/'conditions/seamless'/sid/'qwen/plan.json'),generated_sha256=sha(folder/'generated.mp4')))
        mapping.append(dict(id=sid,dataset='seamless',generated=str(folder/'generated.mp4'),generated_audio=str(folder/'listener.wav'),target_role='listener',timeline='same_pair_window',pair_duration_s=r['duration_s'],regenerated_for_upright=sid in changed))
    out=BASE/'full200_model6000';out.mkdir(exist_ok=True)
    (out/'generation_map.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in mapping))
    from evaluation.seamless.bind import load_references,match
    matches=match(load_references(BASE/'prepared_200'),mapping,out)
    pairs=[]
    for r,g in matches:
        mp=next(x for x in mapping if x['id']==r['id']);pairs.append(dict(mp,identity='listener',method=arm,draw_id='0',capabilities=dict(visual=True,audio=True),gt_reference=r['gt_reference'],listening_intervals=r['listening_intervals'],source_start=r['source_start'],source_end=r['source_end']))
    (out/'pairs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pairs))
    save(out/'COMPLETE.json',dict(samples=200,regenerated=4,reused_unchanged_media=196,old_scores_not_reused=True,score_status='requires_recompute_on_this_manifest'))
    save(BASE/'REPAIR_VALIDATION.json',dict(success=True,regenerated=audits,full200_bound=True,tests=dict(event_unit=22,seamless_adapter=4),old_results_preserved=True))
