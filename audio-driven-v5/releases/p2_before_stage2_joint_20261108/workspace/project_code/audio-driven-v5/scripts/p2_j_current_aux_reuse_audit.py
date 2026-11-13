#!/usr/bin/env python3
"""CPU-only independent Whisper/Feature-teacher incremental membership audit."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json,multiprocessing as mp,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_j_current_qwen_hidden import Reader as HReader,ensure
from p2_j_current_vae_reuse_audit import video_inputs
from p2_whisper_av.cache import Reader as OldWhisper,observation_contract,model_contract
from p2_j_teacher.core import Reader as OldTeacher
OLD=ROOT/'experiments/p2_j_train500h_cache_20261101'
SOURCE=ROOT/'experiments/p2_j_train_dev500h_qwen_hidden_20261105'
WHISPER=ROOT/'experiments/p2_j_speaker_whisper_cache_20261104_prefetch_v2'
TEACHER=ROOT/'experiments/p2_j_teacher500h_node8_20261102/production'
WORK=ROOT/'experiments/p2_j_train_dev500h_aux_reuse_audit_20261105'
_w=_t=_h=None


def initialize():
    global _h
    _h=HReader(SOURCE,verify=False)


def one(sid):
    row=_h.row(sid);out=dict(sample_id=sid,split=row['split'],duration_s=row['duration_s'],source_sha256=digest(row['source']),restored='orientation_restoration' in row)
    if sid not in _w.ids or sid not in _t.ids:
        return dict(out,whisper='new',teacher='new',reason='not_in_old_cache_members')
    try:
        item=_t.item(sid);ensure(digest(item)==_t.source.manifest['entries'][sid],'Old frozen sample differs')
        # Same pair-result bytes prove same normalized paired speaker record.
        ensure(row['source']['source_result_sha256']==item['source']['source_result_sha256'],'Paired speaker metadata changed')
        current=dict(item,source=row['source'],split=row['split'])
        obs=observation_contract(current);wm=read(WHISPER/'cache'/sid/'complete.json');wident=_w.identity(item)
        ensure(obs==observation_contract(item),'Speaker audio/time observation changed')
        ensure(wm['contract']==wident and wm['identity']==digest(wident),'Whisper contract changed')
        wp=WHISPER/'cache'/sid/'payload.pt';ensure(wp.stat().st_size>0 and len(wm['payload_sha256'])==64,'Missing Whisper payload')
        tm=read(TEACHER/'cache'/sid/'complete.json');tident=_t.identity(item)
        ensure(tm['contract']==tident and tm['identity']==digest(tident),'Teacher contract changed')
        tp=TEACHER/'cache'/sid/'payload.pt';ensure(tp.stat().st_size>0 and len(tm['payload_sha256'])==64,'Missing teacher payload')
        vsame=video_inputs(row['source'])==video_inputs(item['source']);asame=row['audio_recipe']==item['audio_recipe'] and row['duration_s']==item['source']['duration_s']
        tdisp='reuse_both' if vsame and asame else 'reuse_audio' if asame else 'reuse_video' if vsame else 'new'
        return dict(out,whisper='reuse',teacher=tdisp,whisper_old_payload=str(wp),whisper_identity=wm['identity'],whisper_payload_sha256=wm['payload_sha256'],whisper_bytes=wp.stat().st_size,
                    teacher_old_payload=str(tp),teacher_identity=tm['identity'],teacher_payload_sha256=tm['payload_sha256'],teacher_bytes=tp.stat().st_size,
                    old_item_sha256=digest(item),current_observation_sha256=digest(obs),reviewed_pts_override=_t.source.manifest.get('video_sampling_overrides',{}).get(sid))
    except Exception:return dict(out,whisper='attention',teacher='attention',error=traceback.format_exc())


def block(args):
    i,ids=args;rows=[one(s) for s in ids];f=WORK/'blocks'/f'{i:06d}.jsonl';f.parent.mkdir(parents=True,exist_ok=True);f.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    stats=Counter()
    for x in rows:stats['whisper_'+x['whisper']]+=1;stats['teacher_'+x['teacher']]+=1
    return dict(samples=len(rows),stats=dict(stats))


def main():
    global _w,_t
    import torch
    torch.set_num_threads(1);WORK.mkdir(parents=True,exist_ok=True);_w=OldWhisper(WHISPER,require_complete=True);_t=OldTeacher(TEACHER);h=HReader(SOURCE)
    ensure(model_contract(_w.protocol['encoder']['path'])==_w.protocol['encoder'],'Whisper encoder model/numerics changed')
    from p2_listen.config import extractor_config
    ensure(extractor_config(h.config).to_dict()==_t.protocol['dependencies']['extractor'],'Feature-teacher encoder/layout differs')
    p=dict(source_protocol_sha256=sha(SOURCE/'protocol.json'),selection_sha256=h.protocol['selection_sha256'],whisper_protocol_sha256=sha(WHISPER/'protocol.json'),whisper_complete_sha256=sha(WHISPER/'COMPLETE.json'),teacher_protocol_sha256=sha(TEACHER/'protocol.json'),teacher_complete_sha256=sha(TEACHER/'COMPLETE.json'),code_sha256=sha(Path(__file__)),old_payload_validation='Accepted old full completion + per-member frozen source/contract/checksum-marker/file presence; payload SHA is verified by reader on access, not all rehashed in this CPU membership audit.')
    atomic_json(WORK/'protocol.json',p,immutable=True);stats=Counter();n=0;ids=h.protocol['sample_ids'];start=time.time()
    with ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork'),initializer=initialize) as pool:
        for x in pool.map(block,[(i,ids[a:a+256]) for i,a in enumerate(range(0,len(ids),256))],chunksize=1):
            stats.update(x['stats']);n+=x['samples']
            if n%16384==0 or n==len(ids):atomic_json(WORK/'progress.json',dict(samples=n,total=len(ids),stats=dict(stats),elapsed_s=time.time()-start))
    ensure(n==284218,'Wrong full member coverage');whisper=[];tv=[];ta=[];counts=Counter();errors=[]
    with (WORK/'reuse_map.jsonl').open('w') as out:
        for f in sorted((WORK/'blocks').glob('*.jsonl')):
            for line in f.open():
                x=json.loads(line);counts[x['split']]+=1;out.write(line)
                if x['whisper']!='reuse':whisper.append(x['sample_id'])
                if x['teacher'] not in ('reuse_both','reuse_video'):tv.append(x['sample_id'])
                if x['teacher'] not in ('reuse_both','reuse_audio'):ta.append(x['sample_id'])
                if 'error' in x:errors.append(x)
    for name,v in [('whisper_increment_ids.json',whisper),('teacher_video_increment_ids.json',tv),('teacher_audio_increment_ids.json',ta)]:atomic_json(WORK/name,v,immutable=True)
    result=dict(status='complete' if not errors else 'attention_required',samples=n,source_split_counts=dict(counts),stats=dict(stats),whisper_to_encode=len(whisper),teacher_video_to_encode=len(tv),teacher_audio_to_encode=len(ta),errors=errors,elapsed_s=time.time()-start,encoding_started=False,old_payloads_modified=False)
    atomic_json(WORK/'SUMMARY.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
