#!/usr/bin/env python3
"""Read-only CPU reuse audit; never encodes VAE or mutates existing caches."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,multiprocessing as mp,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_j_current_qwen_hidden import Reader as HReader,ensure
from p2_j_cache.runtime import CacheReader,cache_identity
OLD=ROOT/'experiments/p2_j_train500h_cache_20261101'
SOURCE=ROOT/'experiments/p2_j_train_dev500h_qwen_hidden_20261105'
WORK=ROOT/'experiments/p2_j_train_dev500h_vae_reuse_audit_20261105'
_old=None
_new=None


def initialize():
    global _new
    _new=HReader(SOURCE,verify=False)


def video_inputs(source):
    return dict(listener_id=source['listener_id'],video=source['source'],start=source['source_start_s'],end=source['source_end_s'],duration=source['duration_s'],transform=source['video_transform'],fps=25,size=512)


def one(sid):
    row=_new.row(sid);base=dict(sample_id=sid,split=row['split'],duration_s=row['duration_s'],caption_sha256=row['caption_file_sha256'],new_source_sha256=digest(row['source']),restored='orientation_restoration' in row)
    if sid not in _old.manifest['entries']:
        return dict(base,disposition='new_member_no_old_cache',video_reusable=False,audio_reusable=False)
    try:
        item=read(OLD/'samples'/(sid+'.json'));ensure(digest(item)==_old.manifest['entries'][sid],'Old frozen sample changed')
        folder=OLD/'cache/vae'/sid;m=read(folder/'complete.json');ident=cache_identity(_old.manifest,item,'vae',_old.assets['vae'])
        ensure(m['contract']==ident and m['identity']==digest(ident),'Old VAE identity changed')
        payload=folder/'payload.pt';actual=sha(payload);ensure(actual==m['payload_sha256'],'Old VAE payload checksum mismatch')
        audit=read(OLD/'media_audit'/(sid+'.json'))
        ensure(audit['sample_id']==sid and audit['final_crop']==item['source']['video_transform']['crop'] and audit['reference_from_first_frame'] and audit['source_video_used'],'Old final crop/reference audit mismatch')
        override=_old.manifest.get('video_sampling_overrides',{}).get(sid)
        ensure(audit.get('video_sampling_override')==override,'Old reviewed PTS exception mismatch')
        vsame=video_inputs(row['source'])==video_inputs(item['source'])
        asame=(row['audio_recipe']==item['audio_recipe'] and row['duration_s']==item['source']['duration_s'])
        if override and vsame:
            ensure(override['source_digest']==digest(item['source']),'Old PTS override source changed')
        disposition='reuse_both' if vsame and asame else 'reuse_audio_reencode_video' if asame else 'reuse_video_reencode_audio' if vsame else 'reencode_both_inputs_changed'
        return dict(base,disposition=disposition,video_reusable=vsame,audio_reusable=asame,caption_changed=row['caption']!=item['caption'],
                    source_metadata_only_changed=(row['source']!=item['source'] and vsame),old_source_sha256=digest(item['source']),
                    old_payload=str(payload),payload_sha256=actual,payload_bytes=payload.stat().st_size,old_identity=m['identity'],
                    audio_recipe_sha256=digest(row['audio_recipe']),reviewed_pts_override=override)
    except Exception:
        return dict(base,disposition='old_cache_unusable',video_reusable=False,audio_reusable=False,error=traceback.format_exc())


def block(args):
    i,ids=args;results=[one(s) for s in ids];path=WORK/'blocks'/f'{i:06d}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    stats=Counter();secs=defaultdict(float);splits=Counter();reused_bytes=0
    for r in results:
        d=r['disposition'];stats[d]+=1;secs[d]+=r['duration_s'];splits[r['split']]+=1
        for key in ['video_reusable','audio_reusable','caption_changed','source_metadata_only_changed','restored']:
            if r.get(key):stats[key]+=1
        if r.get('reviewed_pts_override'):stats['reviewed_pts_override']+=1
        if d=='reuse_both':reused_bytes+=r['payload_bytes']
    return dict(samples=len(results),stats=dict(stats),seconds=dict(secs),splits=dict(splits),reused_bytes=reused_bytes)


def main():
    global _old
    import torch
    torch.set_num_threads(1);WORK.mkdir(parents=True,exist_ok=True);_old=CacheReader(OLD);new=HReader(SOURCE)
    ensure(read(OLD/'COMPLETE.json')['success'],'Old AV cache incomplete')
    ensure(new.config['data']['fps']==_old.manifest['config']['data']['fps']==25 and new.config['data']['size']==_old.manifest['config']['data']['size']==512,'AV geometry differs')
    ids=new.protocol['sample_ids'];oldids=set(_old.manifest['sample_ids']);start=time.time()
    protocol=dict(schema='current-j-vae-reuse-read-only-v1',old_work=str(OLD),old_manifest_sha256=sha(OLD/'manifest.json'),old_complete_sha256=sha(OLD/'COMPLETE.json'),
                  current_qwen_protocol_sha256=sha(SOURCE/'protocol.json'),selection_sha256=new.protocol['selection_sha256'],
                  samples=len(ids),common_ids=len(oldids&set(ids)),new_only_ids=len(set(ids)-oldids),
                  code_sha256=sha(Path(__file__)),assets_verified_by_old_cache_reader=True,
                  verification='Every common VAE payload SHA256, old source/contract/media audit, separate actual video and listener audio input signatures',
                  encoding_started=False)
    atomic_json(WORK/'protocol.json',protocol,immutable=True)
    stats=Counter();seconds=defaultdict(float);splits=Counter();n=0;reused_bytes=0
    blocks=[(i,ids[start:start+256]) for i,start in enumerate(range(0,len(ids),256))]
    with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork'),initializer=initialize) as pool:
        for value in pool.map(block,blocks,chunksize=1):
            n+=value['samples'];stats.update(value['stats']);splits.update(value['splits']);reused_bytes+=value['reused_bytes']
            for k,v in value['seconds'].items():seconds[k]+=v
            if n%8192==0 or n==len(ids):
                p=dict(samples=n,total=len(ids),stats=dict(stats),elapsed_s=time.time()-start);atomic_json(WORK/'progress.json',p);print(json.dumps(p),flush=True)
    ensure(n==len(ids) and dict(splits)==new.protocol['source_split_counts'],'New selection coverage mismatch')
    dispositions=['reuse_both','reuse_audio_reencode_video','reuse_video_reencode_audio','reencode_both_inputs_changed','new_member_no_old_cache','old_cache_unusable']
    summary=dict(status='complete',samples=n,source_split_counts=dict(splits),old_samples=len(oldids),common_ids=protocol['common_ids'],new_only_ids=protocol['new_only_ids'],
                 by_disposition={k:dict(samples=stats[k],hours=seconds[k]/3600) for k in dispositions},
                 video_reusable=stats['video_reusable'],audio_reusable=stats['audio_reusable'],
                 video_encoding_needed=n-stats['video_reusable'],audio_encoding_needed=n-stats['audio_reusable'],
                 caption_changed_among_common=stats['caption_changed'],full_reuse_percent=stats['reuse_both']/n*100,
                 source_metadata_only_changed=stats['source_metadata_only_changed'],restored_selected=stats['restored'],
                 reviewed_pts_overrides_kept=stats['reviewed_pts_override'],old_payload_bytes_referenced_if_reused=reused_bytes,
                 all_common_payloads_sha256_verified=stats['old_cache_unusable']==0,
                 elapsed_s=time.time()-start,encoding_started=False,old_data_modified=False,gemma_not_stopped=True,
                 limits='Checks existing frozen media identities and cached payload bytes; does not decode all raw videos again or claim cross-ID/time-overlap deduplication.')
    atomic_json(WORK/'SUMMARY.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
