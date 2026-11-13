#!/usr/bin/env python3
"""Publish the upright media / repeated-caption repair as one audited label entrance."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time
import orjson
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/listen_j_labels_manifest_20261105'
WORK=ROOT/'experiments/listen_j_upright_regen_20261105'
OUT=ROOT/'experiments/listen_j_labels_upright_20261105'
ORIENT=ROOT/'experiments/listen_j_clean_regen_20261105'
OLD_SHA='b7dc75f51c9f408ff779ad9b7c4f5e42c9eee3b3a2bfdc34eda501269547ddda'
COVER_SHA='9de926f75044dc6da0b8487a3f5c94883946ca2475a7c3623f86ccab71210089'
CAPTION_FIELDS=['caption','caption_file','caption_file_sha256','caption_identity_version','pipeline_id','prompt_version','generation_checks_passed','review_flags','remaining_gaps','finish_reason','completed_at']
RESTORE_FIELDS=['source','quality','crop_provenance','detector_measurement_sha256','face_p10','face_p50']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def read(p):return orjson.loads(Path(p).read_bytes())
def save(p,d):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_bytes(orjson.dumps(d,option=orjson.OPT_INDENT_2)+b'\n');t.replace(p)
def canonical(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sourcekey(r):
 s=r['source']['source'];return hashlib.sha256((s['archive']+'\0'+s['member']).encode()).hexdigest()[:24]
def context():
 jobs={r['sample_id']:r for r in map(orjson.loads,(WORK/'regenerate_candidates.jsonl').read_bytes().splitlines())}
 success={p.stem:(p,read(p)) for p in (WORK/'regenerated').glob('*.json')}
 failed={p.stem:(p,read(p)) for p in (WORK/'regeneration_rejected').glob('*.json')}
 visual={p.stem:(p,read(p)) for p in (WORK/'excluded_visual').glob('*.json')}
 assert set(success).isdisjoint(failed) and set(success).isdisjoint(visual) and set(failed).isdisjoint(visual)
 assert set(success)|set(failed)|set(visual)==set(jobs),('jobs_incomplete',len(jobs),len(success),len(failed),len(visual))
 repeated=set(read(WORK/'repeated_sample_ids.json'));rotated={sid for sid,r in jobs.items() if r.get('_restore_orientation')}
 assert len(jobs)==1734 and len(repeated)==36 and len(rotated)==1699
 assert read(WORK/'restoration_summary.json')['total']==1699
 return jobs,success,failed,visual,repeated,rotated

def validate_caption(old,p,d):
 assert d['sample_id']==old['sample_id'] and d['parent_caption_file']==old['caption_file']
 assert d['parent_caption_sha256']==old['caption_file_sha256']==sha(old['caption_file'])
 parent=read(old['caption_file']);assert d['hard_generation_passed'] and d['finish_reason']=='stop' and d['caption'].strip()
 assert not d['attempts'][-1]['hard_failures']
 assert d['audio_recipe']==old['audio_recipe'] and d['input_evidence']['audio_encoder_sha256']==parent['input_evidence']['audio_encoder_sha256']
 assert d['input_evidence']['listener_audio_is_silent']==old['listener_audio_is_silent']
 assert d['input_evidence']['vocal_supplement']==parent['input_evidence']['vocal_supplement']
 if 'vocal_metadata_source' in d['input_evidence']:assert d['input_evidence']['vocal_metadata_source']==old['source']['source']
 for k in ['sample_id','pair','split','group_id','listener_id','speaker_id','source_start_s','source_end_s','duration_s','normalized_audio']:
  assert d['source'][k]==old['source'][k],(old['sample_id'],k)
 for k in ['video_frames_sha256','audio_encoder_sha256']:assert d['cache_identity'][k]==d['input_evidence'][k]
 assert hashlib.sha256(d['prompt'].encode()).hexdigest()==d['cache_identity']['prompt_sha256']
 assert hashlib.sha256(json.dumps(d['cache_identity'],sort_keys=True).encode()).hexdigest()==d['caption_identity_version']
 assert d['prompt'] in d['rendered_prompt']
 return sha(p)

def replace_row(old,p,d,restored):
 h=validate_caption(old,p,d);r=copy.deepcopy(old)
 for f in CAPTION_FIELDS:r[f]=str(p) if f=='caption_file' else h if f=='caption_file_sha256' else d[f]
 r['regeneration_provenance']=dict(parent_caption_file=old['caption_file'],parent_caption_sha256=old['caption_file_sha256'],hard_generation_passed=True,input_video_frames_sha256=d['input_evidence']['video_frames_sha256'],input_audio_encoder_sha256=d['input_evidence']['audio_encoder_sha256'])
 if restored:
  rp=WORK/'restored_rows'/f"{old['sample_id']}.json";rr=read(rp);assert rr['restoration_usable'];prov=read(rr['provenance_file'])
  assert prov['original_source']==old['source'] and d['original_effective_source']==old['source']
  assert d['source']['orientation_restoration']['source_key']==sourcekey(old)
  assert d['quality']['source_sha256']==canonical(rr['request']['source'])
  measurement=read(d['quality']['measurement_file']);assert sha(d['quality']['measurement_file'])==d['quality']['measurement_sha256']
  assert measurement['source']==rr['request']['source'] and measurement['source_sha256']==d['quality']['source_sha256']
  sizes=[min(q['face_width_native'],q['face_height_native']) for q in measurement['raw_measurements'] if q['detected']]
  assert sizes
  r.update(source=d['source'],quality=d['quality'],crop_provenance=d['crop_provenance'],
   detector_measurement_sha256=d['input_evidence']['nod_supplement']['detector_sha256'],face_p10=float(np.percentile(sizes,10)),face_p50=float(np.percentile(sizes,50)))
  r['orientation_restoration']=dict(restoration_file=str(rp),restoration_sha256=sha(rp),provenance_file=rr['provenance_file'],provenance_sha256=sha(rr['provenance_file']),original_effective_source=old['source'],original_quality=old['quality'],original_face_p10=old['face_p10'],original_face_p50=old['face_p50'],rotation_ccw_degrees=prov['rotation_ccw_degrees'],derived_archive_sha256=prov['derived_archive_sha256'])
 else:
  assert d['source']==old['source'] and d['crop_provenance']==old['crop_provenance']
  parent=read(old['caption_file']);assert d['input_evidence']['video_frames_sha256']==parent['input_evidence']['video_frames_sha256']
 return r

def export():
 assert not (OUT/'EXPORT_COMPLETE.json').exists(),'Frozen completed export must not be overwritten'
 jobs,success,failed,visual,repeated,rotated=context();OUT.mkdir(exist_ok=True,parents=True);started=time.monotonic()
 names=['labels','coverage','excluded','exceptions','replacements'];files={n:(OUT/(n+'.jsonl.tmp')).open('wb') for n in names};hashes={n:hashlib.sha256() for n in names}
 counts=Counter();splits=Counter();flags=Counter();durations=Counter();identities=Counter();oldh=hashlib.sha256();oldch=hashlib.sha256();seen=set();replaced=set();excluded=set()
 def put(n,r,raw=None):
  raw=raw if raw is not None else orjson.dumps(r)+b'\n';files[n].write(raw);hashes[n].update(raw)
 try:
  with (OLD/'coverage.jsonl').open('rb') as cf,(OLD/'labels.jsonl').open('rb') as lf:
   for craw in cf:
    oldch.update(craw);c=orjson.loads(craw);sid=c['sample_id'];assert sid not in seen;seen.add(sid);counts['targets']+=1
    if c['status']=='held':counts['held']+=1;put('coverage',c,craw);put('exceptions',c,craw);continue
    assert c['status']=='caption';raw=next(lf);oldh.update(raw);old=orjson.loads(raw);assert sid==old['sample_id']
    if sid in failed or sid in visual:
     p,d=(failed if sid in failed else visual)[sid]
     ex={k:old[k] for k in ['sample_id','pair','split','group_id','listener_id','duration_s','caption_file','caption_file_sha256']}
     ex.update(status='excluded',reason=d['reason'],diagnostic_file=str(p),diagnostic_sha256=sha(p),original_source_key=sourcekey(old),orientation_repair_attempted=sid in rotated,repeated_caption=sid in repeated)
     put('excluded',ex);c.update(status='excluded',reason=d['reason'],diagnostic_file=str(p));put('coverage',c);excluded.add(sid);counts['excluded']+=1
     counts['generation_failed_excluded' if sid in failed else 'unusable_visual_restoration_excluded']+=1
     if sid in repeated:counts['repeated_excluded']+=1
     if sid in rotated:counts['rotation_excluded']+=1
     continue
    if sid in success:
     p,d=success[sid];r=replace_row(old,p,d,sid in rotated);raw=orjson.dumps(r)+b'\n';replaced.add(sid);counts['regenerated']+=1
     if sid in repeated:assert 'repeated_generation' not in r['review_flags']
     if sid in rotated:counts['orientation_restored']+=1
     if sid in repeated:counts['repeated_regenerated']+=1
     rep=dict(sample_id=sid,original_caption_file=old['caption_file'],original_caption_sha256=old['caption_file_sha256'],caption_file=r['caption_file'],caption_file_sha256=r['caption_file_sha256'],orientation_restored=sid in rotated,repeated_caption=sid in repeated)
     put('replacements',rep);c.update(caption_file=r['caption_file'],caption_file_sha256=r['caption_file_sha256'],generation_checks_passed=r['generation_checks_passed'],disposition='orientation_restored_and_regenerated' if sid in rotated else 'repeated_caption_regenerated')
    else:r=old;counts['byte_preserved_original_labels']+=1;c['disposition']='original_preserved'
    assert r['finish_reason']=='stop' and r['training_admitted'] is False
    counts['labels']+=1;counts['with_review_flags' if r['review_flags'] else 'without_review_flags']+=1;splits[r['split']]+=1;flags.update(r['review_flags']);durations[r['split']]+=r['duration_s'];identities[r['pipeline_id']]+=1
    put('labels',r,raw);put('coverage',c)
    if counts['targets']%200000==0:print(json.dumps(dict(stage='export',counts=dict(counts))),flush=True)
   assert lf.read()==b''
  assert oldh.hexdigest()==OLD_SHA and oldch.hexdigest()==COVER_SHA
  assert replaced==set(success) and excluded==set(failed)|set(visual) and counts['labels']+counts['excluded']+counts['held']==1107323
  assert counts['orientation_restored']+counts['rotation_excluded']==1699 and counts['repeated_regenerated']+counts['repeated_excluded']==36
  for f in files.values():f.close()
  for n in names:(OUT/(n+'.jsonl.tmp')).replace(OUT/(n+'.jsonl'))
  summary=dict(schema='j-upright-unified-labels-v1-20261105',counts=dict(counts),splits=dict(splits),hours_by_split={k:v/3600 for k,v in durations.items()},review_flags=dict(flags),pipeline_counts=dict(identities),
   original_manifest=str(OLD/'labels.jsonl'),original_manifest_sha256=oldh.hexdigest(),original_coverage_sha256=oldch.hexdigest(),
   orientation_source_review=str(ORIENT/'orientation_source_decisions_final.json'),orientation_source_review_sha256=sha(ORIENT/'orientation_source_decisions_final.json'),
   confirmed_rotation_directions=str(WORK/'rotation_sources_confirmed.json'),confirmed_rotation_directions_sha256=sha(WORK/'rotation_sources_confirmed.json'),
   files={n+'.jsonl':dict(sha256=hashes[n].hexdigest(),bytes=(OUT/(n+'.jsonl')).stat().st_size) for n in names},
   policy=dict(keep_automatic_and_review=True,keep_upright_landscape=True,restore_sideways_then_exclude_unusable=True,all_repeated_generation_flags_regenerated=True,failed_retries_excluded=True,original_time_split_and_audio_preserved=True,per_row_caption_identity_required=True,behavior_statistics='not_recomputed_for_this_caption_version',training_and_test_selection='not_reselected'),elapsed_s=time.monotonic()-started)
  save(OUT/'EXPORT_SUMMARY.json',summary);print(json.dumps(dict(counts=dict(counts),splits=dict(splits),elapsed_s=summary['elapsed_s'])),flush=True)
 finally:
  for f in files.values():
   if not f.closed:f.close()

def audit():
 jobs,success,failed,visual,repeated,rotated=context();started=time.monotonic();summary=read(OUT/'EXPORT_SUMMARY.json');counts=Counter();seen=set();groups={};originalhash=hashlib.sha256();coveragehash=hashlib.sha256();special={}
 reviewed=read(ORIENT/'orientation_source_decisions_final.json')['records'];upright={s['source_key'] for s in reviewed if s['decision']=='keep_upright_source'};upright_seen=Counter();sideways={s['source_key'] for s in reviewed if s['decision']=='exclude_sideways_source'}
 R25='naturalistic_V03_S0984_I00000046_local_63463d08404f'
 with (OLD/'coverage.jsonl').open('rb') as oc,(OLD/'labels.jsonl').open('rb') as ol,(OUT/'coverage.jsonl').open('rb') as nc,(OUT/'labels.jsonl').open('rb') as nl,(OUT/'excluded.jsonl').open('rb') as xf:
  for ocraw in oc:
   coveragehash.update(ocraw);oldc=orjson.loads(ocraw);c=orjson.loads(next(nc));sid=oldc['sample_id'];assert sid==c['sample_id'] and sid not in seen;seen.add(sid);counts['targets']+=1
   for f in ['pair','split','duration_s']:assert c[f]==oldc[f]
   if oldc['status']=='held':assert oldc==c;counts['held']+=1;continue
   oldraw=next(ol);originalhash.update(oldraw);o=orjson.loads(oldraw);assert o['sample_id']==sid
   if sid in failed or sid in visual:
    x=orjson.loads(next(xf));assert x['sample_id']==sid and c['status']=='excluded';counts['excluded']+=1
    assert x['caption_file_sha256']==o['caption_file_sha256'] and sha(x['diagnostic_file'])==x['diagnostic_sha256']
    if sid in repeated or sid==R25:special[sid]='excluded'
    continue
   raw=next(nl);r=orjson.loads(raw);assert r['sample_id']==sid and c['status']=='caption'
   if sid in success:
    p,d=success[sid];expected=replace_row(o,p,d,sid in rotated);assert r==expected;counts['regenerated']+=1
    if sid in repeated:assert 'repeated_generation' not in r['review_flags']
    if sid in rotated:
     counts['orientation_restored']+=1;prov=read(r['orientation_restoration']['provenance_file']);s=r['source'];oldsrc=o['source']
     assert prov['original_source']==oldsrc and prov['rotation_ccw_degrees'] in [-90,90]
     assert sha(s['source']['archive'])==prov['derived_archive_sha256']==r['orientation_restoration']['derived_archive_sha256']
     assert s['source']!=oldsrc['source'] and sourcekey(o) in sideways
     assert prov['spatial_resize'] is False and prov['timestamps_preserved'] and prov['external_audio_recipe_unchanged']
     assert abs(prov['first_frame_s']-o['source_start_s'])<.1 and 0<o['source_end_s']-prov['last_frame_s']<.1
     assert prov['representative_pixel_check']['mean_abs_error']<1.5 and prov['representative_pixel_check']['max_abs_error']<=6
     for f in ['source_start_s','source_end_s','duration_s','pair','listener_id','speaker_id','group_id','split','normalized_audio']:assert s[f]==oldsrc[f]
     detector=d['input_evidence']['nod_supplement'];assert sha(detector['detector_file'])==detector['detector_sha256'];dr=read(detector['detector_file']);assert dr['source']==s and dr['status']=='ok'
    else:assert r['source']==o['source'];counts['unrotated_repeated_regenerated']+=1
   else:
    assert raw==oldraw;assert sourcekey(o) not in sideways;counts['byte_identical_original_rows']+=1
   assert r['audio_recipe']==o['audio_recipe'] and r['listener_audio_is_silent']==o['listener_audio_is_silent']
   assert r['finish_reason']=='stop' and r['caption'].strip() and r['training_admitted'] is False
   for f in ['caption_file','caption_file_sha256','generation_checks_passed']:assert c[f]==r[f]
   assert groups.setdefault(r['group_id'],r['split'])==r['split'];counts['labels']+=1
   if sourcekey(o) in upright:upright_seen[sourcekey(o)]+=1
   if sid in repeated or sid==R25:special[sid]='restored_and_regenerated' if sid in rotated else 'regenerated'
   if counts['targets']%200000==0:print(json.dumps(dict(stage='audit',counts=dict(counts))),flush=True)
  assert oc.read()==ol.read()==nc.read()==nl.read()==xf.read()==b''
 assert originalhash.hexdigest()==OLD_SHA and coveragehash.hexdigest()==COVER_SHA
 assert counts['targets']==1107323 and counts['labels']==summary['counts']['labels'] and counts['excluded']==summary['counts']['excluded']
 assert repeated|{R25}==set(special) and set(upright_seen)==upright
 assert (OUT/'exceptions.jsonl').read_bytes()==(OLD/'exceptions.jsonl').read_bytes()
 checked={}
 for n,m in summary['files'].items():
  h=sha(OUT/n);assert h==m['sha256'] and (OUT/n).stat().st_size==m['bytes'];checked[n]=h
 for p,h in [(summary['orientation_source_review'],summary['orientation_source_review_sha256']),(summary['confirmed_rotation_directions'],summary['confirmed_rotation_directions_sha256'])]:assert sha(p)==h
 result=dict(status='passed',counts=dict(counts),duplicate_ids=0,omitted_targets=0,group_cross_split=0,unrestored_sideways_retained=0,repeated_original_caption_retained=0,held_old_label_backfills=0,upright_reviewed_sources_preserved=len(upright_seen),upright_reviewed_clips_preserved=sum(upright_seen.values()),special_samples=special,output_sha256=checked,elapsed_s=time.monotonic()-started)
 save(OUT/'UPRIGHT_AUDIT.json',result);save(OUT/'EXPORT_COMPLETE.json',dict(status='complete',completed_at=time.time(),summary_sha256=sha(OUT/'EXPORT_SUMMARY.json'),audit_sha256=sha(OUT/'UPRIGHT_AUDIT.json')));print(json.dumps(dict(status='passed',counts=dict(counts),elapsed_s=result['elapsed_s'])),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['export','audit']);a=p.parse_args();(export if a.command=='export' else audit)()
