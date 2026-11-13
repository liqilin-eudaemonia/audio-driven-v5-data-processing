"""Real source-contract and transformation checks before four-machine use."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_j_upright_av500h_v3 as m
from listen_richness.production import tokenized,contract
from listen_richness.statistics import snapshot

@pytest.fixture(scope='module')
def cases():
 rows=[]
 with (m.LABELS/'labels.jsonl').open('rb') as f:
  rows.append(m.orjson.loads(f.readline()))
  for b in f:
   row=m.orjson.loads(b)
   if row['review_flags']:
    rows.append(row);break
 from export_j_upright_labels import replace_row
 sid='naturalistic_V03_S0984_I00000046_local_63463d08404f';w=ROOT/'experiments/listen_j_upright_regen_20261105';rr=m.read(w/'restored_rows'/f'{sid}.json');p=w/'regenerated'/f'{sid}.json'
 rows.append(replace_row(rr['original_manifest_row'],p,m.read(p),True))
 return rows

def test_original_pass_matches_old_snapshot(cases):
 row=cases[0];r=m.read(row['caption_file']);old=snapshot(r,row['caption_file'],row['caption_file_sha256'],row['quality'],sample_grid_tolerance=True);new=m.from_manifest(row)
 for k in ('source','caption','face','audio','nod','quality','crop_provenance'):assert old[k]==new[k]

def test_ordinary_review_kept_with_evidence(cases):
 row=cases[1];assert row['review_flags'];new=m.from_manifest(row);assert new['eligible_for_extraction'];assert new['source_caption_review_flags']==row['review_flags'];assert new['audio']['waveform_full_silence']==row['listener_audio_is_silent']

def test_upright_multihop_source_keeps_original(cases):
 row=cases[2];before=copy.deepcopy(row);r=m.read(row['caption_file']);new=m.from_manifest(row)
 assert row==before and r==m.read(row['caption_file']);assert new['source']==row['source'];assert new['orientation_lineage']['original_source_sha256']==m.digest(r['original_source']);assert new['audio']['official_core_s']>0

@pytest.mark.parametrize('key,value',[('caption_file_sha256','0'*64),('caption','silently changed'),('face_p10',0),('listener_audio_is_silent',True)])
def test_bad_manifest_or_evidence_rejected(cases,key,value):
 row=copy.deepcopy(cases[0]);row[key]=value
 with pytest.raises(ValueError):m.from_manifest(row)

def test_wrong_restoration_identity_rejected(cases):
 row=copy.deepcopy(cases[2]);row['orientation_restoration']['derived_archive_sha256']='0'*64
 with pytest.raises(ValueError):m.from_manifest(row)

def test_batched_tokens_equal_unchanged_single_templates(cases):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(contract()['model'],local_files_only=True)
 snaps=[m.from_manifest(r) for r in cases]
 batch=tok.apply_chat_template([m.messages(s['caption']) for s in snaps],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 for s,ids in zip(snaps,batch):assert ids==tokenized(s,tok)['prompt_token_ids']
