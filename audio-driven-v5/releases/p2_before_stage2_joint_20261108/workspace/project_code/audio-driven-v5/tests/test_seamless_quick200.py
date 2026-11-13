import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
spec=importlib.util.spec_from_file_location('seamless_selection',ROOT/'scripts/select_seamless_quick200.py')
selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
DATA=ROOT/'experiments/seamless_quick200_latest_20261105'


def test_frozen_membership_core_and_pair_diversity():
    from evaluation.seamless.bind import load_references
    full=load_references(DATA/'prepared_200');core=load_references(DATA/'prepared_core48')
    selected=selector.rows(DATA/'selection.jsonl')
    assert len(full)==200 and len(core)==48
    assert {r['id'] for r in core} < {r['id'] for r in full}
    assert len({r['pair'] for r in selected})==200
    assert all(r['split']=='test' for r in full)
    assert sum(r['face_p10_px']>=224 for r in full)>=180
    assert sum(r['audio_group']=='silent' for r in full)==64


def test_latest_original_quotes_and_selected_raw_source_rows():
    for r in selector.rows(DATA/'selection.jsonl'):
        for spans in r['selection_hints'].values():
            for span in spans:
                assert r['caption'][span['start']:span['end']]==span['quote']
        raw=Path(r['caption_file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==r['caption_sha256']
        assert json.loads(raw)['caption']==r['caption']
        ref=r['latest_manifest_reference']
        with open(ref['path'],'rb') as f:f.seek(ref['offset']);raw=f.read(ref['bytes'])
        assert hashlib.sha256(raw).hexdigest()==ref['row_sha256']
        assert json.loads(raw)['caption']==r['caption']


def test_geometry_tampering_rejected_even_if_subset_resealed(tmp_path):
    from evaluation.seamless.bind import load_references
    src=DATA/'prepared_200'
    for name in ('seal.json','references.jsonl','source.test.jsonl','crop_config.json'):
        shutil.copyfile(src/name,tmp_path/name)
    refs=selector.rows(tmp_path/'references.jsonl');refs[0]['frames']+=1
    (tmp_path/'references.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs))
    seal=json.loads((tmp_path/'seal.json').read_text());seal['references_sha256']=selector.sha(tmp_path/'references.jsonl')
    (tmp_path/'seal.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError,match='media identity or geometry'):load_references(tmp_path)


def test_negated_actions_are_not_positive_sampling_hints():
    hints=selector.text_hints('She does not nod. She then smiles and raises her hand.')
    assert 'head_nod' not in hints
    assert 'smile' in hints and 'hand_motion' in hints
