"""Versioned migration to published upright sources; preserve selection and historical runs."""
import json,sys,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,rows
from j_upright_response_sft_data import Reader,resolve
from evaluation.seamless.reader import upright_reader_contract
from evaluation.seamless.bind import load_references
WORK=ROOT/'experiments/seamless_upright_eval_fix_20261112'
OLD=ROOT/'experiments/seamless_quick200_latest_20261105'
POOL=ROOT/'experiments/listen_j_labels_upright_20261105'

def migrate(parent,out):
    refs=load_references(parent);replacements={r['sample_id']:r for r in rows(POOL/'replacements.jsonl') if r['orientation_restored']}
    excluded={r['sample_id'] for r in rows(POOL/'excluded.jsonl')}
    assert not excluded.intersection(r['id'] for r in refs),'Excluded source requires explicit selection revision'
    reader=Reader(ROOT/'experiments/listen_j_upright_response_sft_20261105/source')
    out.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/'crop_config.json',out/'crop_config.json')
    changed=[];inputs=[];annotations=[];new=[]
    with (out/'source.test.jsonl').open('wb') as target:
        for old in refs:
            sid=old['id'];g=old['gt_reference'];r=dict(old);ref=dict(g)
            if sid in replacements:
                row=reader.row(sid);assert row['split']=='test';item=resolve(row);replacement=replacements[sid]
                assert row['target']['source_file']==replacement['caption_file'] and row['target']['source_sha256']==replacement['caption_file_sha256']
                assert item['source']['source_start_s']==old['source_start'] and item['source']['source_end_s']==old['source_end']
                raw=(json.dumps(row,ensure_ascii=False)+'\n').encode();ref['upright_reader_contract']=upright_reader_contract()
                r['latest_annotation']=dict(caption=row['target']['text'],caption_file=row['target']['source_file'],caption_sha256=row['target']['source_sha256'],human_ground_truth=False)
                cap=read(row['target']['source_file']);changed.append(dict(id=sid,old_reference=g,new_source=item['source'],replacement=replacement))
                r['orientation_restored']=True
            else:
                with open(g['manifest'],'rb') as f:f.seek(g['offset']);raw=f.read(g['bytes'])
                assert hashlib.sha256(raw).hexdigest()==g['row_sha256'];row=json.loads(raw)
            ref.update(manifest=str(out/'source.test.jsonl'),offset=target.tell(),bytes=len(raw),row_sha256=hashlib.sha256(raw).hexdigest(),crop_config=str(out/'crop_config.json'))
            target.write(raw);r['gt_reference']=ref;new.append(r)
            inputs.append(dict(id=sid,inputs=row['inputs'],gt_reference=ref,conditioning='speaker_audio_and_listener_first_frame_only'))
            annotations.append(dict(id=sid,**r.get('latest_annotation',{})))
    for name,values in [('references',new),('inference_inputs',inputs),('annotations',annotations)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    if (parent/'generation_map.template.jsonl').exists():shutil.copyfile(parent/'generation_map.template.jsonl',out/'generation_map.template.jsonl')
    save(out/'seal.json',dict(schema='seamless-upright-migration-v1',dataset='seamless',expected_clips=len(new),parent_prepared=str(parent),parent_seal_sha256=sha(parent/'seal.json'),references_sha256=sha(out/'references.jsonl'),source_sha256=sha(out/'source.test.jsonl'),crop_config_sha256=sha(out/'crop_config.json'),replaced_ids=[x['id'] for x in changed],replacement_registry=str(POOL/'replacements.jsonl'),replacement_registry_sha256=sha(POOL/'replacements.jsonl')))
    assert len(load_references(out))==len(new)
    save(out/'MIGRATION.json',dict(samples=len(new),changed=changed,unchanged=len(new)-len(changed),historical_inputs_untouched=True))
    return dict(prepared=str(out),samples=len(new),replaced_ids=[x['id'] for x in changed])

if __name__=='__main__':
    result={}
    for name,parent in [('prepared_200',OLD/'prepared_200'),('prepared_core48',OLD/'prepared_core48'),('prepared_full6624',ROOT.parent/'OmniResponse/evaluation/seamless/prepared_20261102')]:
        result[name]=migrate(parent,WORK/name);print(name,result[name]['samples'],len(result[name]['replaced_ids']),flush=True)
    save(WORK/'MIGRATION_COMPLETE.json',result)
