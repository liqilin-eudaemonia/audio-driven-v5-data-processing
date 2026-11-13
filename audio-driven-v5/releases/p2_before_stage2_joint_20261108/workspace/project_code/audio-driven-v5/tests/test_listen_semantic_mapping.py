import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from listen_benchmark.semantic_mapping import PROMPT, AUDIO_PROMPT, request_prompt, digest, make_items, validate_answer, enrich_sample, response_schema
from build_listen_semantic_seven_class import load_answers


def data(modality='visual',behavior='The head moves down and returns up.',raw=None):
    event=dict(event_id='event-0000',start=.2,end=.7,modality=modality,behavior=behavior,
               review_flags=['source_boundary_disagreement'])
    source=dict(id='candidate-0000',start=.2,end=.7,modality=modality,
                behavior=raw or behavior,source='qwen3_vl' if modality=='visual' else 'omni_audio')
    label=dict(sample_id='s1',duration_s=2,time_origin='clip_start_seconds',events=[event])
    trace=dict(source_candidates=[source],events={'event-0000':dict(signature='generic free description',
        source_candidate_ids=['candidate-0000'],boundary_method='source_interval')},excluded=[])
    item=make_items([label],{'samples':{'s1':trace}})[0]
    return label,trace,item


def answer(categories=None,uncertain=False):
    return dict(categories=['head_nod'] if categories is None else categories,uncertain=uncertain,
                reason='The description denotes a downward and upward head movement.',
                evidence_source_ids=['candidate-0000'])


def records(item,value):
    return {item['open_event_id']:dict(payload=item['payload'],answer=value,
        item_id=item['item_id'],response_file='responses/semantic-000.json')}


class SemanticMappingTests(unittest.TestCase):
    def test_paraphrase_uses_model_category_without_keyword_gate(self):
        label,trace,item=data()
        r=enrich_sample(label,trace,records(item,answer()))
        self.assertEqual(r['events'][0]['categories'],['head_nod'])
        self.assertEqual(r['events'][0]['behavior'],label['events'][0]['behavior'])
        self.assertEqual(r['events'][0]['review_flags'],['source_boundary_disagreement'])

    def test_source_language_kept_verbatim_and_input_not_mutated(self):
        label,trace,item=data(raw='The listener makes a slight nod while smiling, indicating engagement.')
        original=copy.deepcopy((label,trace,item))
        r=enrich_sample(label,trace,records(item,answer()))
        event=r['events'][0]
        for k,v in label['events'][0].items():self.assertEqual(event[k],v)
        self.assertEqual(event['source_descriptions'][0]['description'],trace['source_candidates'][0]['behavior'])
        self.assertEqual(event['semantic_evidence'][0]['quote'],trace['source_candidates'][0]['behavior'])
        self.assertEqual((label,trace,item),original)

    def test_unclassified_and_uncertain_events_remain_in_main_labels(self):
        for uncertain,status in [(False,'out_of_scope'),(True,'needs_review')]:
            label,trace,item=data(behavior='Twirls a pen.')
            r=enrich_sample(label,trace,records(item,answer([],uncertain)))
            self.assertEqual(len(r['events']),1)
            self.assertEqual(r['events'][0]['mapping_status'],status)
            self.assertEqual(r['events'][0]['behavior'],'Twirls a pen.')
            self.assertFalse(r['training_admitted']);self.assertFalse(r['K_train'])
            self.assertTrue(all(not spans for spans in r['coverage'].values()))

    def test_unknown_sources_cross_modality_and_output_mutations_rejected(self):
        _,_,item=data()
        bad=answer();bad['evidence_source_ids']=['invented']
        with self.assertRaises(ValueError):validate_answer(item['payload'],bad)
        bad=answer(['listener_vocalization'])
        with self.assertRaises(ValueError):validate_answer(item['payload'],bad)
        bad=answer();bad['start']=1
        with self.assertRaises(ValueError):validate_answer(item['payload'],bad)
        bad=answer();bad['behavior']='Rewritten'
        with self.assertRaises(ValueError):validate_answer(item['payload'],bad)

    def test_uncertainty_and_categories_are_structural_not_semantic_rules(self):
        _,_,item=data(behavior='Eyes widen.')
        # Model semantic judgment is accepted even when a legacy cue-count rule disagrees.
        self.assertEqual(validate_answer(item['payload'],answer(['surprise']))['categories'],['surprise'])
        for bad in [answer(['nod']),answer(['smile','head_nod']),answer(uncertain='false')]:
            with self.assertRaises(ValueError):validate_answer(item['payload'],bad)

    def test_two_asr_words_preserve_independent_boundaries(self):
        label,trace,item=data('audio','Says "Okay".','Okay')
        trace['source_candidates'][0]['source']='asr_aligner'
        trace['events']['event-0000']['boundary_method']='asr_word_anchor'
        label['events'][0]['text']='Okay'
        other=copy.deepcopy(label['events'][0]);other.update(event_id='event-0001',start=1.2,end=1.6)
        label['events'].append(other)
        source=copy.deepcopy(trace['source_candidates'][0]);source.update(id='candidate-0001',start=1.2,end=1.6)
        trace['source_candidates'].append(source)
        trace['events']['event-0001']=dict(trace['events']['event-0000'],source_candidate_ids=['candidate-0001'])
        items=make_items([label],{'samples':{'s1':trace}});all_answers={}
        for it in items:
            a=answer(['listener_vocalization']);a['evidence_source_ids']=['primary']
            all_answers.update(records(it,a))
        r=enrich_sample(label,trace,all_answers)
        self.assertEqual([(e['start'],e['end'],e['text']) for e in r['events']],[(.2,.7,'Okay'),(1.2,1.6,'Okay')])
        label['events'][1]['end']=1.8
        with self.assertRaisesRegex(ValueError,'ASR'):enrich_sample(label,trace,all_answers)

    def test_changed_primary_source_or_missing_response_rejected(self):
        label,trace,item=data();a=records(item,answer())
        a['event-0000']['payload']['primary_description']='Different input'
        with self.assertRaises(ValueError):enrich_sample(label,trace,a)
        with self.assertRaises(ValueError):enrich_sample(label,trace,{})

    def test_schema_has_no_description_or_timestamp_outputs(self):
        _,_,item=data();schema=response_schema({item['item_id']:item['payload']})
        fields=schema['properties'][item['item_id']]['properties']
        self.assertEqual(set(fields),{'categories','uncertain','reason','evidence_source_ids'})

    def test_audio_uses_own_definition_not_visual_instructions(self):
        _,_,item=data('audio','Says "Okay".','Okay')
        self.assertEqual(request_prompt({item['item_id']:item['payload']}),AUDIO_PROMPT)
        self.assertIn('neutral acknowledgments',AUDIO_PROMPT)
        self.assertNotEqual(AUDIO_PROMPT,PROMPT)

    def test_response_loading_requires_exact_prompt_job_and_inputs(self):
        _,_,item=data();payloads={item['item_id']:item['payload']}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'input').mkdir();(root/'responses').mkdir()
            (root/'manifest.json').write_text(json.dumps({'policy':{'prompt':PROMPT,'audio_prompt':AUDIO_PROMPT},'input_sha256':{}}))
            (root/'items.json').write_text(json.dumps([item]))
            (root/'jobs.json').write_text(json.dumps([{'job_id':'semantic-000','payloads':payloads,'payload_sha256':digest(payloads)}]))
            record=dict(job_id='semantic-000',payload_sha256=digest(payloads),prompt_sha256=digest(PROMPT),
                        finish_reason='stop',raw=json.dumps({item['item_id']:answer()}))
            path=root/'responses/semantic-000.json';path.write_text(json.dumps(record))
            answers,_,hashes=load_answers(root);self.assertIn('s1',answers);self.assertEqual(len(hashes),1)
            for field,value in [('finish_reason','length'),('prompt_sha256','stale'),('payload_sha256','wrong')]:
                path.write_text(json.dumps(dict(record,**{field:value})))
                with self.assertRaises(ValueError):load_answers(root)


if __name__=='__main__':unittest.main()
