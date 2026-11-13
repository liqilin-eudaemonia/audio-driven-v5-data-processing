import unittest,tempfile,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import eval_common as common
import eval_chain as chain

class SafetyContracts(unittest.TestCase):
    def test_partial_checkpoint_is_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'step_000200.pt';p.write_bytes(b'complete bytes')
            self.assertFalse(common.ready(p))
            common.write(p.with_suffix('.complete.json'),{'sha256':'wrong'})
            self.assertFalse(common.ready(p))
            common.write(p.with_suffix('.complete.json'),{'sha256':common.sha(p),'step':200})
            self.assertTrue(common.ready(p));p.write_bytes(b'changed bytes')
            self.assertFalse(common.ready(p))

    def test_yields_for_stage_reload_and_lora(self):
        for stage,step,want in [(1,200,True),(1,1100,False),(1,1249,False),(2,1,False)]:
            state={'status':'running','jobs':{'x':{'status':'running','stage':stage,'step':step}}}
            with patch.object(chain,'read',return_value=state):self.assertEqual(chain.training_allows(),want)
        with patch.object(chain,'read',return_value={'status':'running','jobs':{}}):self.assertFalse(chain.training_allows())

    def test_membership_and_media_are_preserved(self):
        e=common.HERE;s=common.read(e/'sampling.json')
        old=common.read(common.ROOT/'experiments/p2_train24_fourway_eval_20261027/sampling.json')['sample_ids']
        new=s['train_sample_ids'];self.assertEqual(len(new),24);self.assertEqual(len(set(new)&set(old)),19)
        self.assertEqual(len(s['replacements']),5);self.assertFalse(set(new)&set(s['test_sample_ids']))
        self.assertEqual(common.read(e/'test24/observations.json'),common.read(common.ROOT/'experiments/p2_test24_fourway_eval_20261027/observations.json'))

    def test_english_rendering_preserves_original_source_and_times(self):
        common.english_extension()
        from p2_guided_prompt_en.formatting import render,is_english_script
        records=[json.loads(l) for l in (common.HERE/'train24/integrated_labels.jsonl').read_text().splitlines()]
        for r in records:
            g=render(r);self.assertEqual(g['source_record'],r);self.assertTrue(is_english_script(g['prompt']))
            originals={e['event_id']:e for e in r['events']}
            for event in g['entries']:
                src=originals[event['source_event_id']]
                spans=src.get('intervals',[{'start':src['start'],'end':src['end']}])
                self.assertIn({'start':event['start'],'end':event['end']},spans)

    def test_live_training_source_unchanged(self):
        frozen=common.read(common.TRAIN/'training_queue/frozen.json')
        self.assertTrue(all(common.sha(p)==h for p,h in frozen.items()))

    def test_plan_reader_accepts_english_trace_without_changing_labels(self):
        from p2_listen.contracts import digest
        from p2_plan_cache.protocol import sample
        obs=common.read(common.HERE/'train24/observations.json')
        records={r['sample_id']:r for r in map(json.loads,(common.HERE/'train24/integrated_labels.jsonl').read_text().splitlines())}
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)
            for o in obs:
                sid=o['sample_id'];item=common.guidance_item(o,records[sid])
                common.write(work/'samples'/(sid+'.json'),item)
                self.assertEqual(sample(work,{'entries':{sid:digest(item)}},sid),item)
                self.assertEqual(item['original_record'],records[sid])

    def test_exact_user_scope_same_original_test24_and_final_checkpoints(self):
        jobs=common.read(common.HERE/'jobs.test24.json');self.assertEqual(len(jobs),336)
        legacy=[j for j in jobs if j['family']=='legacy'];new=[j for j in jobs if j['family']=='matrix']
        ids={o['sample_id'] for o in common.read(common.HERE/'test24/observations.json')}
        self.assertEqual(len(legacy),192);self.assertEqual(len(new),144)
        self.assertEqual({j['sample_id'] for j in jobs},ids)
        self.assertEqual({j['split'] for j in jobs},{'test24'})
        for arm in {j['arm'] for j in legacy}:
            rows=[j for j in legacy if j['arm']==arm];self.assertEqual(len(rows),48)
            self.assertEqual({j['sample_id'] for j in rows},ids)
        for arm in {j['arm'] for j in new}:
            rows=[j for j in new if j['arm']==arm];self.assertEqual(len(rows),48)
            self.assertEqual({j['sample_id'] for j in rows},ids)
            self.assertTrue(all(Path(j['checkpoint']).name=='step_001249.pt' for j in rows))
        self.assertFalse(any('step200' in j['arm'] for j in jobs))

if __name__=='__main__':unittest.main()
