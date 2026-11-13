import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'experiments/pair_boundary_real_audit_20261029')]
import rerun_fixed_pairs as runner
import listen_local_recovery_semantic_trial as trial

class SemanticTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx={};cls.rows={};cls.diag={};cls.saved={}
        before=runner.EXP,runner.repair
        try:
            runner.EXP=ROOT/'experiments/pair_head_seven_20261029';runner.repair=trial
            for pair in ['naturalistic_V02_S3610_I00000088','naturalistic_V02_S3579_I00000175']:
                ctx=runner.context(pair);cls.ctx[pair]=ctx
                cls.rows[pair],cls.diag[pair]=ctx.propose([])
                cls.saved[pair]=json.loads((runner.EXP/'results'/pair/'result.json').read_text())
        finally:runner.EXP,runner.repair=before
        cls.first='naturalistic_V02_S3610_I00000088';cls.second='naturalistic_V02_S3579_I00000175'

    def test_sentence_tail_does_not_append_next_sentence_prefix(self):
        for parent_start,end,text in [(140.652,145.876,"He's mad."),(148.43,153.4,"he ain't about it.")]:
            parent=copy.deepcopy(next(r for r in self.saved[self.first]['superseded_attempts'] if r['start']==parent_start))
            children,_=self.ctx[self.first].propose([parent],domain=parent)
            self.assertEqual(len(children),1)
            self.assertEqual(children[0]['end'],end)
            self.assertTrue(children[0]['text'].endswith(text))
            self.assertEqual(children[0]['semantic_end'],'sentence_terminal')

    def test_soft_pause_restores_long_sentence_and_records_unpaired_prefix(self):
        rows=self.rows[self.first]
        self.assertTrue(any(r['start']==25.16 and r['end']==35.498 and 'Mufasa movie theater' in r['text'] for r in rows))
        self.assertFalse(any(r['start']==21.34 and r['end']==31.35 for r in rows))
        self.assertTrue(any(f.get('text')=='Man,' and 'not admitted' in f.get('note','') for f in self.diag[self.first]['unresolved']))
        self.assertEqual(trial.MAX_WORD_GAP,5)

    def test_existing_complete_repairs_preserved(self):
        rows=self.rows[self.second]
        for start,end,text in [(5.743,11.7,'San Diego'),(149.903,161.853,'marry her.'),(176.179,186.78,"So I'm looking for a new job right now.")]:
            self.assertTrue(any(r['start']==start and r['end']==end and text in r['text'] for r in rows))

    def test_unsafe_quality_recovery_preserves_parent_without_clearing_quality_hold(self):
        parent=copy.deepcopy(next(r for r in self.saved[self.second]['superseded_attempts'] if r['start']==134.119))
        original=copy.deepcopy(parent['audio_final']);children,diag=self.ctx[self.second].propose([parent],domain=parent)
        self.assertEqual(children,[]);self.assertEqual(parent['audio_final'],original)
        self.assertIn('semantic_recovery_hold',parent)
        self.assertTrue(any('because the' in r['text'] for r in parent['semantic_recovery_hold']['proposed_children']))

    def test_initial_intervals_respect_duration_words_and_nonoverlap(self):
        for pair,rows in self.rows.items():
            self.assertTrue(all(2-1e-6<=r['duration_s']<=11.95+1e-6 for r in rows))
            self.assertTrue(all(a['end']<=b['start']+1e-6 for a,b in zip(rows,rows[1:])))
            for person in self.ctx[pair].people:
                for word in person['words']:
                    self.assertFalse(any(word['start']+1e-6<edge<word['end']-1e-6 for r in rows for edge in [r['start'],r['end']]))

if __name__=='__main__':unittest.main()
