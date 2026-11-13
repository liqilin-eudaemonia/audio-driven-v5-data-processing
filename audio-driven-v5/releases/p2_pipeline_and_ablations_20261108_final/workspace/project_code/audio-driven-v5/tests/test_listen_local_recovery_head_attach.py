import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments/pair_boundary_real_audit_20261029'),str(ROOT/'scripts')]
import rerun_fixed_pairs as runner
import numpy as np
import listen_local_recovery_head_attach as head
import listen_local_recovery_localbc as prior

class HeadAttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contexts={};cls.rows={}
        pairs=runner.PAIRS+[r['pair'] for r in json.loads((ROOT/'experiments/pair_boundary_holdout3_20261029/selection.json').read_text())['items']]
        for pair in pairs:
            saved=(runner.EXP,runner.repair)
            try:
                runner.EXP=ROOT/'experiments/pair_head_attach5_20261029';runner.repair=head
                ctx=runner.context(pair)
            finally:runner.EXP,runner.repair=saved
            rows,_=prior.SourceContext.propose(ctx,[])
            cls.contexts[pair]=ctx;cls.rows[pair]=rows
        cls.target=pairs[-1]

    def fixture(self):
        ctx=copy.deepcopy(self.contexts[self.target]);rows=copy.deepcopy(self.rows[self.target])
        event=next(s for s in ctx.people[1]['segments'] if s['text']=='Okay.' and 213<s['start']<214)
        return ctx,rows,event

    def target_changed(self,ctx,rows,protected=()):
        out,audit=head.attach_heads(ctx,rows,protected)
        return any(r.get('head_attachment',{}).get('feedback_text')=='Okay.' for r in out),audit

    def test_real_five_sources_only_three_heads_move_and_no_words_lost(self):
        count=0
        for pair,ctx in self.contexts.items():
            original=self.rows[pair];snapshot=copy.deepcopy(original)
            out,audit=head.attach_heads(ctx,original)
            self.assertEqual(original,snapshot)
            self.assertEqual(len(out),len(original))
            self.assertEqual([r['end'] for r in out],[r['end'] for r in original])
            for before,after in zip(original,out):
                self.assertEqual(before['speaker_index'],after['speaker_index'])
                self.assertLessEqual(after['start'],before['start']+1e-6)
            changed=[r for r in out if r.get('head_attachment')];count+=len(changed)
            if pair==self.target:self.assertEqual([r['head_attachment']['feedback_text'] for r in changed],['Right.','Yeah.','Okay.'])
            else:self.assertFalse(changed)
            self.assertTrue(all(2-1e-6<=r['duration_s']<=11.95+1e-6 for r in out))
            self.assertTrue(all(a['end']<=b['start']+1e-6 for a,b in zip(out,out[1:])))
            for p in ctx.people:
                for w in p['words']:
                    covered=lambda rs:any(r['start']<=w['start']+1e-6 and r['end']>=w['end']-1e-6 for r in rs)
                    if covered(original):self.assertTrue(covered(out))
                    self.assertFalse(any(w['start']+1e-6<t<w['end']-1e-6 for r in out for t in [r['start'],r['end']]))
        self.assertEqual(count,3)

    def test_already_partly_covered_feedback_is_not_duplicated(self):
        ctx,rows,e=self.fixture()
        rows.append(dict(start=e['start'],end=e['start']+.1,speaker_index=0,speech_start=e['start'],sample_id='existing'))
        rows.sort(key=lambda r:r['start'])
        self.assertFalse(self.target_changed(ctx,rows)[0])

    def test_other_speaker_interruption_and_quality_barrier(self):
        ctx,rows,e=self.fixture()
        self.assertFalse(self.target_changed(ctx,rows,[(e['start'],e['end'])])[0])
        ctx.people[0]['words'].append(dict(start=213.97,end=213.98,text='But',feedback=False))
        self.assertFalse(self.target_changed(ctx,rows)[0])

    def test_shot_boundary_and_uncertain_alignment_are_not_crossed(self):
        for mode in ['shot','uncertain','unaligned','gap','duration']:
            ctx,rows,e=self.fixture()
            if mode=='shot':ctx.shots.append(213.97)
            elif mode=='uncertain':e['long_feedback_timing_uncertain']=True
            elif mode=='unaligned':e['word_aligned']=False
            elif mode=='gap':
                r=next(r for r in rows if 213<r['start']<214);r['speech_start']=e['end']+.301
            else:
                r=next(r for r in rows if 213<r['start']<214);r['end']=225.3;r['speech_end']=225.3
                ctx.people[1]['words'].append(dict(start=225.0,end=225.3,text='end.',feedback=False))
                ctx.people[1]['words'].sort(key=lambda w:w['start'])
                rows=[x for x in rows if x is r or x['end']<=r['start'] or x['start']>=r['end']]
            self.assertFalse(self.target_changed(ctx,rows)[0],mode)

    def test_failed_prefix_quality_restores_full_original_candidate(self):
        ctx,rows,_=self.fixture();out,_=head.attach_heads(ctx,rows)
        plan=next(r for r in out if r.get('head_attachment',{}).get('feedback_text')=='Right.')
        original=next(r for r in rows if r['end']==plan['end'])
        restored=head.quality_fallback(plan,{'audio_final':{'pool':'audio_review'}})[0]
        self.assertEqual((restored['start'],restored['end'],restored['text']),
                         (original['start'],original['end'],original['text']))
        self.assertIsNone(head.quality_fallback(plan,{'audio_final':{'pool':'keep_candidate'}}))

    def test_local_quality_recovery_does_not_expand_parent_domain(self):
        ctx,rows,_=self.fixture();r=next(r for r in rows if 213<r['start']<214)
        out,d=ctx.propose([],domain=r)
        self.assertNotIn('head_attachment',d)
        self.assertTrue(all(x['start']>=r['start'] and x['end']<=r['end'] for x in out))

if __name__=='__main__':unittest.main()
