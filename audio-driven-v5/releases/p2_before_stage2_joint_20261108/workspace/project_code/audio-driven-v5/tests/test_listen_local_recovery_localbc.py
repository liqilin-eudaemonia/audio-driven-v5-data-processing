import copy
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/pair_boundary_real_audit_20261029'
sys.path[:0]=[str(ROOT/'scripts'),str(EXP)]
import listen_local_recovery_localbc as local
import rerun_fixed_pairs as run

class LocalFeedbackTests(unittest.TestCase):
    def source(self):
        return json.loads((EXP/(run.PAIRS[0]+'.json')).read_text())['people']

    def test_isolated_feedback_reclassified_without_losing_vad_tails(self):
        people=self.source();snapshot=copy.deepcopy(people);new=local.prepare_people(people)
        self.assertEqual(people,snapshot)
        child=next(s for s in new[0]['segments'] if s['id']=='0_000.local2')
        self.assertEqual(child['text'],'yeah');self.assertTrue(child['feedback'])
        self.assertFalse(next(s for s in new[0]['segments'] if s['id']=='0_000.local1')['feedback'])
        self.assertTrue(any(a<=4.2<b for a,b in local.semantic_blocks(new[0])))
        self.assertFalse(any(a<=10.3<b for a,b in local.semantic_blocks(new[0])))
        for before,after in zip(people,new):
            sig=lambda p:[(w['start'],w['end'],w['text']) for w in p['words']]
            self.assertEqual(sig(before),sig(after));self.assertEqual(before['vad'],after['vad'])

    def test_uncorroborated_pause_and_invalid_alignment_stay_unchanged(self):
        for mode in ['vad','alignment']:
            people=self.source()
            if mode=='vad':people[0]['vad'].append([4.5,10.0])
            else:people[0]['segments'][0]['word_aligned']=False
            new=local.prepare_people(people)
            self.assertEqual(new[0]['segments'][0],people[0]['segments'][0])
            self.assertFalse(new[0]['local_feedback_audit'])

    def test_feedback_word_with_following_content_is_not_promoted(self):
        people=self.source();s=people[0]['segments'][0]
        last=s['words'][-1]
        s['words'].append(dict(last,start=10.45,end=10.65,text='disagree'))
        s['end']=10.65;s['text']+=' disagree';s['word_count']+=1
        new=local.prepare_people(people)
        child=next(s for s in new[0]['segments'] if s['id']=='0_000.local2')
        self.assertEqual(child['text'],'yeah disagree');self.assertFalse(child['feedback'])

    def test_true_terminal_outranks_inferred_clause(self):
        self.assertLess(local.end_priority('sentence_terminal',False),local.end_priority('inferred_clause_end',True))

    def test_real_targets_and_previously_retained_words(self):
        for pair in run.PAIRS:
            ctx=run.context(pair);ctx.__class__=local.SourceContext
            ctx.people=local.prepare_people(ctx.people)
            ctx.regions,ctx.source_gaps=local.role_regions(ctx.people,ctx.limit,[])
            clips,_=ctx.propose([])
            old=json.loads((EXP/'gap5_pairs'/pair/'initial.json').read_text())['clips']
            covers=lambda w,rs,si:any(r['speaker_index']==si and r['start']<=w['start']+1e-6 and r['end']>=w['end']-1e-6 for r in rs)
            for si,p in enumerate(ctx.people):
                for w in p['words']:
                    if covers(w,old,si):self.assertTrue(covers(w,clips,si),w)
                    self.assertFalse(any(w['start']+1e-6<t<w['end']-1e-6 for c in clips for t in [c['start'],c['end']]))
            self.assertTrue(all(2-1e-6<=c['duration_s']<=11.95+1e-6 for c in clips))
            self.assertTrue(all(a['end']<=b['start']+1e-6 for a,b in zip(clips,clips[1:])))
            if pair==run.PAIRS[0]:
                first=next(c for c in clips if 'San Diego?' in c['text'])
                self.assertTrue(any(e['text']=='yeah' and e['label']=='feedback_candidate' for e in first['listener_events']))
                whole=next(c for c in clips if c['text'].startswith('During the time'))
                self.assertIn('I find the therapist',whole['text'])
                self.assertTrue(whole['text'].endswith('marry her.'))
                self.assertIn('insufficient_tail_room',whole['boundary_review_flags'])
                self.assertTrue(any(c['start']<=184.921 and c['end']>=186.642 for c in clips))
                self.assertTrue(any('It was crazy' in c['text'] for c in clips))
            else:
                self.assertTrue(any(c['start']<=335.787 and c['end']>=336.355 for c in clips))

if __name__=='__main__':unittest.main()
