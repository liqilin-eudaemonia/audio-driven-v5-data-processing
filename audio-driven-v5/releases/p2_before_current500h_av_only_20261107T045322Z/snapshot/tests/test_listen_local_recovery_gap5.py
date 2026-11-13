import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/pair_boundary_real_audit_20261029'
sys.path[:0]=[str(ROOT/'scripts'),str(EXP)]
import listen_local_recovery_gap5 as gap5
import rerun_fixed_pairs as run

def word(a,b,text='word'):return dict(start=a,end=b,text=text,feedback=False)

class GapFiveTests(unittest.TestCase):
    def test_exactly_five_stays_joined_and_more_than_five_splits(self):
        for distance,count in [(3.44,1),(5.,1),(5.001,2)]:
            people=[dict(words=[word(1,1.5),word(1.5+distance,2+distance)],segments=[]),dict(words=[],segments=[])]
            regions,_=gap5.role_regions(people,10)
            self.assertEqual(len(regions),count)
            self.assertEqual(len(gap5.sentence_units(people[0]['words'])),count)

    def test_gap_does_not_bridge_other_speaker_or_shot(self):
        people=[dict(words=[word(1,2),word(6,7)],segments=[]),dict(words=[word(4,4.5)],segments=[])]
        regions,_=gap5.role_regions(people,10)
        self.assertEqual([r['speaker_index'] for r in regions],[0,1,0])
        people[1]['words']=[]
        regions,_=gap5.role_regions(people,10,[4])
        self.assertFalse(any(r['start']<4<r['end'] for r in regions))

    def test_real_plan_matches_trial_and_preserves_prior_repairs(self):
        trial=json.loads((EXP/'gap5_trial/segmentation.json').read_text())
        for expected in trial['pairs']:
            ctx=run.context(expected['pair']);ctx.__class__=gap5.SourceContext
            ctx.regions,ctx.source_gaps=gap5.role_regions(ctx.people,ctx.limit)
            clips,_=ctx.propose([])
            signature=lambda rows:[(r['speaker_index'],r['start'],r['end'],r['text']) for r in rows]
            self.assertEqual(signature(clips),signature(expected['clips']))
            self.assertTrue(all(2-1e-6<=r['duration_s']<=11.95+1e-6 for r in clips))
            self.assertTrue(all(a['end']<=b['start']+1e-6 for a,b in zip(clips,clips[1:])))
            if '3579' in expected['pair']:
                self.assertTrue(any('It was crazy' in r['text'] for r in clips))
                targets=[(184.921,186.642)]
            else:targets=[(335.787,335.928),(336.115,336.355)]
            for a,b in targets:self.assertTrue(any(r['start']<=a and r['end']>=b for r in clips))

if __name__=='__main__':unittest.main()
