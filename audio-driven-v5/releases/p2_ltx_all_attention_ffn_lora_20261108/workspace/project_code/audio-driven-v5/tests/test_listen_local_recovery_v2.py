import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import listen_local_recovery_v2 as r

def word(a,b,feedback=False):return dict(start=a,end=b,text='word.',feedback=feedback)
def covers(rows,a,b):return any(c['start']<=a+1e-6 and c['end']>=b-1e-6 for c in rows)

class BoundaryRepairTests(unittest.TestCase):
    def people(self,next_start=7):
        return [dict(words=[word(1,2),word(2.1,3)],segments=[]),dict(words=[word(3.1,3.5,True),word(next_start,next_start+.5)],
            segments=[dict(start=3.1,end=3.5,feedback=True)])]

    def test_feedback_tail_is_completed_without_overlapping_next_role(self):
        regions,_=r.role_regions(self.people(),10)
        self.assertAlmostEqual(regions[0]['end'],3.8)
        self.assertLessEqual(regions[0]['end'],regions[1]['start'])

    def test_tail_extension_never_crosses_shot_or_next_role(self):
        for shots,people in [([3.4],self.people()),([],self.people(3.4))]:
            regions,_=r.role_regions(people,10,shots)
            self.assertLess(regions[0]['end'],3.5)

    def test_uncertain_or_late_feedback_does_not_expand_role(self):
        for update in [dict(long_feedback_timing_uncertain=True),dict(start=3.25)]:
            people=self.people();people[1]['segments'][0].update(update)
            regions,_=r.role_regions(people,10)
            self.assertAlmostEqual(regions[0]['end'],3.3)

    def test_real_cases_retain_omitted_sentence_and_response(self):
        exp=ROOT/'experiments/pair_boundary_real_audit_20261029'
        for n in [1,2]:
            evidence=json.loads((exp/'review'/f'case{n}_evidence.json').read_text())
            d=json.loads((exp/(evidence['pair']+'.json')).read_text());f=evidence['finding']
            a,b=f['domain'];si=f['speaker_index'];rms=[np.array(x) for x in d['rms']]
            original,_=r.base.recover_region(d['people'],rms,si,a,b,d['limit'])
            if n==2:
                regions,_=r.role_regions(d['people'],d['limit'])
                b=next(c['end'] for c in regions if abs(c['start']-a)<1e-5)
            fixed,failures=r.recover_region(d['people'],rms,si,a,b,d['limit'])
            self.assertFalse(failures)
            targets=[(184.921,186.642)] if n==1 else [(335.787,335.928),(336.115,336.355)]
            for x,y in targets:
                self.assertFalse(covers(original,x,y))
                self.assertTrue(covers(fixed,x,y))
            for c in fixed:self.assertTrue(2<=c['duration_s']<=11.95)
            for p in d['people']:
                for w in p['words']:
                    if p is d['people'][si] and covers(original,w['start'],w['end']):
                        self.assertTrue(covers(fixed,w['start'],w['end']))
                    self.assertFalse(any(w['start']+1e-6<t<w['end']-1e-6 for c in fixed for t in (c['start'],c['end'])))
            self.assertTrue(all(x['end']<=y['start'] for x,y in zip(fixed,fixed[1:])))

if __name__=='__main__':unittest.main()
