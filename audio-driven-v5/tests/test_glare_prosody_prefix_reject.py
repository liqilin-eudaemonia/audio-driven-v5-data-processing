import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_prefix_reject import impossible_prefix
from glare_prosody_subset import audit_raw


class PrefixRejectTests(unittest.TestCase):
    def test_no_prefix_of_admissible_answer_is_rejected(self):
        valid=['NULL',' null \n','\nNuLl','0.00, 0.50, 0.80',
               '-0.00, 0.01, 0.30','0.0, 2.4000005, 0.30',
               '\n0.00, 0.50, 0.80\n\n0.50, 2.40, 0.90\n',
               '0.0000001, 0.004, 0.30','\t0.00,\t0.50,\t0.80\t',
               '0.00,\u20030.50,\u20030.80']
        for raw in valid:
            self.assertTrue(audit_raw(raw)['accepted'],raw)
            for n in range(len(raw)+1):
                with self.subTest(raw=raw,n=n):self.assertIsNone(impossible_prefix(raw[:n]))

    def test_irreversible_bad_prefixes_and_still_possible_partial_number(self):
        from unittest.mock import patch
        import glare_prosody_prefix_reject as module
        pattern=module.PREFIX
        class SlowTestBudget:
            def fullmatch(self,*args,**kwargs):
                kwargs['timeout']=1.
                return pattern.fullmatch(*args,**kwargs)
        for raw in ["I'm",'Sorry',r'0.00, 0.50, 0.80\n','NULL\n0',
                    '2.34, 2.79,','-0.10, 0.20,','0.50, 0.20,',
                    '0.00, 0.50, 0.80\n0.30, 0.80,']:
            # Test language/interval proof independently of a5ms scheduling budget.
            with patch.object(module,'PREFIX',SlowTestBudget()):
                self.assertIsNotNone(impossible_prefix(raw),raw)
            self.assertFalse(audit_raw(raw)['accepted'])
        for raw in ['0.00, 0.0','0.20, 0.2','0.00, 2.','0.00, 0.50,','N','NU','NUL','0.00,\ufffd']:
            self.assertIsNone(impossible_prefix(raw),raw)


    def test_random_valid_numeric_answers_never_early_rejected(self):
        import random
        rng=random.Random(137)
        for case in range(256):
            points=sorted(rng.sample(range(241),8))
            text='\n'.join(f'{points[i]/100:.2f}, {points[i+1]/100:.2f}, {rng.choice([.3,.5,.8,1.]):.2f}' for i in range(0,8,2))
            self.assertTrue(audit_raw(text)['accepted'])
            for n in range(len(text)+1):
                self.assertIsNone(impossible_prefix(text[:n]),(case,n,text))


    def test_regex_timeout_defers_rejection_instead_of_dropping_data(self):
        import glare_prosody_prefix_reject as module
        from unittest.mock import Mock,patch
        matcher=Mock();matcher.fullmatch.side_effect=TimeoutError('CPU preemption')
        with patch.object(module,'PREFIX',matcher):
            self.assertIsNone(impossible_prefix('0.00, 0.50, 0.80'))
            self.assertIsNone(impossible_prefix('Sorry'))


if __name__=='__main__':unittest.main()
