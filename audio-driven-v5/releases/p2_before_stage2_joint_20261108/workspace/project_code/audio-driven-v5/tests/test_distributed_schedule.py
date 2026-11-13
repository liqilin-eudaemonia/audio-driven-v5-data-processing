import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('train_listen_distributed',Path(__file__).resolve().parents[1]/'scripts/train_listen_distributed.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ScheduleTests(unittest.TestCase):
    def test_all_24_ranks_visit_each_example_once_per_epoch(self):
        actual=[]
        for step in range(1,5):
            for rank in range(24):
                actual.extend(module.sample_indices(192,123,step,48,24,rank))
        self.assertEqual(sorted(i for i,_ in actual),list(range(192)))
        self.assertEqual(sorted(o for _,o in actual),list(range(192)))

    def test_resume_and_epoch_crossing(self):
        full={o:i for s in range(1,8) for r in range(24)
              for i,o in module.sample_indices(101,77,s,48,24,r)}
        resumed={o:i for s in range(4,8) for r in range(24)
                 for i,o in module.sample_indices(101,77,s,48,24,r)}
        self.assertEqual(resumed,{o:i for o,i in full.items() if o>=144})
        for start in (0,101,202):
            self.assertEqual(sorted(full[o] for o in range(start,start+101)),list(range(101)))


if __name__=='__main__':unittest.main()
