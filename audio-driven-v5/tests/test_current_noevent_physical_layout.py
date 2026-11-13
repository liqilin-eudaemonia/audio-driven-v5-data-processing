import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('shift',ROOT/'scripts/p2_current_noevent_node11shift_20261108.py')
shift=importlib.util.module_from_spec(spec);spec.loader.exec_module(shift)


class LayoutTests(unittest.TestCase):
    def cluster(self):
        c=json.loads((shift.WORK/'cluster.json').read_text())
        c['nodes'][-1]['gpus']=list(range(1,8))
        return c

    def test_only_physical_placement_changes(self):
        values=shift.layout(self.cluster())
        self.assertEqual([r for n in values for r in n['ranks']],list(range(31)))
        self.assertEqual(values[-1]['gpus'],list(range(1,8)))
        self.assertEqual(values[-1]['ranks'],list(range(24,31)))
        self.assertEqual([len(v['gpus']) for v in values],[8,8,8,7])

    def test_gpu0_or_missing_rank_is_rejected(self):
        for gpus in [list(range(7)),list(range(1,7))]:
            c=self.cluster();c['nodes'][-1]['gpus']=gpus
            with self.assertRaises(ValueError):shift.layout(c)


if __name__=='__main__':unittest.main()
