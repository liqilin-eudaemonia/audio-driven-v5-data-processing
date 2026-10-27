import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('test_panel_selector',Path(__file__).resolve().parents[1]/'scripts/select_processed_test_panel.py')
selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)


def row(i,category='head_nod',face=320,**changes):
    r=dict(sample_id=f'sample-{i:03d}',pair=f'pair-{i:03d}',split='test',label_issue_flag=0,
           face_valid_fraction=1.,face_short_side_p10=face,non_smile_event_count=1,
           source_start_s=0.,source_end_s=6.,duration_s=6.,
           **{c+'_count':int(c==category) for c in selector.CATEGORIES})
    r.update(changes);return r


class TestCountPanel(unittest.TestCase):
    def pool(self):
        return [row(i,selector.CATEGORIES[i%7] if i%7!=1 else 'head_nod') for i in range(80)]+[row(99,'head_shake',160)]

    def test_48_with_rare_supplement_and_quality(self):
        rows=self.pool()+[row(101,'head_nod',500,split='train'),row(102,'head_nod',500,label_issue_flag=1)]
        selected,policy=selector.select(rows,48)
        self.assertEqual(len(selected),48)
        self.assertEqual(policy['core_samples'],43)
        self.assertIn('sample-099',{r['sample_id'] for r in selected})
        self.assertTrue(all(r['split']=='test' and not r['label_issue_flag'] for r in selected))
        self.assertTrue(all(any(r[c+'_count'] for r in selected) for c in selector.CATEGORIES))

    def test_missing_rare_class_fails(self):
        with self.assertRaises(ValueError):
            selector.select(self.pool()[:-1],48)

    def test_source_overlap_not_selected_twice(self):
        pool=self.pool();pool[0]['face_short_side_p10']=500
        pool.append(dict(pool[0],sample_id='duplicate-source'))
        selected,_=selector.select(pool,48)
        self.assertLessEqual(sum(r['pair']==pool[0]['pair'] for r in selected),1)


if __name__=='__main__':
    unittest.main()
