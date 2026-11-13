import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from listen_j_quality_parallel import annotation_drained,annotation_owner_index,export_annotation_index
from listen_benchmark.j_annotation import write,read,result_path


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_both_branches_must_drain(self):
        w=self.root/'node';w.mkdir();(w/'ready.jsonl').write_text('{"job_id":"x"}\n');n=(w/'ready.jsonl').stat().st_size
        for b in ('vlm','nod'):write(w/'queues'/b/'state.json',{'offset':n,'inflight':{}})
        self.assertTrue(annotation_drained(w))
        write(w/'queues/nod/state.json',{'offset':n,'inflight':{'x':{}}})
        self.assertFalse(annotation_drained(w))

    def test_join_uses_new_annotation_owner_and_only_quality_kept_rows(self):
        q=self.root/'quality';q.mkdir();w=self.root/'local';w.mkdir()
        (w/'ready.jsonl').write_text('{"job_id":"pair"}\n')
        crop={'x':0,'y':0,'width':10,'height':10};new={**crop,'x':1}
        row={'sample_id':'kept','source':{'pair':'pair','video_transform':{'crop':new},
             'quality_screening':{'original_crop':crop}}}
        (q/'retained_rows.jsonl').write_text(json.dumps(row)+'\n')
        write(result_path(w,'vlm','kept'),{'source':'original'})
        write(result_path(w,'vlm','excluded'),{'source':'original'})
        export_annotation_index(q,{'nodes':{'node12':{'work':str(w)}}})
        rows=[json.loads(x) for x in (q/'retained_annotation_index.jsonl').read_text().splitlines()]
        self.assertEqual([r['sample_id'] for r in rows],['kept'])
        self.assertEqual(rows[0]['annotations']['vlm'],str(result_path(w,'vlm','kept')))
        self.assertIsNone(rows[0]['annotations']['nod'])
        self.assertTrue(rows[0]['crop_differs_from_annotation'])
        self.assertEqual(read(q/'parallel_complete.json')['counts']['missing_nod'],1)

    def test_duplicate_source_owner_rejected(self):
        nodes={}
        for name in ('one','two'):
            w=self.root/name;w.mkdir();(w/'ready.jsonl').write_text('{"job_id":"same"}\n');nodes[name]={'work':str(w)}
        with self.assertRaises(ValueError):annotation_owner_index({'nodes':nodes})

    def test_delegated_visual_and_original_nod_join_from_different_nodes(self):
        nodes={}
        for name in ('local','remote'):
            w=self.root/name;w.mkdir();(w/'ready.jsonl').write_text('{"job_id":"pair"}\n')
            nodes[name]={'work':str(w)}
        assignment={'vlm':nodes['remote']['work'],'nod':nodes['local']['work']}
        scope={'nodes':nodes,'branch_ownership_overrides':{'pair':assignment}}
        q=self.root/'quality';q.mkdir();crop={'x':0,'y':0,'width':10,'height':10}
        row={'sample_id':'s','source':{'pair':'pair','video_transform':{'crop':crop},
             'quality_screening':{'original_crop':crop}}}
        (q/'retained_rows.jsonl').write_text(json.dumps(row)+'\n')
        for branch,w in assignment.items():write(result_path(w,branch,'s'),{'source':'original'})
        export_annotation_index(q,scope)
        joined=json.loads((q/'retained_annotation_index.jsonl').read_text())
        self.assertEqual(joined['annotation_work_by_branch'],assignment)
        self.assertTrue(all(joined['annotations'].values()))
        self.assertEqual(read(q/'parallel_complete.json')['counts']['both_labels_present'],1)

    def test_delegation_cannot_hide_unexpected_third_owner(self):
        nodes={}
        for name in ('one','two','three'):
            w=self.root/name;w.mkdir();(w/'ready.jsonl').write_text('{"job_id":"same"}\n');nodes[name]={'work':str(w)}
        scope={'nodes':nodes,'branch_ownership_overrides':{'same':{'vlm':nodes['one']['work'],'nod':nodes['two']['work']}}}
        with self.assertRaises(ValueError):annotation_owner_index(scope)


if __name__=='__main__':unittest.main()
