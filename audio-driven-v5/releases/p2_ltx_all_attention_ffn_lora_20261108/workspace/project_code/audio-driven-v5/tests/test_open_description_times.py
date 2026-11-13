"""Regression checks for table-only temporal inventory; no model inference."""
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('open_times', Path(__file__).resolve().parents[1]/'scripts/report_open_backchannel_times.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class OpenTimeReportTests(unittest.TestCase):
    def test_fragmented_audio_keeps_silent_gap(self):
        e = dict(start=0,end=4,intervals=[dict(start=0,end=1),dict(start=3,end=4)])
        self.assertEqual(REPORT.intervals(e,4),[[0,1],[3,4]])
        self.assertEqual(sum(b-a for a,b in REPORT.intervals(e,4)),2)

    def test_overlapping_duplicates_do_not_inflate_time(self):
        self.assertEqual(REPORT.union([(1,2),(1,2),(1.5,3),(4,5)]),[[1,3],[4,5]])

    def test_invalid_boundary_fails(self):
        with self.assertRaises(ValueError):
            REPORT.intervals(dict(start=-1,end=1),2)

    def test_quantile_nearest_rank(self):
        self.assertEqual(REPORT.quantile({'100':1,'200':2,'900':1},.5),.2)
        self.assertEqual(REPORT.quantile({'100':1,'200':2,'900':1},.9),.9)

    def test_scope_verbatim_and_source_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'shards').mkdir(); (root/'labels').mkdir();(root/'evidence').mkdir()
            event=dict(event_id='v1',modality='visual',description='Closes eyes.',start=1,end=2,categories=[])
            row=dict(sample_id='test',duration_s=5,source=dict(source_start_s=20,source={'member':'example.mp4'}),
                     events=[event,dict(event,event_id='v2')],unresolved_observations=[dict(observation_id='v3',reason='out_of_scope')])
            (root/'post.jsonl').write_text(json.dumps(row)+'\n')
            result=REPORT.scan(('retained','retained-test',[str(root/'post.jsonl')],tmp,tmp))
            self.assertEqual(result['totals']['records'],2)
            self.assertEqual(result['totals']['union_seconds'],1)
            stat=json.loads((root/'shards/retained-test.stats.jsonl').read_text())
            self.assertEqual(stat['key'],['retained','visual','Closes eyes.'])
            self.assertEqual(stat['samples'],1)
            with gzip.open(root/'shards/retained-test.events.jsonl.gz','rt') as f:
                detail=json.loads(f.readline())
            self.assertEqual(detail['source_intervals'],[[21,22]])
            self.assertEqual(detail['original_record'],event)
            (root/'labels/0000.jsonl').write_text(json.dumps(row)+'\n')
            evidence=dict(sample_id='test',original_events=[dict(event_id='v3',modality='visual',behavior='Head tilt.',start=3,end=4,
                            categories=[],mapping_status='out_of_scope'),dict(event_id='a1',modality='audio',behavior='excluded speech',start=0,end=1)])
            (root/'evidence/0000.jsonl').write_text(json.dumps(evidence)+'\n')
            result=REPORT.scan(('unassigned_visual','evidence-test',[str(root/'evidence/0000.jsonl')],tmp,tmp))
            self.assertEqual(result['totals']['records'],1)
            self.assertNotIn('audio_records',result['totals'])


if __name__ == '__main__':
    unittest.main()
