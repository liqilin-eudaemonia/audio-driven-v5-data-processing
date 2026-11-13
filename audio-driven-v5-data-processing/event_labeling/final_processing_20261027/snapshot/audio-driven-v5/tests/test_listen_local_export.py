import collections
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import listen_local_export as export


class ExportTests(unittest.TestCase):
    def setup_rows(self,d):
        root=Path(d)
        for folder in ['results','summaries']:(root/folder).mkdir()
        expected={};counts=collections.Counter()
        for i in range(5):
            row=dict(sample_id=str(i),split='train',admission_pool='primary',audio_final=dict(pool='keep_candidate' if i%2 else 'audio_review'),qc3={'evidence':[1,2]},qc4={'quality':True},start=i,end=i+1)
            name=f'p{i}';(root/'results'/(name+'.json')).write_text(json.dumps(dict(clips=[row],superseded_attempts=[dict(row,sample_id='parent')])));(root/'summaries'/(name+'.json')).write_text('{}')
            counts['train/primary/'+row['audio_final']['pool']]+=1
            compact={k:v for k,v in row.items() if k not in ('qc3','qc4')};compact['quality_evidence_file']=str(root/'results'/(name+'.json'));expected[str(i)]=compact
        return root,dict(complete_pairs=5,counts=dict(counts),final=True),expected

    def test_parallel_export_matches_serial_rows_and_excludes_parents(self):
        with tempfile.TemporaryDirectory() as d:
            root,summary,expected=self.setup_rows(d)
            result=export.export(root,summary,workers=2,shards=3)
            actual={}
            for p in (root/'manifests').glob('*.jsonl'):
                for line in p.read_text().splitlines():
                    row=json.loads(line);self.assertNotIn(row['sample_id'],actual);actual[row['sample_id']]=row
                self.assertEqual(export.digest(p),result['manifest_sha256'][p.stem])
            self.assertEqual(actual,expected)
            self.assertTrue(result['export']['counts_verified'])

    def test_count_mismatch_never_publishes_manifest_index(self):
        with tempfile.TemporaryDirectory() as d:
            root,summary,_=self.setup_rows(d);summary['counts']['train/primary/audio_review']+=1
            with self.assertRaises(AssertionError):export.export(root,summary,workers=2,shards=3)
            self.assertFalse((root/'manifests/index.json').exists())


if __name__=='__main__':unittest.main()
