import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_quality_pipeline import cropped_row,source_job,measurement_path,export_node
from listen_benchmark.j_annotation import read,write,result_path
from listen_visual_quality.io import canonical_hash,file_hash
from listen_visual_quality.cli import load_policy


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real=read(ROOT/'experiments/visual_quality_holdout120_20261030/data.json')['results']
        cls.policy=load_policy(ROOT/'configs/listen_visual_quality_v1.json')

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.old=self.base/'old';self.work=self.base/'quality';self.target=self.base/'annotation'
        self.work.mkdir();self.old.mkdir()
        self.r=copy.deepcopy(next(r for r in self.real if r['crop']['changed'] and r['focus']['measured_fraction']>=.8 and r['focus']['low_fraction']==0))
        self.row={'sample_id':self.r['id'],'source':copy.deepcopy(self.r['source'])}
        self.decision={'decision':'retain','source_sha256':canonical_hash(self.row['source'])}

    def test_crop_applied_without_changing_time_or_input(self):
        before=copy.deepcopy(self.row)
        new=cropped_row(self.row,self.r,self.decision,'measurement.json')
        self.assertEqual(self.row,before)
        self.assertEqual(new['source']['video_transform']['crop'],self.r['crop']['new'])
        for k in ('source_start_s','source_end_s','source','sample_id'):
            self.assertEqual(new['source'][k],self.row['source'][k])

    def test_unknown_excluded_or_wrong_source_never_admitted(self):
        for decision in ({**self.decision,'decision':'review'},{**self.decision,'decision':'exclude'},
                         {**self.decision,'source_sha256':'bad'}):
            with self.assertRaises(ValueError):cropped_row(self.row,self.r,decision,'m')

    def test_outside_or_overgrown_crop_rejected(self):
        for change in ({'x':-1},{'width':100000,'height':100000}):
            r=copy.deepcopy(self.r);r['crop']['new'].update(change)
            with self.assertRaises(ValueError):cropped_row(self.row,r,self.decision,'m')

    def prepared(self):
        job={'job_id':'one','rows':[self.row],'source_result_sha256':self.row['source']['source_result_sha256'],'excluded':[]}
        write(self.old/'jobs/one.json',job)
        r=copy.deepcopy(self.r);r['measurement_hash']='frozen'
        write(measurement_path(self.work,self.row['sample_id']),r)
        for b in ('vlm','nod'):
            (self.old/b/'results').mkdir(parents=True,exist_ok=True)
        oldlabel=result_path(self.old,'vlm',self.row['sample_id'])
        write(oldlabel,{'source':self.row['source'],'sample_id':self.row['sample_id'],'keep':'original'})
        return oldlabel

    def test_cached_measurements_and_existing_labels_are_not_recomputed(self):
        oldlabel=self.prepared();original=oldlabel.read_bytes()
        with patch('listen_visual_quality.measurement.measure_row',side_effect=AssertionError('must reuse')):
            receipt=source_job(self.work,{'old_work':str(self.old)},'one',self.policy,{},'frozen')
        self.assertEqual(oldlabel.read_bytes(),original)
        self.assertEqual(receipt['annotation_reuse'],{'vlm':1,'nod':0})
        d=read(self.work/'approved_jobs/one.json')['quality_decisions'][0]
        self.assertFalse(d['annotation_reuse']['vlm']['crop_matches_final'])
        self.assertTrue(d['annotation_reuse']['vlm']['user_authorized_no_reannotation'])

    def test_existing_labels_for_other_times_are_not_reused(self):
        path=self.prepared();v=read(path);v['source']['source_start_s']+=1;write(path,v)
        with self.assertRaisesRegex(ValueError,'source mismatch'):
            source_job(self.work,{'old_work':str(self.old)},'one',self.policy,{},'frozen')

    def test_export_retains_speed_configuration_and_existing_result_store(self):
        oldlabel=self.prepared()
        receipt=source_job(self.work,{'old_work':str(self.old)},'one',self.policy,{},'frozen')
        write(self.work/'quality/completed/one.json',receipt)
        write(self.work/'queues/quality/state.json',{'failed':{}})
        (self.work/'ready.jsonl').write_text('{"job_id":"one","count":1}\n')
        cfg={'batch_size':512,'tokens':32768,'renderer_workers':2,'gpu_execution_overrides':{'0':{'memory':.8}}}
        for name in ('config.json','selected.json','resource_tuning.json'):write(self.work/('original_'+name),cfg)
        report=export_node(self.work,{'old_work':str(self.old),'annotation_work':str(self.target),
                                    'hostname':'test','node':'node17','scope_clips':1})
        self.assertEqual(report['decisions'],{'retain':1,'exclude':0,'review':0})
        self.assertEqual(read(self.target/'selected.json'),cfg)
        self.assertEqual(read(self.target/'resource_tuning.json'),cfg)
        self.assertEqual(read(self.target/'config.json')['gpu_execution_overrides'],cfg['gpu_execution_overrides'])
        self.assertEqual(result_path(self.target,'vlm',self.row['sample_id']).resolve(),oldlabel.resolve())
        self.assertFalse(result_path(self.target,'nod',self.row['sample_id']).exists())
        self.assertEqual(read(self.target/'jobs/one.json')['rows'][0]['source']['video_transform']['crop'],self.r['crop']['new'])


if __name__=='__main__':unittest.main()
