import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import write,read,result_path
from listen_benchmark.j_omni_crop import effective_row,corrected_nod,result_sources,attempts_root,caption_crop_info,REVISION
from listen_benchmark.j_omni import pending_caption_rows
from listen_visual_quality.io import canonical_hash,file_hash

class CropTests(unittest.TestCase):
 def fixture(self,d,changed=True):
  w=Path(d);row=dict(sample_id='s',source=dict(pair='pair',duration_s=1,video_transform=dict(crop=dict(x=10,y=20,width=100,height=100),source_size=[300,300]),crop_proposal=dict(revision='old')))
  new=dict(x=10,y=15 if changed else 20,width=100,height=100)
  m=dict(status='ok',source=row['source'],contract_sha256='frozen',crop=dict(new=new,status='translation_proposal' if changed else 'unchanged'))
  p=w/'m.json';write(p,m)
  decision=dict(sample_id='s',decision='retain',source_sha256=canonical_hash(row['source']),measurement_file=str(p),measurement_sha256=file_hash(p))
  write(w/'measure/completed/pair.json',dict(contract_sha256='frozen',decisions=[decision]))
  cfg=dict(bounded_crop_input=dict(revision=REVISION),visual_quality_gate=dict(work=str(w),contract_sha256='frozen'))
  return row,cfg,p
 def test_applies_exact_proposal_preserves_original_and_records_actual(self):
  with tempfile.TemporaryDirectory() as d:
   row,cfg,_=self.fixture(d);before=copy.deepcopy(row);new,provenance=effective_row(row,cfg)
   self.assertEqual(row,before);self.assertEqual(new['source']['video_transform']['crop']['y'],15)
   self.assertTrue(provenance['coordinates_changed']);self.assertEqual(new['source']['duration_s'],1)
   out=result_sources(row,dict(effective_source=new['source'],evidence=dict(crop_application=provenance)))
   self.assertEqual(out['original_source'],row['source']);self.assertEqual(out['source'],new['source'])
 def test_unchanged_coordinates_reuse_existing_nod_without_subprocess(self):
  with tempfile.TemporaryDirectory() as d:
   row,cfg,_=self.fixture(d,False);new,_=effective_row(row,cfg)
   with patch('listen_benchmark.j_omni.nod_evidence',return_value={'events':[]}) as nod,patch('subprocess.run') as run:
    out=corrected_nod(row,new,'old',d,'res',cfg)
   nod.assert_called_once_with(row,'old');run.assert_not_called();self.assertEqual(out['crop_reuse'],'identical_video_transform')
 def test_changed_crop_reuses_only_matching_corrected_upstream_nod(self):
  with tempfile.TemporaryDirectory() as d:
   row,cfg,_=self.fixture(d);new,_=effective_row(row,cfg)
   write(result_path(d,'nod','s'),dict(source=new['source']))
   with patch('listen_benchmark.j_omni.nod_evidence',return_value={'events':[]}) as nod,patch('subprocess.run') as run:
    out=corrected_nod(row,new,d,d,'res',cfg)
   nod.assert_called_once_with(new,d);run.assert_not_called();self.assertEqual(out['crop_reuse'],'upstream_corrected_crop')
 def test_early_nod_receipt_works_before_downstream_publication(self):
  with tempfile.TemporaryDirectory() as d:
   row,cfg,_=self.fixture(d);p=Path(d)/'measure/completed/pair.json'
   decision=read(p)['decisions'][0];decision['contract_sha256']='frozen';p.unlink()
   applied,prov=effective_row(row,cfg,decision=decision)
   self.assertTrue(prov['coordinates_changed']);self.assertEqual(applied['source']['video_transform']['crop']['y'],15)
   decision['contract_sha256']='wrong'
   with self.assertRaises(ValueError):effective_row(row,cfg,decision=decision)
 def test_mismatched_source_and_tampered_measurement_are_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   row,cfg,p=self.fixture(d);bad=copy.deepcopy(row);bad['source']['duration_s']=2
   with self.assertRaises(ValueError):effective_row(bad,cfg)
   m=read(p);m['crop']['new']['x']=400;write(p,m)
   with self.assertRaises(ValueError):effective_row(row,cfg)
 def test_invalid_bound_rejected_even_with_matching_checksum(self):
  with tempfile.TemporaryDirectory() as d:
   row,cfg,p=self.fixture(d);m=read(p);m['crop']['new']['width']=106;m['crop']['new']['height']=106;write(p,m)
   f=Path(d)/'measure/completed/pair.json';r=read(f);r['decisions'][0]['measurement_sha256']=file_hash(p);write(f,r)
   with self.assertRaises(ValueError):effective_row(row,cfg)
 def test_completed_original_caption_is_never_rewritten_or_requeued(self):
  with tempfile.TemporaryDirectory() as d:
   row,cfg,_=self.fixture(d);p=result_path(d,'caption','s');old=dict(source=row['source'],pipeline_id='p',caption='Untouched')
   write(p,old);before=p.read_bytes();todo,_=pending_caption_rows(d,dict(rows=[row]),'p',{'s':{'decision':'retain'}})
   self.assertEqual(todo,[]);self.assertEqual(before,p.read_bytes());self.assertEqual(caption_crop_info(old)['label_origin'],'legacy_original_crop_preserved')
 def test_corrected_caption_resumes_against_original_job(self):
  with tempfile.TemporaryDirectory() as d:
   row,cfg,_=self.fixture(d);new,prov=effective_row(row,cfg);write(Path(d)/'config.json',cfg)
   out=dict(pipeline_id='p',**result_sources(row,dict(effective_source=new['source'],evidence=dict(crop_application=prov))))
   write(result_path(d,'caption','s'),out)
   self.assertEqual(pending_caption_rows(d,dict(rows=[row]),'p')[0],[])
   self.assertNotEqual(attempts_root(d,cfg),attempts_root(d,{}))
if __name__=='__main__':unittest.main()
