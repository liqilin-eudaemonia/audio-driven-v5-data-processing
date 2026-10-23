import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_listen_local_full as run
import listen_full_source_runtime as runtime
import listen_crop_distribution as crops


class FullRunTests(unittest.TestCase):
    def test_numpy_evidence_is_saved_as_native_json_types(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'result.json'
            run.write(path,dict(qc=[dict(accepted=np.bool_(True),frames=np.int64(7),score=np.float32(.5))]))
            row=json.loads(path.read_text())['qc'][0]
            self.assertIs(row['accepted'],True)
            self.assertEqual(row['frames'],7)
            self.assertEqual(row['score'],.5)
            with self.assertRaises(TypeError):run.write(path,dict(unsupported=object()))
            self.assertEqual(json.loads(path.read_text())['qc'][0],row)

    def test_rejected_parent_is_not_double_counted_and_child_is_rescreened(self):
        parent=dict(sample_id='parent',start=0.,end=9.,duration_s=9.,speaker_id='a',listener_id='b',speaker_index=0,listener_index=1)
        child=dict(parent,sample_id='child',start=4.,end=7.,duration_s=3.)
        context=types.SimpleNamespace(limit=10.,shots=[],shots_by_track=[[],[]],regions=[],source_gaps=[],geometry=[{},{}],
            propose=lambda *a,**kw:([copy.deepcopy(child)],dict(blocked=[],unresolved=[])))
        prepared=dict(context=context,candidates=[parent],diagnostic=dict(blocked=[],unresolved=[]),
                      norm={},cache={},old_kept=[],elapsed=1.)
        screened=[]
        def raw(plan,*args):screened.append(plan['sample_id']);return dict(plan)
        def finish(plan,row,feedback):
            row.update(plan);bad=plan['sample_id']=='parent'
            row['audio_final']=dict(pool='audio_reject' if bad else 'keep_candidate',reasons=['local'] if bad else [],
                quality=dict(hard_reasons=['local'] if bad else [],review_reasons=[],
                             evidence_windows=[dict(start=1,end=2,reason='local')] if bad else []))
            return row
        with patch.object(run.baseline,'RAW_SCREEN',side_effect=raw),patch.object(run.baseline.original,'assess',side_effect=lambda r,f:r), \
             patch.object(run.baseline,'finish_row',side_effect=finish),patch.object(run.baseline,'OLD_FEEDBACK',[]), \
             patch.object(crops,'measure',return_value={'status':'unmeasured'}):
            result,audit=run.screen_prepared(dict(pair='p',split='train',group_id='g'),prepared,[1,2],dict(rejects=[],positives=[]))
        self.assertEqual(screened,['parent','child'])
        self.assertEqual([r['sample_id'] for r in result['clips']],['child'])
        self.assertEqual([r['sample_id'] for r in result['superseded_attempts']],['parent'])
        self.assertEqual(result['coverage']['new_kept_seconds'],3)
        self.assertEqual(result['counts']['fresh_clips'],2)

    def test_source_failure_is_accounted_separately(self):
        p=dict(context=None,candidates=[],norm=None,cache={},elapsed=0,diagnostic=dict(blocked=[],unresolved=[dict(reason='Missing transcript')]))
        r,a=run.screen_prepared(dict(pair='p',split='train',group_id='g'),p,[],{})
        self.assertFalse(r['source_ok']);self.assertIsNone(r['source_seconds']);self.assertEqual(r['clips'],[])

    def test_crop_statistics_do_not_drop_unmeasured_or_head_fit_warnings(self):
        base=dict(split='train',admission_pool='primary',audio_final=dict(pool='keep_candidate'),duration_s=4.)
        measured=dict(status='proposal_only',size=512.,output_face_height_px=180.,face_area_fraction=.12,fits_estimated_head=False)
        rows=[dict(base,crop_proposals=dict(listener=measured)),dict(base,crop_proposals=dict(listener={'status':'unmeasured'}))]
        s=crops.summarize(rows)['train/primary/keep_candidate']['listener']
        self.assertEqual(s['measured_clips'],1);self.assertEqual(s['unmeasured_clips'],1)
        self.assertEqual(sum(s['side_seconds'])+s['unmeasured_seconds'],8)
        self.assertEqual(s['estimated_head_not_fitting_clips'],1)

    def test_invalid_crop_geometry_is_unmeasured_not_a_rejection(self):
        with patch.object(crops.policy,'preview_crop',side_effect=ZeroDivisionError()):
            self.assertEqual(crops.measure({},0,4)['status'],'unmeasured')

    def test_gpu_failure_falls_back_and_never_silently_means_no_cuts(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.tar';source.write_bytes(b'test')
            with patch.object(runtime,'CACHE',Path(d)),patch.object(runtime,'decoder_slot',return_value=contextlib.nullcontext()), \
                 patch.object(runtime.policy.old.base.media,'uri',return_value='source'), \
                 patch.object(runtime.subprocess,'run',return_value=types.SimpleNamespace(returncode=1,stderr='GPU unavailable',stdout='')), \
                 patch.object(runtime,'CPU_DETECT',return_value=[2.]) as cpu:
                item=dict(source=dict(archive=str(source),member='a.mp4'),duration=4.)
                self.assertEqual(runtime.detect_shots(item),[2.]);self.assertEqual(runtime.detect_shots(item),[2.])
                self.assertEqual(cpu.call_count,1)
                record=json.loads(next(Path(d).glob('*.json')).read_text());self.assertEqual(record['decoder'],'cpu_fallback')


if __name__=='__main__':unittest.main()
