import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import listen_v8_face_statistics as faces
import plan_listen_v8_audio_full as planner
import run_listen_v8_audio_full as run


def geometry(width=40, height=50, frames=75):
    points = np.zeros((frames, 110, 3))
    points[:, 23:91, 0] = np.linspace(100, 100+width, 68)
    points[:, 23:91, 1] = np.linspace(100, 100+height, 68)
    points[:, 23:91, 2] = 1
    return dict(width=1920, height=1080, fps=25., rotation_degrees=0,
                arrays={'boxes_and_keypoints:keypoints':points})


class FaceStatisticsTests(unittest.TestCase):
    def test_native_pixels_and_ratio_not_crop_size(self):
        value = faces.clip_geometry(geometry(), 0, 3)
        self.assertEqual(value['valid_frames'], 75)
        self.assertAlmostEqual(value['face_width_px']['p50'], 37.6)
        self.assertAlmostEqual(value['face_area_fraction']['p50'], 37.6*47/(1920*1080))
        self.assertEqual(sum(value['frame_short_side_histogram']),75)

    def test_missing_face_is_not_zero_size(self):
        g = geometry(); g['arrays']['boxes_and_keypoints:keypoints'] = None
        value = faces.clip_geometry(g, 0, 3)
        self.assertEqual(value['status'],'unmeasured')
        self.assertNotIn('face_short_side_px',value)
        self.assertEqual(value['expected_frames'],75)

    def test_invalid_frames_remain_in_coverage_denominator(self):
        g = geometry(); g['arrays']['boxes_and_keypoints:keypoints'][:60,:,2] = 0
        value = faces.clip_geometry(g,0,3)
        self.assertEqual(value['valid_fraction'],.2)
        self.assertEqual(value['valid_frames'],15)

    def test_missing_stats_do_not_remove_audio_candidate(self):
        g=geometry(); g['arrays']['boxes_and_keypoints:keypoints']=None
        row=dict(split='train',admission_pool='primary',audio_final={'pool':'keep_candidate'},duration_s=3,
                 quality_provenance={'reused':False},face_geometry={'listener':faces.clip_geometry(g,0,3)})
        value=faces.summarize_clips([row])['train/primary/keep_candidate']
        self.assertEqual(value['clips'],1);self.assertEqual(value['seconds'],3)
        self.assertEqual(value['roles']['listener']['unmeasured_seconds'],3)
        self.assertEqual(sum(value['roles']['listener']['median_short_side_counts']),0)

    def test_aggregate_counts_and_duration(self):
        row=dict(split='train',admission_pool='primary',audio_final={'pool':'keep_candidate'},duration_s=3,
                 quality_provenance={'reused':True},face_geometry={'listener':faces.clip_geometry(geometry(),0,3)})
        one=faces.summarize_clips([row]);total={}
        faces.merge_counts(total,one);faces.merge_counts(total,one)
        value=total['train/primary/keep_candidate']
        self.assertEqual(value['seconds'],6)
        self.assertEqual(sum(value['roles']['listener']['median_short_side_counts']),2)
        self.assertEqual(sum(value['roles']['listener']['frame_short_side_counts']),150)


class NoVisualGateTests(unittest.TestCase):
    def test_tiny_and_missing_faces_reach_audio_stage(self):
        item=dict(pair='pair',split='train',group_id='g',a={},b={})
        cut=dict(start=0.,end=3.,duration_s=3.,semantic_end='sentence_terminal')
        people=[dict(segments=[],words=[{'start':.2}],vad=[]) for _ in (0,1)]
        unit=dict(start=.2,end=2.7,text='Complete.',words=[{'start':.2}])
        def loaded(record):
            g=geometry(8,10);g['arrays']['boxes_and_keypoints:keypoints']=None
            return {'id':'person'},g,dict(duration=3.,source={},width=1920,height=1080,fps=25.)
        with patch.object(planner,'load_person',side_effect=loaded), \
             patch.object(planner.rules,'prepare',side_effect=people), \
             patch.object(planner.rules,'track_floors',return_value=([dict(speaker_index=0,start=0.,end=3.,start_anchor='x')],[])), \
             patch.object(planner,'sentence_units',return_value=[unit]), \
             patch.object(planner,'listener_compatibility',return_value={'substantive':[]}), \
             patch.object(planner,'group_units',side_effect=lambda eligible,person:(eligible,[])), \
             patch.object(planner,'raw_rms',return_value=np.zeros(300)), \
             patch.object(planner,'acoustic_cut',return_value=(cut,None)), \
             patch.object(planner,'boundary_activity',return_value={}), \
             patch.object(planner.rules,'full_turn_crop',side_effect=AssertionError('Crop must not gate audio')):
            result=planner.inspect_pair(item)
        self.assertEqual(len(result['clips']),1)
        self.assertIsNone(result['clips'][0]['listener_crop'])
        self.assertEqual(result['clips'][0]['face_geometry']['listener']['status'],'unmeasured')

    def test_cache_requires_exact_interval_source_and_waveform(self):
        plan=dict(speaker_id='a',listener_id='b',start=0.,end=3.,speaker_index=0,listener_index=1,sources={'speaker':{'id':'a'},'listener':{'id':'b'}})
        old=dict(plan,qc3={},qc4={},normalized_audio={'speaker':{'sha256':'a'},'listener':{'sha256':'b'}})
        norm={'tracks':[{'sha256':'a'},{'sha256':'b'}]}
        self.assertTrue(run.cache_matches(plan,old,norm))
        changed=copy.deepcopy(plan);changed['end']+=.01
        self.assertFalse(run.cache_matches(changed,old,norm))
        changed=copy.deepcopy(plan);changed['sources']['listener']['id']='c'
        self.assertFalse(run.cache_matches(changed,old,norm))
        self.assertFalse(run.cache_matches(plan,old,{'tracks':[{'sha256':'a'},{'sha256':'new'}]}))


if __name__=='__main__':unittest.main()
