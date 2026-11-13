import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import listen_v8_recovery as r
import numpy as np

class RecoveryTests(unittest.TestCase):
    def person(self):
        s=dict(id='long',start=1.,end=10.,text='A long substantive sentence.',feedback=False,
            word_count=9,formal_takeover=True,word_aligned=True,
            words=[dict(start=1.,end=1.4,text='First'),dict(start=9.6,end=10.,text='last.')])
        return dict(segments=[s],vad=[(1.,1.4),(9.6,10.)])
    def test_verified_gap_is_not_substantive_overlap(self):
        p=self.person();before=copy.deepcopy(p)
        self.assertTrue(r.BASE_COMPATIBILITY(p,3,7)['substantive'])
        self.assertFalse(r.listener_compatibility(p,3,7)['substantive'])
        self.assertEqual(p,before)
    def test_actual_word_or_vad_keeps_blocker(self):
        for kind in ('word','vad','jitter'):
            p=self.person()
            if kind=='word':p['segments'][0]['words'].insert(1,dict(start=4.,end=4.3,text='answer'))
            elif kind=='vad':p['vad'].append((4.,4.3))
            else:p['vad'].append((2.95,2.99))
            self.assertTrue(r.listener_compatibility(p,3,7)['substantive'])
    def test_uncertain_alignment_or_missing_vad_is_not_rescued(self):
        for kind in ('alignment','missing_vad','invalid_word'):
            p=self.person()
            if kind=='alignment':p['segments'][0]['word_aligned']=False
            elif kind=='missing_vad':del p['vad']
            else:p['segments'][0]['words'][0]['end']=5
            self.assertTrue(r.listener_compatibility(p,3,7)['substantive'])
    def test_same_gap_is_not_reblocked_by_tail_envelope(self):
        listener=self.person();speaker=dict(words=[dict(start=3.,end=3.4,text='A'),dict(start=6.6,end=7.,text='sentence.')],segments=[],vad=[(3,7)])
        listener['words']=listener['segments'][0]['words']
        unit=dict(start=3.,end=7.,semantic_flags=[],text='A sentence.',start_evidence='after_sentence_terminal',end_evidence='sentence_terminal')
        cut,error=r.acoustic_cut(unit,[speaker,listener],[np.full(1200,1e-5)]*2,0,12.)
        self.assertIsNone(error);self.assertEqual(cut['asr_gap_tail_exemptions'],['long'])
        self.assertLess(cut['end'],9.6)
        listener['vad'].append((6.9,7.3))
        cut,error=r.acoustic_cut(unit,[speaker,listener],[np.full(1200,1e-5)]*2,0,12.)
        self.assertIsNone(cut)
    def row(self):return dict(duration_s=4.,start=1.,end=5.,speech_start=1.2,speech_end=4.8,
        boundary_review_flags=['end_no_quiet_cut'],audio_qc_status='keep',
        release={'pool':'boundary_review','reasons':['remaining_boundary_or_human_hold']},
        qc3={'listener':{'reasons':[]},'speaker':{'reasons':[]}},qc4={'task_reasons':[],'audio_reasons':[]})
    def test_soft_boundary_does_not_override_hard_flags_or_audio(self):
        row=self.row();self.assertTrue(r.analysis(row)['boundary_recovery_candidate'])
        for flag in ('end_crosses_annotated_voice','word_timestamp_conflict','untranscribed_listener_tail','unknown_future_flag'):
            x=copy.deepcopy(row);x['boundary_review_flags'].append(flag)
            self.assertFalse(r.analysis(x)['boundary_recovery_candidate'])
        row['qc4']['audio_reasons']=['strong_reference_match'];self.assertFalse(r.analysis(row)['boundary_recovery_candidate'])
    def test_human_hold_never_rescued(self):
        row=self.row();row['release']['reasons']=['human_repair_reserve']
        self.assertFalse(r.analysis(row)['boundary_recovery_candidate'])
    def test_identity_is_review_potential_not_admitted(self):
        row=self.row();row['release']['pool']='repair_reserve';row['audio_qc_status']='review'
        row['qc3']['listener']['reasons']=['reaction_identity_uncertain']
        out=r.analysis(row);self.assertTrue(out['identity_only_review_potential']);self.assertFalse(out['training_admitted'])
        row['qc4']['audio_reasons']=['strong_reference_match'];self.assertFalse(r.analysis(row)['identity_only_review_potential'])
    def test_increment_uses_union_not_added_clip_sum(self):
        self.assertEqual(r.incremental_seconds([(0,5)],[(3,7),(6,10)]),5)
        self.assertEqual(r.incremental_seconds([(0,5)],[(1,4)]),0)

if __name__=='__main__':unittest.main()
