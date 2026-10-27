import copy
import sys
import unittest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import listen_local_recovery as r


def word(a,b,text='word',feedback=False):
    return dict(start=a,end=b,text=text,feedback=feedback)


class RecoveryTests(unittest.TestCase):
    def test_local_overlap_does_not_remove_before_and_after(self):
        self.assertEqual(r.subtract([(0,20)],[(8,9),(8.5,10)]),[(0,8),(10,20)])
        self.assertEqual(r.seconds([(0,9),(8,10)]),10)
        self.assertEqual(r.union([[0,1],(0,2)]),[[0,2]])

    def test_source_scan_role_changes_and_shots_on_both_sides(self):
        people=[dict(words=[word(t,t+.8) for t in range(1,10)]),
                dict(words=[word(t,t+.8) for t in range(12,21)])]
        regions,gaps=r.role_regions(people,24,[5,16])
        self.assertEqual([x['speaker_index'] for x in regions],[0,0,1,1])
        self.assertTrue(all(not x['start']<t<x['end'] for x in regions for t in [5,16]))
        self.assertTrue(any(x['start']<5 and x['end']==5 for x in regions))
        self.assertTrue(any(x['start']==5 and x['end']>5 for x in regions))
        self.assertTrue(any(x['start']==16 and x['end']>16 for x in regions))

    def test_simultaneous_speech_is_localized(self):
        people=[dict(words=[word(t,t+.9) for t in range(1,10)]),dict(words=[word(5,5.5)])]
        regions,gaps=r.role_regions(people,12)
        self.assertTrue(any(x['end']<=5 for x in regions))
        self.assertTrue(any(x['start']>=5.5 for x in regions))
        self.assertTrue(any(x['reason']=='simultaneous_content_speech' for x in gaps))

    def test_coarse_envelope_needs_word_and_vad_clearance(self):
        s=dict(start=1,end=15,word_count=8,feedback=False,word_aligned=True,
               words=[word(1,1.5),word(14,15)])
        blocks=r.semantic_blocks(dict(segments=[s],vad=[(1,2),(13,15)]))
        self.assertEqual(r.subtract([(1,15)],blocks),[(2.15,12.85)])
        bad=copy.deepcopy(s);bad['word_aligned']=False
        self.assertEqual(r.subtract([(1,15)],r.semantic_blocks(dict(segments=[bad],vad=[]))),[])

    def test_audio_windows_offset_and_global_reason_blocks_salvage(self):
        row=dict(start=100,end=110,duration_s=10,audio_final=dict(pool='audio_reject',reasons=['local'],quality=dict(
            hard_reasons=['local'],review_reasons=[],evidence_windows=[dict(start=3,end=4,reason='local')])))
        spans,kind=r.quality_blocks(row)
        self.assertEqual(kind,'localized_quality')
        self.assertEqual(spans,[[102.85,104.15]])
        row['audio_final']['quality']['hard_reasons'].append('global_floor')
        self.assertEqual(r.quality_blocks(row),([(100,110)],'nonlocalizable_quality'))

    def test_human_rejection_cannot_be_bypassed_by_local_noise(self):
        row=dict(start=100,end=110,duration_s=10,audio_final=dict(pool='audio_reject',reasons=['human_repair_reserve'],
                 quality=dict(hard_reasons=['local'],evidence_windows=[dict(start=3,end=4,reason='local')])))
        self.assertEqual(r.quality_blocks(row)[1],'human_or_task_hold')

    def test_router_omitted_windows_are_recovered_from_qc_evidence(self):
        row=dict(start=100,end=110,duration_s=10,
                 audio_final=dict(pool='audio_reject',reasons=['strong_reference_match'],quality=dict(
                     hard_reasons=['strong_reference_match'],review_reasons=['local_match_unresolved'],evidence_windows=[])),
                 qc4=dict(reference=dict(windows=[dict(start=2,end=3,suspicious=True),dict(start=7,end=8,suspicious=False)])),
                 qc3=dict(listener=dict(events=[dict(start=5,end=5.5,decision=dict(reasons=['possible_local_speaker_match']))],background=[])))
        spans,kind=r.quality_blocks(row)
        self.assertEqual(kind,'localized_quality')
        self.assertEqual(spans,[[101.85,103.15],[104.85,105.65]])
        row['audio_final']['quality']['review_reasons'].append('stationary_background_unresolved')
        self.assertEqual(r.quality_blocks(row)[1],'nonlocalizable_quality')

    def test_long_sentence_emits_multiple_children_and_labels_fragments(self):
        words=[word(.5+t*.5,.8+t*.5,'word.' if t==59 else 'word') for t in range(60)]
        people=[dict(words=words,segments=[],vad=[]),dict(words=[],segments=[],vad=[])]
        def cut(unit,people,rms,si,limit,left,right):
            a=max(left,unit['start']-.1);b=min(right,unit['end']+.1)
            return dict(start=a,end=b,duration_s=b-a),None
        with patch.object(r.old,'acoustic_cut',side_effect=cut):
            clips,bad=r.recover_region(people,[np.zeros(4000)]*2,0,0,31,31)
        self.assertGreaterEqual(len(clips),3)
        self.assertTrue(all(2<=c['duration_s']<=11.95 for c in clips))
        self.assertTrue(all(not c['recovery_complete_sentence'] for c in clips))
        self.assertTrue(all(a['end']<=b['start'] for a,b in zip(clips,clips[1:])))

    def test_failed_long_group_retries_smaller_unit(self):
        words=[word(.5+t*.5,.8+t*.5,'word.' if t in (9,19) else 'word') for t in range(20)]
        people=[dict(words=words,segments=[],vad=[]),dict(words=[],segments=[],vad=[])]
        def cut(unit,people,rms,si,limit,left,right):
            if unit['end']-unit['start']>6:return None,'tail_conflict'
            a=max(left,unit['start']-.1);b=min(right,unit['end']+.1)
            return dict(start=a,end=b,duration_s=b-a),None
        with patch.object(r.old,'acoustic_cut',side_effect=cut):
            clips,bad=r.recover_region(people,[np.zeros(1200)]*2,0,0,11,11)
        self.assertEqual(len(clips),2)
        self.assertTrue(all(c['recovery_complete_sentence'] for c in clips))

    def test_real_pcm_boundaries_keep_multiple_children_without_cutting_words(self):
        words=[word(1+t*.5,1.25+t*.5,'word.' if t==49 else 'word') for t in range(50)]
        people=[dict(words=words,segments=[],vad=[]),dict(words=[],segments=[],vad=[])]
        rms=[np.full(3000,1e-5),np.full(3000,1e-5)]
        for w in words:rms[0][round(w['start']*100):round(w['end']*100)]=.1
        clips,bad=r.recover_region(people,rms,0,0,28,30)
        self.assertGreaterEqual(len(clips),3)
        self.assertTrue(all(2<=c['duration_s']<=11.95 for c in clips))
        self.assertTrue(all(not w['start']<edge<w['end'] for c in clips for edge in [c['start'],c['end']] for w in words))
        self.assertTrue(all(a['end']<=b['start'] for a,b in zip(clips,clips[1:])))

    def test_detector_reads_actual_hard_cut_frame_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'cuts.mkv')
            subprocess.run(['ffmpeg','-v','error','-y','-filter_complex_threads','1',
                '-f','lavfi','-i','color=red:size=320x180:rate=25:duration=2',
                '-f','lavfi','-i','color=blue:size=320x180:rate=25:duration=2',
                '-f','lavfi','-i','color=white:size=320x180:rate=25:duration=2',
                '-filter_complex','[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]',
                '-map','[v]','-c:v','ffv1','-threads','1',path],check=True)
            with patch.object(r.old.base.media,'uri',return_value=path):
                cuts=r.detect_shots(dict(source={},duration=6))
            self.assertEqual(cuts,[2.,4.])


if __name__=='__main__':unittest.main()
