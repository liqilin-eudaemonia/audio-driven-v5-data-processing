import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_listen_qc3_bulk import human_decision


class HumanAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(pair='pair',listener_id='listener',start=10.,end=18.)
        self.positive=dict(self.row,decision='retain_valid_listener_response')
        self.negative=dict(listener_id='listener',start=16.,end=20.,decision='reject')

    def test_confirmed_reaction_overrides_review_only_for_exact_source_clip(self):
        self.assertEqual(human_decision(self.row,'review',[],[self.positive])[0],'keep')
        for changed in [dict(start=9.9),dict(end=18.1),dict(pair='other'),dict(listener_id='other')]:
            self.assertEqual(human_decision(dict(self.row,**changed),'review',[],[self.positive])[0],'review')

    def test_negative_source_interval_survives_recut_and_has_priority(self):
        self.assertEqual(human_decision(self.row,'keep',[self.negative],[self.positive])[0],'exclude')
        self.assertEqual(human_decision(dict(self.row,start=17.),'keep',[self.negative],[])[0],'exclude')
        self.assertEqual(human_decision(dict(self.row,end=16.),'keep',[self.negative],[])[0],'keep')

    def test_unreviewed_data_keeps_automatic_triage(self):
        for state in ['keep','review','exclude']:
            self.assertEqual(human_decision(self.row,state,[],[]),(state,[]))


if __name__=='__main__':unittest.main()
