"""Guards for real-model S0 fixtures; these unit tests do not allocate a GPU."""
import copy
import unittest
from p2_listen.contracts import EVENTS
from p2_listen.training import supervision_policy


class DebugSupervision(unittest.TestCase):
    def fixtures(self):
        stats={k:dict(positive=4.,negative=0.,count=4,enabled=False,
                      disabled_reason="no_negative_support",pos_weight=1.) for k in EVENTS}
        stats["head_shake"].update(positive=0.,count=0,disabled_reason="no_known_support")
        return stats,[{"provenance":{"debug_only":True,"formal_training_admitted":False}}]

    def test_formal_training_rejects_debug_labels(self):
        stats,labels=self.fixtures()
        with self.assertRaisesRegex(ValueError,"formal training"):
            supervision_policy(stats,list(EVENTS),labels)

    def test_positive_override_requires_debug_s0(self):
        stats,labels=self.fixtures()
        with self.assertRaises(ValueError):
            supervision_policy(stats,list(EVENTS),labels,debug_positive_events=True)
        with self.assertRaises(ValueError):
            supervision_policy(stats,list(EVENTS),[{"provenance":{}}],s0=True,debug_positive_events=True)

    def test_debug_mode_keeps_unknown_class_disabled_and_statistics_unchanged(self):
        stats,labels=self.fixtures();before=copy.deepcopy(stats)
        result=supervision_policy(stats,list(EVENTS),labels,s0=True,debug_positive_events=True)
        self.assertEqual(stats,before)
        self.assertFalse(result["head_shake"]["enabled"])
        for k in EVENTS:
            self.assertEqual(result[k]["negative"],0)
            if k!="head_shake":
                self.assertTrue(result[k]["enabled"])
                self.assertEqual(result[k]["pos_weight"],1.)

    def test_s0_without_override_keeps_formal_class_policy(self):
        stats,labels=self.fixtures()
        self.assertEqual(supervision_policy(stats,list(EVENTS),labels,s0=True),stats)


if __name__=="__main__":unittest.main()
