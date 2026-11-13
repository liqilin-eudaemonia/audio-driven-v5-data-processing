import copy
import unittest
from p2_listen.evaluation import event_counts,evaluate
from test_p2_listen_core import obs,label
from p2_listen.contracts import EVENTS


class Evaluation(unittest.TestCase):
    def test_duplicate_prediction_is_false_positive(self):
        l=label();p=[l["events"][1],copy.deepcopy(l["events"][1])]
        c=event_counts(l,p,"head_nod")
        self.assertEqual((c["tp"],c["fp"],c["fn"]),(1,1,0))

    def test_unknown_censored_reference(self):
        l=label();l["unknown"]["head_nod"]=[[.3,.4]]
        c=event_counts(l,l["events"],"head_nod")
        self.assertEqual(c["reference_count"],0);self.assertEqual(c["excluded_reference"],1)

    def test_cross_unknown_still_penalizes_negative_occupancy(self):
        l=label();l["unknown"]["head_nod"]=[[.6,.7]]
        c=event_counts(l,[dict(type="head_nod",start=.5,end=.8)],"head_nod")
        self.assertEqual(c["censored_predictions"],1)
        self.assertAlmostEqual(c["false_seconds"],.2)

    def matrix(self):
        p=dict(schema="p2-listen-evaluation-v2-visual6-audio1",taxonomy=list(EVENTS),K_train=list(EVENTS),K_primary=["head_nod","listener_vocalization"],sample_ids=["s1"],sample_seeds=[1,2],train_seeds=[7],arms=["E0","E1"],bootstrap_repeats=5)
        r=[]
        for arm in p["arms"]:
            for seed in p["sample_seeds"]:
                r.append(dict(sample_id="s1",arm=arm,train_seed=7,sample_seed=seed,status="success",independent_verified=True,events=label()["events"]))
        return p,r

    def test_failures_count_as_false_negatives(self):
        p,r=self.matrix();r[0].update(status="failed",events=[])
        result=evaluate(p,[obs()],[label()],r)
        e0=result["training_seeds"]["7"]["E0"]
        self.assertEqual(e0["failure_rate"],.5)
        self.assertAlmostEqual(e0["macro_f1"],2/3)
        self.assertAlmostEqual(result["mean_delta_macro_f1"],1/3)

    def test_missing_result_rejected(self):
        p,r=self.matrix()
        with self.assertRaises(ValueError):evaluate(p,[obs()],[label()],r[:-1])

    def test_duplicate_result_rejected(self):
        p,r=self.matrix()
        with self.assertRaises(ValueError):evaluate(p,[obs()],[label()],r+[r[0]])

    def test_trained_head_not_accepted_as_judge(self):
        p,r=self.matrix();r[0]["independent_verified"]=False
        with self.assertRaises(ValueError):evaluate(p,[obs()],[label()],r)

    def test_no_reference_positives_not_perfect_score(self):
        p,r=self.matrix();l=label();l["events"]=[]
        for x in r:x["events"]=[]
        self.assertIsNone(evaluate(p,[obs()],[l],r)["training_seeds"]["7"]["E0"]["macro_f1"])


if __name__=="__main__":unittest.main()
