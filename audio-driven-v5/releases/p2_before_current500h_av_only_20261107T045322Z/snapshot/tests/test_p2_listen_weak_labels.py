import copy
import unittest

import torch

from p2_listen.annotations import class_statistics, rasterize, validate_reference
from p2_listen.candidates import CANDIDATE_SCHEMA, MAPPING_VERSION
from p2_listen.contracts import EVENTS, VISUAL_TYPES, digest
from p2_listen.events import event_sums, normalized_event_loss
from p2_listen.layout import make_layout
from p2_listen.training import supervision_policy
from p2_listen.weak_labels import POLICY, background_seconds, derive_sample, full_receipt


def fixtures():
    c = dict(schema=CANDIDATE_SCHEMA, sample_id="s1", duration_s=1.,
             events=[dict(event_id="e1", modality="visual", categories=["head_nod"],
                          start=.25, end=.5, description="original nod", review_flags=["raw_model_unverified"])],
             automatic_fusion={}, unresolved_observations=[])
    raw = dict(schema="listen-open-semantic-events-v1", version="open-to-visual6-audio1-semantic-v3",
               sample_id="s1", duration_s=1., events=[], training_admitted=False, K_train=[],
               unlabeled_time_policy="unknown", coverage={k:[] for k in EVENTS}, unknown={k:[[0.,1.]] for k in EVENTS})
    cfg = {"events":dict(admission_policy=POLICY, mapping_version=MAPPING_VERSION)}
    target = {"media_hashes":dict(target_video="a"*64, target_audio="b"*64)}
    media = dict(source_start_s=4., source_video={"member":"listener.mp4"},
                 video_transform=dict(crop={"x":0,"y":0,"width":600,"height":600},output_size=[512,512]),
                 audio_sources={"listener":dict(declared_source_sha256="c"*64,start_sample=64000,end_sample=80000,sample_rate=16000)})
    receipts = {}
    for branch in ("visual","audio"):
        r = dict(record_kind="result",sample_id="s1",view="full",request_id="s1@full",branch=branch,
                 status="schema_pass",time_origin="window_start_seconds",window_start_s=0.,window_end_s=1.,
                 parent_duration_s=1.,parsed={"events":[]},raw_response={"finish_reason":"stop"},
                 source=dict(sample_id="s1",duration_s=1.,source_start_s=4.,source=media["source_video"],
                             listener_id="l1",speaker_id="p1"))
        r["input_evidence"] = {"video":dict(source=media["source_video"],video_transform=media["video_transform"],
            source_time_origin_s=4.,frames=[dict(time_s=t,size=[512,512],jpeg_sha256="d"*64) for t in (0.,.25,.5,.75)])} if branch=="visual" else {
            "audio":dict(role="listener",isolated=True,wav_sha256="b"*64,source_waveform_sha256_declared="c"*64,
                         source_start_sample=64000,source_end_sample=80000,source_rate=16000)}
        receipts[branch] = r
    return c,raw,cfg,target,receipts,media


class WeakLabels(unittest.TestCase):
    def setUp(self):
        self.c,self.raw,self.cfg,self.target,self.receipts,self.media = fixtures()

    def derive(self, replace=True):
        return derive_sample(self.c,self.raw,self.cfg,self.target,self.receipts,
                             debug_only=True,replace_placeholder=replace)[0]

    def test_event_background_and_fraction_are_not_intensity(self):
        self.c["events"][0]["end"] = .375
        l = self.derive(); r = rasterize(l,make_layout(1.,(1,1)))
        self.assertEqual(r["head_nod"]["y"].tolist(),[[0.,.5,0.,0.]])
        self.assertTrue(r["head_nod"]["known"].all())
        self.assertAlmostEqual(background_seconds(l,"head_nod"),.875)
        self.assertEqual(l["provenance"]["background_weight"],1.)

    def test_empty_success_means_weak_background(self):
        self.c["events"] = []
        r = rasterize(self.derive(),make_layout(1.,(1,1)))
        self.assertTrue(all(x["known"].all() and not x["y"].any() for x in r.values()))

    def test_missing_audio_does_not_disable_visual(self):
        self.receipts["audio"] = None
        r = rasterize(self.derive(),make_layout(1.,(1,1)))
        self.assertTrue(r["head_nod"]["known"].all())
        self.assertFalse(r["listener_vocalization"]["known"].any())

    def test_placeholder_requires_explicit_opt_in(self):
        r = rasterize(self.derive(False),make_layout(1.,(1,1)))
        self.assertFalse(any(x["known"].any() for x in r.values()))

    def test_actual_unknown_never_cleared_and_padding_masked(self):
        self.raw["unknown"] = {k:[] for k in EVENTS}
        self.raw["unknown"]["head_nod"] = [[.5,.75]]
        l = self.derive(); layout = make_layout(1.25,(1,1));r = rasterize(l,layout)
        self.assertEqual(r["head_nod"]["known"].tolist(),[[True,True,False,True,False]])
        self.assertFalse(l["provenance"]["placeholder_unknown_replaced"])

    def test_unknown_schema_full_unknown_is_preserved(self):
        self.raw["version"] = "unrecognized-version"
        self.assertEqual(self.derive()["unknown"],self.raw["unknown"])

    def test_uncertain_event_is_masked_not_rejected_as_background(self):
        self.c["events"][0]["review_flags"].append("content_conflict")
        l = self.derive();r=rasterize(l,make_layout(1.,(1,1)))
        self.assertEqual(l["events"],[])
        self.assertEqual(r["head_nod"]["known"].tolist(),[[True,False,True,True]])

    def test_censored_edge_and_unlocated_audio_are_masked(self):
        self.c["events"][0]["boundary_censored"] = True
        self.c["automatic_fusion"]["alignment_failures"] = [dict(text="unlocated")]
        r = rasterize(self.derive(),make_layout(1.,(1,1)))
        self.assertFalse(r["head_nod"]["known"][0,1])
        self.assertFalse(r["listener_vocalization"]["known"].any())
        self.assertTrue(r["smile"]["known"].all())

    def test_invalid_time_masks_whole_class_not_clipped(self):
        self.c["events"][0]["end"] = 2.
        r = rasterize(self.derive(),make_layout(1.,(1,1)))
        self.assertFalse(r["head_nod"]["known"].any())

    def test_raw_is_immutable_and_debug_cannot_be_formal_or_reference(self):
        before = digest([self.c,self.raw,self.receipts])
        l=self.derive()
        self.assertEqual(before,digest([self.c,self.raw,self.receipts]))
        self.assertEqual(l["events"][0]["source"],self.c["events"][0])
        self.assertFalse(l["provenance"]["human_verified"])
        self.assertFalse(l["provenance"]["formal_training_admitted"])
        with self.assertRaises(ValueError): validate_reference(l)
        stats=class_statistics([rasterize(l,make_layout(1.,(1,1)))])
        with self.assertRaises(ValueError): supervision_policy(stats,EVENTS,[l])
        self.assertTrue(stats["head_nod"]["enabled"])
        self.assertEqual(stats["head_shake"]["disabled_reason"],"no_positive_support")

    def test_background_and_event_gradients_opposite_unknown_zero(self):
        self.raw["unknown"] = {k:[] for k in EVENTS}
        self.raw["unknown"]["head_nod"] = [[.5,.75]]
        r=rasterize(self.derive(),make_layout(1.,(1,1)))
        logits={k:torch.zeros_like(v["y"],requires_grad=True) for k,v in r.items()}
        sums,counts=event_sums(logits,r,{k:1 for k in EVENTS})
        normalized_event_loss(sums,counts).backward()
        g=logits["head_nod"].grad[0]
        self.assertGreater(float(g[0]),0.)
        self.assertLess(float(g[1]),0.)
        self.assertEqual(float(g[2]),0.)

    def test_explicit_policy_required(self):
        self.cfg["events"]["admission_policy"]="explicit-review-v1"
        with self.assertRaises(ValueError): self.derive()

    def test_individual_debug_record_cannot_be_promoted(self):
        self.raw["provenance"]={"debug_only":True}
        l,_,_=derive_sample(self.c,self.raw,self.cfg,self.target,self.receipts,
                            debug_only=False,replace_placeholder=True)
        self.assertTrue(l["provenance"]["debug_only"])
        self.assertFalse(l["provenance"]["formal_training_admitted"])

    def test_always_on_prediction_is_penalized_and_event_gaps_stay_zero(self):
        self.c["events"].append(dict(self.c["events"][0],event_id="e2",start=.75,end=1.))
        r=rasterize(self.derive(),make_layout(1.,(1,1)))
        self.assertEqual(r["head_nod"]["y"].tolist(),[[0.,1.,0.,1.]])
        y=r["head_nod"]["y"]
        always_on=torch.nn.functional.binary_cross_entropy_with_logits(torch.full_like(y,8.),y)
        timed=torch.nn.functional.binary_cross_entropy_with_logits(y*16.-8.,y)
        self.assertGreater(float(always_on),float(timed)+1.)


class ReceiptEvidence(unittest.TestCase):
    def setUp(self):
        _,_,_,self.target,self.receipts,self.media=fixtures()

    def check(self,branch,rr=None):
        return full_receipt(rr if rr is not None else [self.receipts[branch]],branch,"s1",1.,self.media,self.target)

    def test_valid_full_video_and_listener_audio(self):
        for branch in self.receipts:
            self.assertIsNotNone(self.check(branch)[0])

    def test_empty_failed_truncated_or_partial_input_not_background(self):
        self.assertIsNone(self.check("audio",[])[0])
        for key,value in (("status","failed"),("window_end_s",.5),("view","w000")):
            r=copy.deepcopy(self.receipts["audio"]);r[key]=value
            self.assertIsNone(self.check("audio",[r])[0])
        self.receipts["audio"]["raw_response"]["finish_reason"]="length"
        self.assertIsNone(self.check("audio")[0])

    def test_speaker_audio_or_wrong_hash_not_used(self):
        for key,value in (("role","speaker"),("isolated",False),("wav_sha256","f"*64)):
            r=copy.deepcopy(self.receipts["audio"]);r["input_evidence"]["audio"][key]=value
            self.assertIsNone(self.check("audio",[r])[0])

    def test_bad_crop_size_or_missing_frames_not_used(self):
        r=copy.deepcopy(self.receipts["visual"]);r["input_evidence"]["video"]["frames"] = []
        self.assertIsNone(self.check("visual",[r])[0])
        r=copy.deepcopy(self.receipts["visual"]);r["input_evidence"]["video"]["frames"][0]["size"]=[384,384]
        self.assertIsNone(self.check("visual",[r])[0])

    def test_mismatched_source_and_duplicate_success_fail_closed(self):
        r=copy.deepcopy(self.receipts["audio"]);r["source"]["sample_id"]="wrong"
        with self.assertRaises(ValueError): self.check("audio",[r])
        with self.assertRaises(ValueError): self.check("audio",[self.receipts["audio"]]*2)


if __name__ == "__main__":
    unittest.main()
