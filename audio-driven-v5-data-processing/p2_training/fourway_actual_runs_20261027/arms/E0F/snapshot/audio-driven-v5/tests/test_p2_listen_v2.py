import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from p2_listen import VERSION
from p2_listen.annotations import (class_statistics, mask_classes, rasterize, validate,
                                  validate_contract, validate_reference)
from p2_listen.cache import load_part, save_part
from p2_listen.candidates import (CANDIDATE_SCHEMA, DECISION_SCHEMA, convert_sample,
                                 export_candidates)
from p2_listen.config import load
from p2_listen.contracts import EVENTS, VISUAL_TYPES, LABEL_SCHEMA, ROOT, atomic_json, digest, sha
from p2_listen.events import EventHeads, event_sums, normalized_event_loss
from p2_listen.evaluation import evaluate, event_counts
from p2_listen.features import identity
from p2_listen.layout import make_layout
from p2_listen.training import restore_checkpoint
from test_p2_listen_core import label, obs
import test_p2_listen_evaluation as evaluation_tests


def candidate():
    return dict(schema=CANDIDATE_SCHEMA, sample_id="s1", duration_s=1., training_admitted=False,
                events=[dict(event_id=f"e{i}", modality="audio", categories=["listener_vocalization"],
                             start=a, end=b, description=text, text=text, original_descriptions=[text],
                             review_flags=["boundary_review"], status={"boundary":"unverified"},
                             description_source={"pass_id":"full", "word":i},
                             timing_evidence={"start":a,"end":b,"window_offset_already_applied":True})
                        for i,(a,b,text) in enumerate(((.1,.2,"Okay"),(.3,.4,"Okay"),(.5,.6,"嘿"),(.7,.8,"Ed")))],
                automatic_fusion={"alignment_failures":[]}, unresolved_observations=[])


def decision(c):
    return dict(schema=DECISION_SCHEMA, sample_id=c["sample_id"], candidate_sha256=digest(c),
                coverage={k:[[0.,1.]] for k in EVENTS}, unknown={k:[] for k in EVENTS},
                coverage_sources={k:{"reference":"human-review-1", "verification":"human"} for k in EVENTS},
                media_hashes={"target_video":"a"*64,"target_audio":"b"*64},
                admission_record={"reference":"human-review-1"}, human_verified=True, source_review_reference="human-review-1",
                events=[dict(candidate_event_id=e["event_id"],decision="accepted",reason="Reviewed media and original source",
                             verification="human",boundary_status="verified",reviewed_flags=e["review_flags"])
                        for e in c["events"]])


class SevenChannels(unittest.TestCase):
    def test_all_seven_outputs_route_to_correct_modality(self):
        torch.manual_seed(5)
        heads=EventHeads(12,7,8)
        layout=make_layout(1.,(2,2))
        u=SimpleNamespace(video=torch.randn(1,16,12,requires_grad=True),audio=torch.randn(1,8,7,requires_grad=True),
                          video_valid=layout["video_valid"],audio_valid=layout["audio_valid"])
        out=heads(u,layout,"joint_event")
        self.assertEqual(list(out),list(EVENTS))
        self.assertEqual(heads.heads["video"][-1].out_features,6)
        self.assertEqual(heads.heads["audio"][-1].out_features,1)
        for k in EVENTS:
            j=0 if k in VISUAL_TYPES else 1
            self.assertEqual(out[k].shape,(1,4 if j==0 else 8))
            gradients=torch.autograd.grad(out[k].sum(),[u.video,u.audio],retain_graph=True,allow_unused=True)
            self.assertGreater(gradients[j].abs().sum(),0)
            self.assertIsNone(gradients[1-j])
        for k,v in heads(u,layout,"probe_only").items():
            grads=torch.autograd.grad(v.sum(),[u.video,u.audio],retain_graph=True,allow_unused=True)
            self.assertEqual(grads,(None,None))

    def test_nod_smile_voice_overlap_with_different_grids(self):
        l=label()
        for e in l["events"]: e.update(start=.25,end=.5)
        l["events"].append(dict(l["events"][1],event_id="smile1",type="smile"))
        r=rasterize(validate(l,obs()),make_layout(1.,(1,1)))
        self.assertEqual(set(r),set(EVENTS))
        self.assertEqual(float(r["head_nod"]["y"][0,1]),1.)
        self.assertEqual(float(r["smile"]["y"][0,1]),1.)
        self.assertEqual(r["listener_vocalization"]["y"][0,2:4].tolist(),[1.,1.])

    def test_loss_equal_class_weight_not_equal_modality_weight(self):
        logits={k:torch.full((1,4 if k in VISUAL_TYPES else 8),float(i)/4,requires_grad=True) for i,k in enumerate(EVENTS)}
        labels={k:dict(y=torch.zeros_like(v),known=torch.ones_like(v,dtype=torch.bool)) for k,v in logits.items()}
        sums,counts=event_sums(logits,labels,{k:1 for k in EVENTS})
        loss=normalized_event_loss(sums,counts)
        expected=sum(torch.nn.functional.softplus(v).mean() for v in logits.values())/7
        self.assertTrue(torch.allclose(loss,expected))
        labels["surprise"]["known"].zero_()
        sums,counts=event_sums(logits,labels,{k:1 for k in EVENTS})
        expected=sum(torch.nn.functional.softplus(v).mean() for k,v in logits.items() if k!="surprise")/6
        self.assertTrue(torch.allclose(normalized_event_loss(sums,counts),expected))

    def test_structural_padding_and_disabled_class_mask(self):
        layout=make_layout(1.,(1,1));layout["video_valid"][:,1]=False
        r=rasterize(label(),layout,["head_nod","listener_vocalization"])
        self.assertFalse(r["head_nod"]["known"][0,1])
        self.assertTrue(r["listener_vocalization"]["known"].all())
        self.assertTrue(all(not r[k]["known"].any() for k in VISUAL_TYPES if k!="head_nod"))

    def test_sparse_support_records_effective_classes_and_does_not_mutate_cache(self):
        r=rasterize(label(),make_layout(1.,(1,1)))
        r["frown"]["known"].zero_()
        r["laugh"]["y"].fill_(1)
        stats=class_statistics([r])
        enabled=[k for k in EVENTS if stats[k]["enabled"]]
        self.assertEqual(enabled,["head_nod","listener_vocalization"])
        self.assertEqual(stats["frown"]["disabled_reason"],"no_known_support")
        self.assertEqual(stats["laugh"]["disabled_reason"],"no_negative_support")
        self.assertEqual(stats["smile"]["disabled_reason"],"no_positive_support")
        masked=mask_classes(r,enabled)
        self.assertFalse(masked["smile"]["known"].any())
        self.assertTrue(r["smile"]["known"].all())

    def test_fractional_positive_weight_uses_frozen_train_support(self):
        r=rasterize(label(),make_layout(1.,(1,1)))
        r["head_nod"]["y"].zero_();r["head_nod"]["y"][0,0]=.5
        self.assertEqual(class_statistics([r])["head_nod"]["pos_weight"],3.5)

    def test_missing_seventh_key_is_an_error(self):
        r=rasterize(label(),make_layout(1.,(1,1)))
        logits={k:torch.zeros_like(v["y"]) for k,v in r.items() if k!="surprise"}
        with self.assertRaises(ValueError):event_sums(logits,r,{k:1 for k in EVENTS})

    def test_all_unknown_still_updates_base_model(self):
        from p2_listen.audit import fixture
        from p2_listen.model import ListeningExtractor
        from p2_listen.training import optimization_step
        cfg,sample=fixture(load());sample=(*sample[:4],mask_classes(sample[4],[]))
        m=ListeningExtractor(cfg,8)
        t=dict(connector_lr=.001,feature_lr=.001,event_lr=.001,clip_norm=1.)
        opt=torch.optim.AdamW(m.optimizer_groups(t),weight_decay=0)
        before=copy.deepcopy(m.state_dict())
        r=optimization_step(m,opt,[sample],lambda u,_:(u.video-.2).square().mean()+(u.audio+.1).square().mean(),
                            mode="joint_event",pos_weights={k:1. for k in EVENTS},event_weight=.05,seed=1,ordinals=[0],config=t,audit=True)
        self.assertEqual(r["event"],0)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in m.state_dict().items() if k.startswith("extractor.")))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.state_dict().items() if k.startswith("event_heads.")))


class VersionAndIsolation(unittest.TestCase):
    def test_old_config_and_label_rejected(self):
        with self.assertRaises(ValueError):load(ROOT/"configs/p2_listen_event_base.json")
        l=label();l["schema"]="listen-events-v1"
        with self.assertRaises(ValueError):validate(l,obs())

    def test_invalid_taxonomy_order_or_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"config.json"
            for keys in (["listener_vocalization","head_nod"],["head_nod","head_nod"],["unknown"]):
                c=load();c["events"]["K_train"]=keys;atomic_json(p,c)
                with self.assertRaises(ValueError):load(p)

    def test_old_checkpoint_refused_even_when_identity_claims_to_match(self):
        m=torch.nn.Linear(2,2);before=copy.deepcopy(m.state_dict())
        with self.assertRaises(ValueError):restore_checkpoint(dict(identity={}),m,None,{})
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.state_dict().items()))

    def test_seven_class_checkpoint_wrong_taxonomy_refused(self):
        saved=dict(version=VERSION,label_schema=LABEL_SCHEMA,taxonomy=list(reversed(EVENTS)),identity={})
        with self.assertRaises(ValueError):restore_checkpoint(saved,torch.nn.Linear(2,2),None,{})

    def test_label_mapping_and_k_train_change_only_event_cache(self):
        c=load();l=label();o=obs()
        fake=SimpleNamespace(to_dict=lambda:copy.deepcopy(c["extractor"]))
        with patch("p2_listen.features.weights_identity",return_value={"mock":"stable"}),patch("p2_listen.features.extractor_config",return_value=fake):
            h=identity(c,"input",o);e=identity(c,"events",o,label=l)
            c2=copy.deepcopy(c);c2["events"]["K_train"]=["head_nod"]
            c2["events"]["mapping_version"]="new-mapping"
            l2=copy.deepcopy(l);l2["unknown"]["head_nod"]=[[0.,1.]]
            self.assertEqual(h,identity(c2,"input",o))
            e2=identity(c2,"events",o,label=l2)
            self.assertNotEqual(e,e2)
            with tempfile.TemporaryDirectory() as tmp:
                save_part(Path(tmp)/"input",h,{"h":torch.zeros(1)})
                save_part(Path(tmp)/"events",e,{"events":torch.zeros(1)})
                load_part(Path(tmp)/"input",identity(c2,"input",o))
                with self.assertRaises(ValueError):load_part(Path(tmp)/"events",e2)
            o2=copy.deepcopy(o);o2["media_hashes"]["speaker_audio"]="new"
            self.assertNotEqual(h,identity(c,"input",o2))

    def test_admitted_boolean_without_evidence_is_rejected(self):
        l=label();l["provenance"]["admission_record"]={}
        with self.assertRaises(ValueError):validate(l,obs())
        l=label();l["provenance"]["coverage_sources"]["smile"]={}
        with self.assertRaises(ValueError):validate(l,obs())

    def test_weak_reference_and_mapping_mismatch_refused(self):
        l=label();l["events"][0]["verification"]="weak_accepted"
        with self.assertRaises(ValueError):validate_reference(l)
        l=label();l["provenance"]["mapping_version"]="other"
        with self.assertRaises(ValueError):validate_contract(l,load())


class CandidateAdapter(unittest.TestCase):
    def test_partial_review_exports_separate_admitted_file(self):
        c=candidate();c2=copy.deepcopy(c);c2["sample_id"]="s2"
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            atomic_json(p/"c.json",[c,c2]);atomic_json(p/"r.json",{"samples":[c,c2]})
            atomic_json(p/"o.json",[dict(sample_id=x["sample_id"],duration_s=1.,events=x["events"]) for x in (c,c2)])
            atomic_json(p/"d.json",[decision(c)])
            summary=export_candidates(load(),p/"c.json",p/"r.json",p/"o.json",p/"out",p/"d.json")
            self.assertEqual(summary["admitted_samples"],1)
            admitted=json.loads((p/"out/labels.admitted.json").read_text())
            self.assertEqual([l["sample_id"] for l in admitted],["s1"])
            self.assertEqual(len(json.loads((p/"out/labels.json").read_text())),2)
            validate(admitted[0],obs())

    def test_no_decisions_no_negatives_no_admission(self):
        c=candidate();before=copy.deepcopy(c)
        l,a=convert_sample(c,load())
        self.assertEqual(c,before)
        self.assertFalse(l["training_admitted"])
        self.assertEqual(l["events"],[])
        self.assertEqual(len(a["mapping"]),4)
        self.assertTrue(all(not v["known"].any() for v in rasterize(l,make_layout(1.,(1,1))).values()))
        with self.assertRaises(ValueError):validate(l,obs())

    def test_original_words_repeat_count_sources_and_times_survive(self):
        c=candidate();l,a=convert_sample(c,load(),decision(c))
        validate(l,obs());validate_reference(l)
        self.assertEqual([e["description"] for e in l["events"]],["Okay","Okay","嘿","Ed"])
        self.assertEqual([e["source"] for e in l["events"]],c["events"])
        self.assertEqual([(e["start"],e["end"]) for e in l["events"]],[(e["start"],e["end"]) for e in c["events"]])
        self.assertEqual(len({m["training_event_id"] for m in a["mapping"]}),4)

    def test_candidate_boolean_does_not_admit_and_stale_decision_rejected(self):
        c=candidate();c["training_admitted"]=True
        self.assertFalse(convert_sample(c,load())[0]["training_admitted"])
        d=decision(c);c["events"][0]["text"]="changed"
        with self.assertRaises(ValueError):convert_sample(c,load(),d)

    def test_review_flags_cannot_be_ignored(self):
        c=candidate();d=decision(c);d["events"][0]["reviewed_flags"]=[]
        with self.assertRaises(ValueError):convert_sample(c,load(),d)

    def test_missing_candidate_decision_masks_its_interval(self):
        c=candidate();d=decision(c);d["events"]=d["events"][1:]
        l,a=convert_sample(c,load(),d)
        self.assertIn([.1,.2],l["unknown"]["listener_vocalization"])
        self.assertEqual(len(l["events"]),3)
        self.assertEqual(a["mapping"][0]["status"],"unresolved")

    def test_invalid_times_are_not_clipped_or_interpolated(self):
        for a,b in ((.2,.2),(-.1,.2),(.1,1.01),(float("nan"),.2)):
            c=candidate();c["events"][0].update(start=a,end=b)
            if math_isfinite(a):
                with self.assertRaises(ValueError):convert_sample(c,load(),decision(c))
            else:
                with self.assertRaises(ValueError):convert_sample(c,load())

    def test_modality_mismatch_and_compound_categories_not_inferred(self):
        for categories in (["smile"],["head_nod","smile"]):
            c=candidate();c["events"][0]["categories"]=categories
            with self.assertRaises(ValueError):convert_sample(c,load(),decision(c))

    def test_unlocated_word_masks_audio_without_inventing_boundaries(self):
        c=candidate();c["automatic_fusion"]["alignment_failures"]=[{"text":"Yeah", "reason":"no_interval"}]
        l,a=convert_sample(c,load(),decision(c))
        self.assertEqual(l["unknown"]["listener_vocalization"],[[0.,1.]])
        self.assertFalse(rasterize(l,make_layout(1.,(1,1)))["listener_vocalization"]["known"].any())
        self.assertEqual(a["excluded"][-1]["reason"],"unlocated_words")

    def test_export_preserves_input_bytes_and_rejects_overwrite(self):
        c=candidate()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            atomic_json(p/"c.json",[c]);atomic_json(p/"r.json",{"samples":[c]})
            atomic_json(p/"o.json",[{"sample_id":"s1","duration_s":1.,"events":c["events"]}])
            summary=export_candidates(load(),p/"c.json",p/"r.json",p/"o.json",p/"out")
            self.assertEqual(summary["accepted_events"],0)
            self.assertEqual(sha(p/"c.json"),sha(p/"out/sources/candidates.json"))
            with self.assertRaises(FileExistsError):export_candidates(load(),p/"c.json",p/"r.json",p/"o.json",p/"out")


def math_isfinite(x):
    import math
    return math.isfinite(x)


class FrozenEvaluation(unittest.TestCase):
    def test_single_primary_reports_class_f1(self):
        p,r=evaluation_tests.Evaluation().matrix();p["K_primary"]=["head_nod"]
        result=evaluate(p,[obs()],[label()],r)
        score=result["training_seeds"]["7"]["E0"]
        self.assertEqual(score["primary_metric"],"head_nod_f1")
        self.assertEqual(score["primary_f1"],1.)
        self.assertIsNone(score["macro_f1"])
        self.assertIsNone(result["mean_delta_macro_f1"])

    def test_missing_primary_class_makes_full_macro_unavailable(self):
        p,r=evaluation_tests.Evaluation().matrix();p["K_primary"]=["head_nod","smile","listener_vocalization"]
        result=evaluate(p,[obs()],[label()],r)["training_seeds"]["7"]["E0"]
        self.assertIsNone(result["macro_f1"])
        self.assertIsNone(result["smile"]["f1"])
        self.assertEqual(result["supported_primary_classes"],["head_nod","listener_vocalization"])
        self.assertEqual(result["supported_primary_macro_f1"],1)

    def test_visual_audio_and_enabled_flags_reported_separately(self):
        p,r=evaluation_tests.Evaluation().matrix();p["K_train"]=["head_nod","listener_vocalization"]
        result=evaluate(p,[obs()],[label()],r)["training_seeds"]["7"]["E0"]
        self.assertEqual(result["visual_primary_classes"],["head_nod"])
        self.assertEqual(result["visual_macro_f1"],1.)
        self.assertEqual(result["audio_f1"],1.)
        self.assertFalse(result["smile"]["supervision_enabled"])

    def test_primary_cannot_include_disabled_or_duplicate_class(self):
        p,r=evaluation_tests.Evaluation().matrix();p["K_train"]=["head_nod"]
        with self.assertRaises(ValueError):evaluate(p,[obs()],[label()],r)
        p,r=evaluation_tests.Evaluation().matrix();p["K_primary"]=["head_nod","head_nod"]
        with self.assertRaises(ValueError):evaluate(p,[obs()],[label()],r)

    def test_onset_tolerance_is_point_two_and_one_to_one(self):
        l=label();l["events"]=[dict(l["events"][1],start=.1,end=.2),dict(l["events"][1],start=.3,end=.4,event_id="nod2")]
        counts=event_counts(l,[dict(type="head_nod",start=.2,end=.4)],"head_nod")
        self.assertEqual(counts["onset_hits_020"],1)
        counts=event_counts(label(),[dict(type="head_nod",start=.46,end=.7)],"head_nod")
        self.assertEqual(counts["onset_hits_020"],0)
        self.assertEqual(counts["onset_hits_050"],1)


if __name__ == "__main__":
    unittest.main()
