import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from p2_listen.config import load,activate
from p2_listen import VERSION
from p2_listen.contracts import EVENTS, LABEL_SCHEMA, observation,audit_splits,target,digest,sha,atomic_json
from p2_listen.layout import make_layout,geometry,aggregate_audio,pool_by_time
from p2_listen.annotations import validate,rasterize,class_statistics
from p2_listen.cache import save_part,load_part
from p2_listen.events import event_sums,normalized_event_loss
from p2_listen.runtime import masked_mse,support_masks,append_negative,guided_sample,align_audio_latents

torch.set_num_threads(2)


def obs():
    return dict(schema="p2-listen-observation-v1",sample_id="s1",experiment_split="train",official_split="train",
        group_id="g1",grouping_ids=["person:1","session:1"],speaker_audio="a.wav",listener_image="a.png",
        requested_duration_s=1.,geometry=dict(fps=25,height=32,width=32),media_hashes=dict(speaker_audio="x",listener_image="y"))


def label():
    return dict(schema=LABEL_SCHEMA,sample_id="s1",time_origin="clip_start_seconds",duration_s=1.,
        taxonomy=list(EVENTS),coverage={k:[[0.,1.]] for k in EVENTS}, unknown={k:[] for k in EVENTS},
        training_admitted=True,
        events=[dict(event_id="voice1",type="listener_vocalization",start=.125,end=.25,description="Yeah",verification="human",boundary_status="verified",source={"record":"audio1"}),
                dict(event_id="nod1",type="head_nod",start=.25,end=.5,description="Nods",verification="human",boundary_status="verified",source={"record":"video1"})],
        provenance={"human_verified":True,"mapping_version":"listen-candidate-training-v2","admission_policy":"explicit-review-v1",
                    "coverage_sources":{k:{"reference":"test-review", "verification":"human"} for k in EVENTS},
                    "admission_record":{"reference":"test-review"}})


class Contracts(unittest.TestCase):
    def test_observation_rejects_gt_and_plan(self):
        for key in ("targets","text_plan","events","reply"):
            with self.assertRaises(ValueError):observation(dict(obs(),**{key:[]}),verify=False)

    def test_absolute_paths_and_no_target_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=obs();r["speaker_audio"]="speaker.wav";r["listener_image"]="first.png"
            for k in ("speaker_audio","listener_image"):
                (Path(tmp)/r[k]).write_bytes(k.encode());r["media_hashes"][k]=sha(Path(tmp)/r[k])
            actual=observation(r,tmp)
            self.assertEqual(actual["speaker_audio"],str(Path(tmp)/"speaker.wav"))
            (Path(tmp)/"speaker.wav").write_bytes(b"changed")
            with self.assertRaises(ValueError):observation(r,tmp)

    def test_connected_person_cannot_cross_split(self):
        a=obs();b=dict(obs(),sample_id="s2",group_id="different",experiment_split="dev")
        with self.assertRaises(ValueError):audit_splits([a,b])

    def test_unsafe_id_rejected(self):
        with self.assertRaises(ValueError):observation(dict(obs(),sample_id="../bad"),verify=False)

    def test_source_only_geometry(self):
        for duration in (2,6.4,8.04,11.95):
            g=geometry(dict(obs(),requested_duration_s=duration))
            self.assertEqual((g["frames"]-1)%8,0)
            self.assertGreaterEqual(g["duration_s"]+1e-6,duration)
            self.assertLess(g["padding_s"],.321)

    def test_invalid_config_rejected(self):
        c=load();c["generation"]["u_in_cfg"]="positive_only"
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"c.json";atomic_json(p,c)
            with self.assertRaises(ValueError):load(p)


class Labels(unittest.TestCase):
    def test_unknown_and_short_events(self):
        l=label();l["unknown"]["head_nod"]=[[.3,.35]]
        r=rasterize(validate(l,obs()),make_layout(1.,(1,1)))
        self.assertEqual(r["listener_vocalization"]["y"][0,1],1.)
        self.assertFalse(r["head_nod"]["known"][0,1])

    def test_coverage_omission_not_negative(self):
        l=label();l["events"]=[];l["coverage"]["head_nod"]=[]
        r=rasterize(l,make_layout(1.,(1,1)))
        self.assertFalse(r["head_nod"]["known"].any())
        self.assertTrue(r["listener_vocalization"]["known"].all())

    def test_fractional_occupancy(self):
        l=label();l["events"][1].update(start=.3,end=.4)
        r=rasterize(l,make_layout(1.,(1,1)))
        self.assertAlmostEqual(float(r["head_nod"]["y"][0,1]),.4,places=5)

    def test_partial_padding_bin_unknown(self):
        l=label();l["duration_s"]=.9
        l["coverage"]={k:[[0.,.9]] for k in l["coverage"]}
        r=rasterize(l,make_layout(.9,(1,1)))
        self.assertFalse(r["head_nod"]["known"][0,-1])
        self.assertFalse(r["listener_vocalization"]["known"][0,-1])

    def test_censored_event_unknown(self):
        l=label();l["events"][1]["boundary_censored"]=True
        self.assertFalse(rasterize(l,make_layout(1.,(1,1)))["head_nod"]["known"][0,1])

    def test_invalid_time_origin(self):
        l=label();l["time_origin"]="source_seconds"
        with self.assertRaises(ValueError):validate(l,obs())

    def test_audio_aggregation_preserves_time(self):
        x=torch.arange(25).float()[None,:,None]
        intervals=torch.tensor([[[0.,.08],[.08,.16],[.96,1.04]]])
        y,mask=aggregate_audio(x,intervals)
        self.assertAlmostEqual(y[0,0,0].item(),.5,places=5)
        self.assertAlmostEqual(y[0,1,0].item(),2.5,places=5)
        self.assertFalse(mask[0,2])

    def test_pool_masks_spatial_tokens(self):
        x=torch.tensor([[[1.],[100.],[3.],[5.]]])
        y,_=pool_by_time(x,torch.tensor([[0,0,1,1]]),torch.tensor([[True,False,True,True]]),2)
        self.assertEqual(y.tolist(),[[[1.],[4.]]])

    def test_no_positive_labels_disables_classes(self):
        l=label();l["events"]=[]
        stats=class_statistics([rasterize(l,make_layout(1.,(1,1)))])
        self.assertTrue(all(not v["enabled"] and v["disabled_reason"]=="no_positive_support" for v in stats.values()))


class CacheAndLoss(unittest.TestCase):
    def test_independent_cache_parts_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"input";save_part(p,{"audio":"v1"},{"x":torch.ones(2)})
            save_part(Path(tmp)/"events",{"label":"v2"},{"x":torch.zeros(2)})
            self.assertEqual(load_part(p,{"audio":"v1"})["x"].sum(),2)
            with self.assertRaises(ValueError):load_part(p,{"audio":"v2"})
            (p/"payload.pt").write_bytes(b"bad")
            with self.assertRaises(ValueError):load_part(p,{"audio":"v1"})

    def test_all_unknown_differentiable_zero(self):
        logits={k:torch.ones(1,2,requires_grad=True) for k in EVENTS}
        labels={k:dict(y=torch.zeros(1,2),known=torch.zeros(1,2,dtype=torch.bool)) for k in logits}
        sums,counts=event_sums(logits,labels,{k:1 for k in logits})
        z=normalized_event_loss(sums,counts);z.backward()
        self.assertEqual(float(z.detach()),0)
        self.assertTrue(all(v.grad is not None and not v.grad.any() for v in logits.values()))

    def test_masked_audio_tail_has_zero_gradient(self):
        x=torch.ones(1,4,2,requires_grad=True)
        loss=masked_mse(x,torch.zeros_like(x),torch.tensor([[True,True,False,False]]));loss.backward()
        self.assertTrue((x.grad[:,2:]==0).all());self.assertGreater(float(x.grad[:,:2].sum()),0)

    def test_tail_support_masks(self):
        g=geometry(dict(obs(),requested_duration_s=2.))
        m=support_masks(g,(1,128,(g["frames"]-1)//8+1,1,1),(1,8,round(g["duration_s"]/.04),16))
        self.assertFalse(m["video"][0,-1]);self.assertFalse(m["audio"][0,-1])

    def test_audio_centered_endpoint_alignment_is_bounded(self):
        x=torch.arange(11.).reshape(1,1,11,1)
        z,n=align_audio_latents(x,(1,1,10,1))
        self.assertEqual(n,1);self.assertTrue(torch.equal(z,x[:,:,:10]))
        with self.assertRaises(ValueError):align_audio_latents(x,(1,1,9,1))

    def test_cfg_negative_modality_masks(self):
        u=SimpleNamespace(video=torch.ones(1,3,4),audio=torch.ones(1,2,2),video_valid=torch.ones(1,3,dtype=torch.bool),audio_valid=torch.ones(1,2,dtype=torch.bool))
        text=dict(video=torch.zeros(1,5,4),audio=torch.zeros(1,5,2),mask=torch.ones(1,5,dtype=torch.bool))
        neg=append_negative(text,u)
        self.assertEqual(neg["video_mask"].shape,(1,8));self.assertEqual(neg["audio_mask"].shape,(1,7))
        self.assertTrue(torch.equal(neg["video"][:,-3:],u.video))
        self.assertEqual(append_negative(text,None)["video"].shape,(1,5,4))

    def test_rng_capture_does_not_initialize_cuda(self):
        from p2_listen.training import rng_state
        with patch("torch.cuda.is_initialized",return_value=False),patch("torch.cuda.get_rng_state",side_effect=AssertionError):
            self.assertIsNone(rng_state()["cuda"])


class GradientAndResume(unittest.TestCase):
    def test_real_p2_small_model_gradient_and_resume(self):
        from p2_listen.audit import cpu_audit
        evidence=cpu_audit(load())
        self.assertEqual(evidence["probe_only"]["event_to_u_norms"],[0.,0.])
        self.assertTrue(evidence["resume_exact"])

    def test_probe_weight_cannot_change_backbone_update(self):
        from p2_listen.audit import fixture
        from p2_listen.model import ListeningExtractor
        from p2_listen.training import optimization_step
        cfg,sample=fixture(load());torch.manual_seed(44)
        initial=ListeningExtractor(cfg,8)
        t=dict(connector_lr=.01,feature_lr=.01,event_lr=.01,clip_norm=.001)
        models=[]
        for weight in (0.,10000.):
            m=copy.deepcopy(initial);o=torch.optim.AdamW(m.optimizer_groups(t),weight_decay=0)
            optimization_step(m,o,[sample],lambda u,_:(u.video-.2).square().mean()+u.audio.square().mean(),
                mode="probe_only",pos_weights={k:1. for k in EVENTS},event_weight=weight,
                seed=2,ordinals=[0],config=t)
            models.append(m)
        self.assertTrue(all(torch.equal(v,models[1].extractor.state_dict()[k]) for k,v in models[0].extractor.state_dict().items()))

    def test_resume_rejects_changed_identity_before_mutation(self):
        from p2_listen.training import restore_checkpoint
        m=torch.nn.Linear(2,2);before=copy.deepcopy(m.state_dict())
        with self.assertRaises(ValueError):restore_checkpoint({"version":VERSION,"taxonomy":list(EVENTS),"label_schema":LABEL_SCHEMA,"identity":{"arm":"E0"}},m,None,{"arm":"E1"})
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))

    def test_sample_schedule_global_batch(self):
        from p2_listen.training import indices
        single=indices(7,3,8,9)
        distributed=indices(7,3,8,9,0,2)+indices(7,3,8,9,1,2)
        self.assertEqual(single,sorted(distributed,key=lambda x:x[1]))


if __name__=="__main__":unittest.main()
