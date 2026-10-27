"""AV/Event ablation: no teacher I/O, no feature modules, and matched gradients/resume."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from p2_listen.audit import fixture
from p2_listen.config import load,feature_loss_enabled,training_cache_parts,extractor_config
from p2_listen.contracts import ROOT,EVENTS,atomic_json
from p2_listen.model import ListeningExtractor,build_model
from p2_listen.training import optimization_step,save_checkpoint,checked_checkpoint,restore_checkpoint,rng_state
from p2_listen.training_data import load_training_sample
from p2_listen import features
from test_p2_listen_core import obs,label


def config():
    return load(ROOT/"configs/p2_listen_av_event_v1.json")


def training():
    return dict(connector_lr=.001,event_lr=.001,clip_norm=1.)


class Configuration(unittest.TestCase):
    def roundtrip(self,c):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"config.json";atomic_json(path,c)
            return load(path)

    def test_legacy_defaults_and_explicit_full_mode(self):
        c=load();self.assertTrue(feature_loss_enabled(c))
        c["training"]["feature_loss_enabled"]=True
        self.assertTrue(feature_loss_enabled(self.roundtrip(c)))
        self.assertIn("teacher",training_cache_parts(c))

    def test_av_only_accepts_no_feature_learning_rate_and_excludes_teacher(self):
        c=config();self.assertNotIn("feature_lr",c["training"])
        self.assertFalse(feature_loss_enabled(self.roundtrip(c)))
        self.assertEqual(training_cache_parts(c),["input","text","vae","events"])

    def test_non_boolean_switch_rejected(self):
        for value in (0,1,"false",None):
            with self.subTest(value=value):
                c=config();c["training"]["feature_loss_enabled"]=value
                with self.assertRaisesRegex(ValueError,"boolean"):self.roundtrip(c)

    def test_switch_does_not_change_existing_extractor_cache_contract(self):
        full=load();off=copy.deepcopy(full);off["training"]["feature_loss_enabled"]=False
        self.assertEqual(extractor_config(full).to_dict(),extractor_config(off).to_dict())
        with patch.object(features,"weights_identity",return_value={"assets":"fixed"}):
            for part in ("input","teacher","vae","text","events"):
                self.assertEqual(features.identity(full,part,obs(),{"target":"fixed"},label()),
                                 features.identity(off,part,obs(),{"target":"fixed"},label()))


class ArchitectureAndUpdates(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(151)
        self.cfg,self.full_sample=fixture(config())
        self.sample=(*self.full_sample[:2],None,*self.full_sample[3:])
        self.weights={k:1. for k in EVENTS}

    def model(self):
        return ListeningExtractor(self.cfg,8,feature_loss_enabled=False)

    def update(self,m,opt,mode="joint_event",step=0):
        return optimization_step(m,opt,[self.sample],
            lambda u,_:(u.video-.2).square().mean()+(u.audio+.1).square().mean(),
            mode=mode,pos_weights=self.weights,event_weight=.05,seed=77,ordinals=[step],
            config=training(),audit=True)

    def test_decoders_are_never_constructed_and_optimizer_owns_only_live_parameters(self):
        with patch("omni_av_p2.model.EmbeddingDecoder",side_effect=AssertionError("decoder constructed")):
            m=self.model()
        self.assertFalse(hasattr(m.extractor,"video_decoder"))
        self.assertFalse(hasattr(m.extractor,"audio_decoder"))
        self.assertFalse(any("decoder" in k for k in m.state_dict()))
        groups=m.optimizer_groups(training())
        self.assertEqual([g["name"] for g in groups],["connector","events"])
        params=[p for g in groups for p in g["params"]]
        self.assertEqual(len(params),len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params},{id(p) for p in m.parameters()})

    def test_accidental_teacher_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"omit teacher"):
            self.model()(self.sample[0],self.sample[1],self.full_sample[2])

    def test_shared_factory_loads_decoder_free_checkpoint_for_inference(self):
        c=config()
        with patch("p2_listen.config.extractor_config",return_value=self.cfg):
            source=build_model(c);deploy=build_model(c).eval()
        deploy.load_state_dict(source.state_dict(),strict=True)
        with torch.no_grad():result=deploy(self.sample[0],self.sample[1])
        self.assertFalse(deploy.feature_loss_enabled)
        self.assertEqual(result["features"],{})
        self.assertEqual(set(result["logits"]),set(EVENTS))

    def test_e0_updates_av_connector_and_probe_without_event_gradient_to_u(self):
        m=self.model();before=copy.deepcopy(m.state_dict())
        opt=torch.optim.AdamW(m.optimizer_groups(training()),weight_decay=0)
        r=self.update(m,opt,"probe_only")
        self.assertEqual(r["base"],r["av"])
        self.assertIsNone(r["feature_losses"])
        self.assertIsNone(r["paths_first_micro"]["feature"])
        self.assertEqual(r["paths_first_micro"]["event"],[0.,0.])
        self.assertTrue(all(x>0 for x in r["paths_first_micro"]["av"]))
        for prefix in ("extractor.connector.","event_heads."):
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in m.state_dict().items() if k.startswith(prefix)))

    def test_e1_event_gradient_reaches_both_u_modalities_without_features(self):
        m=self.model();opt=torch.optim.AdamW(m.optimizer_groups(training()),weight_decay=0)
        r=self.update(m,opt)
        self.assertEqual(r["base"],r["av"])
        self.assertFalse(r["feature_loss_enabled"])
        self.assertTrue(all(x>0 for x in r["paths_first_micro"]["event"]))
        self.assertTrue(all(x>0 for x in r["paths_first_micro"]["av"]))

    def test_all_unknown_event_batch_still_trains_av(self):
        self.sample=copy.deepcopy(self.sample)
        for value in self.sample[4].values():value["known"].fill_(False)
        m=self.model();opt=torch.optim.AdamW(m.optimizer_groups(training()),weight_decay=0)
        r=self.update(m,opt)
        self.assertEqual(r["event"],0.)
        self.assertGreater(r["grad_norm"]["main"],0.)
        self.assertEqual(r["paths_first_micro"]["event"],[0.,0.])

    def test_resume_next_update_exact_and_full_architecture_refused(self):
        m=self.model();opt=torch.optim.AdamW(m.optimizer_groups(training()),weight_decay=0)
        self.update(m,opt)
        identity={"feature_loss_enabled":False}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"step.pt";save_checkpoint(path,m,opt,1,identity,[rng_state()])
            saved=checked_checkpoint(path)
            resumed=self.model();opt2=torch.optim.AdamW(resumed.optimizer_groups(training()),weight_decay=0)
            self.assertEqual(restore_checkpoint(saved,resumed,opt2,identity),1)
            self.update(m,opt,step=1);self.update(resumed,opt2,step=1)
            self.assertTrue(all(torch.equal(v,resumed.state_dict()[k]) for k,v in m.state_dict().items()))
            full=ListeningExtractor(self.cfg,8)
            full_opt=torch.optim.AdamW(full.optimizer_groups(dict(training(),feature_lr=.001)))
            with self.assertRaisesRegex(ValueError,"identity mismatch"):
                restore_checkpoint(saved,full,full_opt,{"feature_loss_enabled":True})
            with self.assertRaisesRegex(ValueError,"Invalid checkpoint tensors"):
                restore_checkpoint(saved,full,full_opt,identity)


class CacheAccess(unittest.TestCase):
    def test_no_teacher_path_access_and_missing_vae_still_fails(self):
        c=config();o=obs();l=label();l["provenance"]["admission_policy"]=c["events"]["admission_policy"]
        visited=[]
        def payload(folder,identity,device="cpu"):
            part=Path(folder).parent.name
            self.assertNotEqual(part,"teacher")
            visited.append(part);return {"part":part}
        with patch.object(features,"load_input",return_value=("condition","layout")),\
             patch.object(features,"identity",return_value={}),\
             patch("p2_listen.training_data.read",return_value=l),\
             patch.object(features,"load_training_sample",side_effect=AssertionError("legacy loader accessed")),\
             patch.object(features,"load_part",side_effect=payload):
            sample=load_training_sample(c,o,{})
            self.assertIsNone(sample[2]);self.assertEqual(visited,["vae","events"])
            with patch.object(features,"load_part",side_effect=FileNotFoundError("missing VAE")):
                with self.assertRaises(FileNotFoundError):load_training_sample(c,o,{})

    def test_full_mode_still_requires_teacher_loader(self):
        with patch.object(features,"load_training_sample",side_effect=FileNotFoundError("missing teacher")):
            with self.assertRaisesRegex(FileNotFoundError,"teacher"):
                load_training_sample(load(),obs(),{})

    def test_cache_all_cli_excludes_teacher_only_for_av_mode(self):
        spec=importlib.util.spec_from_file_location("p2_cli_av_test",ROOT/"scripts/p2_listen.py")
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        for c in (load(),config()):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"config.json";atomic_json(path,c)
                with patch("sys.argv",["p2_listen.py","cache","--config",str(path),"--part","all"]),\
                     patch("p2_listen.dataset.load_observations",return_value=[obs()]),\
                     patch("p2_listen.dataset.load_targets",return_value={}),\
                     patch.object(features,"cache_parts") as cache:
                    cli.main()
                self.assertEqual(cache.call_args.args[3],training_cache_parts(c))


if __name__=="__main__":unittest.main()
