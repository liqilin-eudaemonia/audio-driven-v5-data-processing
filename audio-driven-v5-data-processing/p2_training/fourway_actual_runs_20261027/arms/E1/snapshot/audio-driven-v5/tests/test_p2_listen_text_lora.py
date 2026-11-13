"""Native-attention CPU checks for the paired, decoder-free text LoRA experiment."""
import copy
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from p2_listen.audit import fixture, TinyTextRenderer, audit_model, audit_callback
from p2_listen.config import load, text_lora_config, group_learning_rate, source_identity
from p2_listen.contracts import ROOT, EVENTS, atomic_json
from p2_listen.events import normalized_event_loss
from p2_listen.model import ListeningExtractor, build_model
from p2_listen.generation import load_generation_model
from p2_listen.training import (optimization_step, save_checkpoint, checked_checkpoint,
                               restore_checkpoint, rng_state, clip_groups)
from p2_listen import features
from test_p2_listen_core import obs, label


def config():
    return load(ROOT/'configs/p2_listen_av_event_text_lora_v1.json')


class Configuration(unittest.TestCase):
    def test_legacy_disabled_and_new_config_explicit(self):
        self.assertIsNone(text_lora_config(load()))
        c=config()
        self.assertFalse(c['training']['feature_loss_enabled'])
        self.assertEqual(text_lora_config(c),dict(rank=8,alpha=8.,learning_rate=1e-5))
        self.assertEqual(group_learning_rate(c['training'],'text_lora'),1e-5)

    def test_invalid_lora_settings_rejected_before_model_load(self):
        valid=config()['training']['text_lora']
        cases=[False,{'enabled':1},{'enabled':False,'rank':8},dict(valid,rank=True),
               dict(valid,rank=0),dict(valid,alpha=float('nan')),dict(valid,learning_rate=-1),
               dict(valid,unknown=2),{'enabled':True}]
        for value in cases:
            with self.subTest(value=value),tempfile.TemporaryDirectory() as directory:
                c=config();c['training']['text_lora']=value
                # Validation helper runs before allocation and also from config.load.
                with self.assertRaises(ValueError):text_lora_config(c)

    def test_caches_unchanged_and_lora_source_in_checkpoint_identity(self):
        c=config();off=copy.deepcopy(c);del off['training']['text_lora']
        with patch.object(features,'weights_identity',return_value={'assets':'fixed'}):
            for part in ('input','teacher','vae','text','events'):
                self.assertEqual(features.identity(c,part,obs(),{'target':'fixed'},label()),
                                 features.identity(off,part,obs(),{'target':'fixed'},label()))
        files=source_identity(c)
        self.assertTrue(any(p.endswith('/omni_av_p21/lora.py') for p in files))


class NativeAttentionLoRA(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(67)
        self.c=config();self.c['events']['head_width']=8
        self.cfg,s=fixture(self.c)
        self.sample=(*s[:2],None,*s[3:])
        self.t=dict(self.c['training'],connector_lr=.001,event_lr=.001)
        self.weights={k:1. for k in EVENTS}

    def model(self):
        return audit_model(self.c,self.cfg)

    def update(self,m,o,mode='joint_event',step=0,audit=True):
        return optimization_step(m,o,[self.sample],audit_callback(m),mode=mode,
            pos_weights=self.weights,event_weight=.05,seed=79,ordinals=[step],config=self.t,audit=audit)

    def test_factory_requires_renderer_for_enabled_lora(self):
        with patch('p2_listen.config.extractor_config',return_value=self.cfg):
            with self.assertRaisesRegex(ValueError,'requires the renderer'):build_model(self.c)
            m=build_model(self.c,TinyTextRenderer()).to('cpu')
        m.assert_parameter_ownership()
        self.assertEqual(len(m.text_lora_targets),16)

    def test_only_text_attention_adapters_train_and_all_are_owned_once(self):
        m=self.model();groups=m.optimizer_groups(self.t)
        self.assertEqual([g['name'] for g in groups],['connector','events','text_lora'])
        params=[p for g in groups for p in g['params']]
        self.assertEqual(len(params),len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params},{id(p) for p in m.parameters()})
        m.assert_parameter_ownership()
        for name,p in m._renderer.named_parameters():
            self.assertEqual(p.requires_grad,'.update.' in name)
        self.assertFalse(any('decoder' in k or '.base.' in k or '_renderer' in k for k in m.state_dict()))
        self.assertTrue(all(p.dtype==torch.float32 for p in m.text_lora.parameters()))
        copied=copy.deepcopy(m);copied.assert_parameter_ownership()

    def test_zero_initialization_preserves_native_attention_outputs(self):
        renderer=TinyTextRenderer();baseline=copy.deepcopy(renderer)
        m=ListeningExtractor(self.cfg,8,feature_loss_enabled=False,
                             text_lora=text_lora_config(self.c),renderer=renderer)
        u=m(self.sample[0],self.sample[1])['u']
        for want,got in zip(baseline(u),renderer(u)):self.assertTrue(torch.equal(want,got))

    def test_injector_refuses_reinjection_and_validates_all_targets_before_mutation(self):
        from omni_av_p21.lora import inject_text_lora,AdaptedLinear
        from omni_av_p21.config import LoRAConfig
        m=self.model()
        with self.assertRaisesRegex(ValueError,'already installed'):inject_text_lora(m._renderer,LoRAConfig())
        bad=TinyTextRenderer();del bad.transformer_blocks[-1].audio_attn2
        with self.assertRaisesRegex(ValueError,'Missing native text'):inject_text_lora(bad,LoRAConfig())
        self.assertFalse(any(isinstance(x,AdaptedLinear) for x in bad.modules()))

    def test_optimizer_refuses_inconsistent_lora_settings(self):
        m=self.model();t=copy.deepcopy(self.t);t['text_lora']['rank']=4
        with self.assertRaisesRegex(ValueError,'settings differ'):m.optimizer_groups(t)

    def test_e0_e1_av_updates_lora_and_base_stays_frozen(self):
        initial=self.model()
        for mode in ('probe_only','joint_event'):
            with self.subTest(mode=mode):
                m=copy.deepcopy(initial)
                frozen={n:p.detach().clone() for n,p in m._renderer.named_parameters() if not p.requires_grad}
                before=copy.deepcopy(m.state_dict());opt=torch.optim.AdamW(m.optimizer_groups(self.t),weight_decay=0)
                for step in range(3):r=self.update(m,opt,mode,step)
                self.assertGreater(r['paths_first_micro']['text_lora']['av'],0)
                self.assertEqual(r['paths_first_micro']['text_lora']['event'],0)
                event=r['paths_first_micro']['event']
                self.assertEqual(any(event),mode=='joint_event')
                self.assertEqual(r['base'],r['av']);self.assertIsNone(r['feature_losses'])
                for prefix in ('extractor.connector.','event_heads.','text_lora.'):
                    self.assertTrue(any(not torch.equal(before[k],v) for k,v in m.state_dict().items() if k.startswith(prefix)))
                for n,p in m._renderer.named_parameters():
                    if n in frozen:
                        self.assertTrue(torch.equal(p,frozen[n]));self.assertIsNone(p.grad)

    def test_event_only_backward_never_updates_lora(self):
        m=self.model();s=self.sample
        r=m(s[0],s[1],None,s[4],'joint_event',self.weights)
        normalized_event_loss(r['event_sums'],r['event_counts']).backward()
        self.assertTrue(all(p.grad is None for p in m.text_lora.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.extractor.connector.parameters()))

    def test_probe_loss_cannot_change_connector_or_lora_update(self):
        a=self.model();b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.optimizer_groups(self.t),weight_decay=0)
        ob=torch.optim.AdamW(b.optimizer_groups(self.t),weight_decay=0)
        self.update(a,oa,'probe_only')
        optimization_step(b,ob,[self.sample],audit_callback(b),mode='probe_only',pos_weights=self.weights,
                          event_weight=0.,seed=79,ordinals=[0],config=self.t)
        for k,v in a.state_dict().items():
            if not k.startswith('event_heads.'):self.assertTrue(torch.equal(v,b.state_dict()[k]),k)

    def test_lora_is_clipped_with_connector_and_not_with_event_head(self):
        m=self.model()
        for p in m.parameters():p.grad=torch.ones_like(p)
        clip_groups(m,.25)
        norm=lambda ps:sum(float(p.grad.double().square().sum()) for p in ps)**.5
        self.assertLessEqual(norm(m.main_parameters()),.250001)
        self.assertLessEqual(norm(m.event_heads.parameters()),.250001)

    def test_all_unknown_still_updates_lora_from_av(self):
        self.sample=copy.deepcopy(self.sample)
        for v in self.sample[4].values():v['known'].fill_(False)
        m=self.model();before=copy.deepcopy(m.text_lora.state_dict())
        opt=torch.optim.AdamW(m.optimizer_groups(self.t),weight_decay=0)
        r=self.update(m,opt)
        self.assertEqual(r['event'],0)
        self.assertGreater(r['paths_first_micro']['text_lora']['av'],0)
        self.assertTrue(any(not torch.equal(v,m.text_lora.state_dict()[k]) for k,v in before.items()))

    def test_resume_exact_next_update_including_optimizer_and_renderer_outputs(self):
        m=self.model();initial=copy.deepcopy(m)
        opt=torch.optim.AdamW(m.optimizer_groups(self.t),weight_decay=0)
        self.update(m,opt);identity={'config':self.c,'text_lora_targets':m.text_lora_targets}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'step.pt';save_checkpoint(p,m,opt,1,identity,[rng_state()]);saved=checked_checkpoint(p)
            restored=initial;other=torch.optim.AdamW(restored.optimizer_groups(self.t),weight_decay=0)
            restore_checkpoint(saved,restored,other,identity)
            self.update(m,opt,step=1);self.update(restored,other,step=1)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items()))
            for i,state in opt.state_dict()['state'].items():
                for k,v in state.items():self.assertTrue(torch.equal(v,other.state_dict()['state'][i][k]))
            off=ListeningExtractor(self.cfg,8,feature_loss_enabled=False)
            with self.assertRaisesRegex(ValueError,'Invalid checkpoint tensors'):
                restore_checkpoint(saved,off,torch.optim.AdamW(off.optimizer_groups(self.t)),identity)

    def test_generation_load_preserves_trained_adapter_and_rejects_missing_weights(self):
        m=self.model();clean=copy.deepcopy(m._renderer)
        # Capture original base values in a fresh renderer (no copied adapters).
        fresh=TinyTextRenderer()
        state={k.replace('.base.','.'):v for k,v in clean.state_dict().items() if '.update.' not in k}
        fresh.load_state_dict(state)
        opt=torch.optim.AdamW(m.optimizer_groups(self.t),weight_decay=0);self.update(m,opt)
        saved={'identity':{'text_lora_targets':m.text_lora_targets},'model':copy.deepcopy(m.state_dict())}
        with patch('p2_listen.config.extractor_config',return_value=self.cfg):
            deployed=load_generation_model(self.c,saved,fresh,'cpu')
        u=deployed(self.sample[0],self.sample[1])['u']
        for x,y in zip(m._renderer(u),fresh(u)):self.assertTrue(torch.equal(x,y))
        self.assertTrue(all(not p.requires_grad for p in fresh.parameters()))
        self.assertTrue(all(not p.requires_grad for p in deployed.parameters()))
        missing=copy.deepcopy(saved);del missing['model'][next(k for k in missing['model'] if k.startswith('text_lora.'))]
        with patch('p2_listen.config.extractor_config',return_value=self.cfg):
            with self.assertRaises(RuntimeError):load_generation_model(self.c,missing,TinyTextRenderer(),'cpu')
        wrong=copy.deepcopy(saved);wrong['identity']['text_lora_targets']=[]
        with patch('p2_listen.config.extractor_config',return_value=self.cfg):
            with self.assertRaisesRegex(ValueError,'target layers differ'):load_generation_model(self.c,wrong,TinyTextRenderer(),'cpu')

    def test_cpu_bfloat16_autocast_preserves_fp32_adapter_master_weights(self):
        m=self.model()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            r=m(self.sample[0],self.sample[1],av_callback=audit_callback(m))
        r['av'].backward()
        self.assertTrue(all(p.dtype==torch.float32 for p in m.text_lora.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.text_lora.parameters()))

    def test_generation_modes_keep_lora_for_remove_u_and_native_text_only(self):
        """Exercise the real orchestration; replace large models, VAE and media I/O."""
        from p2_listen.generation import generate
        from p2_listen.contracts import sha
        from omni_av_p21.lora import AdaptedLinear
        m=self.model();base_state={k.replace('.base.','.'):v.clone() for k,v in m._renderer.state_dict().items()
                                  if '.update.' not in k}
        opt=torch.optim.AdamW(m.optimizer_groups(self.t),weight_decay=0);self.update(m,opt)
        identity=dict(config=self.c,source=source_identity(self.c),assets={},arm='E1',seed=1,step=1,
                      s0=True,text_lora_targets=m.text_lora_targets)
        saved=dict(identity=identity,model=copy.deepcopy(m.state_dict()),step=1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);o=obs()
            for key in ('speaker_audio','listener_image'):
                p=root/key;p.write_bytes(b'fixture');o[key]=str(p);o['media_hashes'][key]=sha(p)
            checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'checked by mocked reader')
            for mode in ('normal','remove_u','text_only'):
                with self.subTest(mode=mode),ExitStack() as stack:
                    renderer=TinyTextRenderer();renderer.load_state_dict(base_state)
                    prepared=SimpleNamespace()
                    seen={}
                    def sampler(used,inputs,negative,*args):
                        seen['adapted']=any(isinstance(x,AdaptedLinear) for x in used.modules())
                        if mode!='text_only':
                            for j,adapter in enumerate(m.text_lora):
                                target=used.get_submodule(m.text_lora_targets[j]).update
                                self.assertTrue(torch.equal(target.b.weight,adapter.b.weight))
                        return {}
                    def decode(*args,**kwargs):
                        torch.save(dict(wave=torch.zeros(2,16000),rate=16000),kwargs['raw_audio_path'])
                        return {}
                    mocks={
                        'p2_listen.config.extractor_config':dict(return_value=self.cfg),
                        'p2_listen.training.checked_checkpoint':dict(return_value=saved),
                        'p2_listen.generation.weights_identity':dict(return_value=None),
                        'p2_listen.generation.load_input':dict(return_value=self.sample[:2]),
                        'p2_listen.generation.load_text':dict(return_value={'positive':{},'negative':{}}),
                        'p2_listen.generation.unload':dict(),
                        'p2_listen.generation.guided_sample':dict(side_effect=sampler),
                        'p2_listen.generation.append_negative':dict(side_effect=lambda text,u:seen.update(has_u=u is not None) or {}),
                        'p2_listen.generation.subprocess.run':dict(),
                        'omni_av_p2.inference.reference_latents':dict(return_value={}),
                        'omni_av_p2.ltx.load_renderer':dict(return_value=(renderer,SimpleNamespace(prepare_training_inputs=lambda *a:prepared),None)),
                        'omni_av_p2.ltx.append_conditions':dict(side_effect=lambda p,u:p),
                        'omni_av_p2.ltx.unpack_latents':dict(return_value=(None,None)),
                        'omni_av.ltx_runtime.with_text':dict(return_value={}),
                        'omni_av_p3_staged.media.decode_and_save':dict(side_effect=decode),
                        'soundfile.write':dict(),
                        'torch.autocast':dict(side_effect=lambda *a,**kw:nullcontext()),
                    }
                    # Pretrained-asset checks still run and must agree.
                    saved['identity']['assets']={k:None for k in ('omni','ltx','gemma')}
                    for name,kwargs in mocks.items():stack.enter_context(patch(name,**kwargs))
                    result=generate(self.c,o,root/mode,checkpoint=checkpoint if mode!='text_only' else None,
                                    mode=mode,device='cpu')
                    self.assertEqual(result['status'],'success')
                    self.assertEqual(seen['adapted'],mode!='text_only')
                    self.assertEqual(seen['has_u'],mode=='normal')
                    self.assertEqual(result['text_lora_enabled'],mode!='text_only')


if __name__=='__main__':unittest.main()
