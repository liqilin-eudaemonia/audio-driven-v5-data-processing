import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from p2_listen.config import tensorboard_config
from p2_listen.contracts import EVENTS
from p2_listen.monitoring import (TrainingMonitor, open_monitor, global_scalars,
                                  rank_zero_action, truncate_jsonl)


def record_fixture(base=2., av=2.):
    return dict(base=base,av=av,av_loss_by_modality=dict(video=av*.75,audio=av*.25),feature_losses=None,
        event=.8,event_loss_per_class={k:(.8 if k=='head_nod' else None) for k in EVENTS},
        event_loss_by_modality=dict(video=.8,audio=0.),event_known_bins={k:(10 if k=='head_nod' else 0) for k in EVENTS},
        grad_norm=dict(main=2.,events=.1))


def metrics(record=None, **kwargs):
    options=dict(local_samples=2,media_seconds=8.,
        timings=dict(step_seconds=4.,data_seconds=1.,compute_seconds=2.,checkpoint_seconds=1.,
                     allocated_gib=1.,reserved_gib=2.,peak_allocated_gib=3.,peak_reserved_gib=4.),
        optimizer=SimpleNamespace(param_groups=[dict(name='connector',lr=3e-5),dict(name='text_lora',lr=1e-5)]),
        event_weight=.05,arm='E0',step=3,dataset_size=50,device='cpu',world=1)
    options.update(kwargs)
    return global_scalars(record or record_fixture(),**options)


class MonitoringTests(unittest.TestCase):
    def test_config_strict_and_legacy_disabled(self):
        self.assertFalse(tensorboard_config({'training':{}})['enabled'])
        for value in [None,[],{'enabled':1},{'log_every':0},{'flush_secs':True},{'typo':1}]:
            with self.assertRaises(ValueError):tensorboard_config({'training':{'tensorboard':value}})

    def test_metric_semantics_and_missing_labels(self):
        scalars=metrics()
        self.assertEqual(scalars['loss/av_video']+scalars['loss/av_audio'],scalars['loss/av'])
        self.assertEqual(scalars['loss/generator_objective'],2.)
        self.assertAlmostEqual(scalars['loss/optimization_objective'],2.04)
        self.assertAlmostEqual(metrics(arm='E1')['loss/generator_objective'],2.04)
        self.assertNotIn('event_loss/head_shake',scalars)
        self.assertEqual(scalars['supervision/known_bins/head_shake'],0)
        self.assertEqual(scalars['throughput/samples_per_second'],.5)
        self.assertEqual(scalars['throughput/media_seconds_per_second'],2.)
        self.assertEqual(scalars['progress/samples_seen'],6)
        self.assertNotIn('loss/feature_video',scalars)
        with self.assertRaises(ValueError):metrics(world=2)

    def test_tensorboard_actual_events_resume_and_rng(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        c={'training':{'tensorboard':{'enabled':True,'flush_secs':1}}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            rng=torch.get_rng_state().clone()
            logger=TrainingMonitor(path,c)
            for step in [1,2,3]:logger.write(step,{'loss/av':float(step)},flush=True)
            logger.close()
            (path/'train_rank0.jsonl').write_text(''.join(json.dumps({'step':s})+'\n' for s in [1,2,3])+'{"step":')
            logger=open_monitor(path,c,start=2,resume=True,rank=0,world=1)
            self.assertEqual([json.loads(s)['step'] for s in (path/'metrics.jsonl').read_text().splitlines()],[1,2])
            self.assertEqual([json.loads(s)['step'] for s in (path/'train_rank0.jsonl').read_text().splitlines()],[1,2])
            logger.write(3,{'loss/av':30.},flush=True);logger.close()
            ea=EventAccumulator(str(path/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
            events=ea.Scalars('loss/av')
            self.assertEqual([x.step for x in events],[1,2,3])
            self.assertEqual([x.value for x in events],[1.,2.,30.])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_only_rank_zero_writes_and_errors_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=open_monitor(tmp,{'training':{}},start=0,resume=False,rank=0,world=1)
            self.assertIsNone(result);self.assertFalse((Path(tmp)/'tensorboard').exists())
            self.assertIsNone(rank_zero_action(lambda:1/0,rank=1,world=1))
            with self.assertRaisesRegex(RuntimeError,'Training monitor failed'):
                rank_zero_action(lambda:1/0,rank=0,world=1)

    def test_corrupt_committed_logs_are_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'log.jsonl';original='{"step":1}\nbroken\n{"step":2}\n';path.write_text(original)
            with self.assertRaises(ValueError):truncate_jsonl(path,2)
            self.assertEqual(path.read_text(),original)
            path.write_text('{"step":1}\n{"step":1}\n')
            with self.assertRaises(ValueError):truncate_jsonl(path,2)

    def test_av_components_do_not_change_objective_or_gradient(self):
        from p2_listen.runtime import av_loss
        p=torch.nn.Parameter(torch.tensor([[[1.,2.],[3.,4.]]]))
        prepared=SimpleNamespace(video=None,audio=None,video_targets=torch.zeros_like(p),audio_targets=torch.ones_like(p),
            video_loss_mask=torch.tensor([[True,False]]),audio_loss_mask=torch.tensor([[True,True]]))
        support={'video':torch.tensor([[True,True]]),'audio':torch.tensor([[False,True]])}
        renderer=lambda **kw:(p,p*2)
        loss=av_loss(renderer,prepared,support);g=torch.autograd.grad(loss,p)[0]
        total,parts=av_loss(renderer,prepared,support,return_components=True)
        self.assertTrue(torch.equal(loss,total));self.assertTrue(torch.equal(g,torch.autograd.grad(total,p)[0]))
        self.assertTrue(torch.equal(parts['video']+parts['audio'],total))
        self.assertFalse(any(x.requires_grad for x in parts.values()))


if __name__=='__main__':unittest.main()
