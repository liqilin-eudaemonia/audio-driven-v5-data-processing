import importlib.util
from types import SimpleNamespace
import unittest

import torch

from p2_listen.contracts import ROOT
from p2_listen.events import EventHeads
from p2_listen.layout import make_layout

spec = importlib.util.spec_from_file_location("p2_replay_precision", ROOT / "scripts/check_p2_listen_gpu_replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class EventPrecision(unittest.TestCase):
    def test_bf16_autocast_preserves_fp32_head_values_and_joint_gradient(self):
        torch.manual_seed(17)
        heads = EventHeads(12,7,8)
        layout = make_layout(1.,(4,4))
        u = SimpleNamespace(video=torch.randn(1,64,12,dtype=torch.bfloat16,requires_grad=True),
                            audio=torch.randn(1,8,7,dtype=torch.bfloat16,requires_grad=True),
                            video_valid=layout["video_valid"],audio_valid=layout["audio_valid"])
        reference = heads(u,layout,"joint_event")
        with torch.autocast("cpu",dtype=torch.bfloat16):
            actual = heads(u,layout,"joint_event")
        for k in actual:
            self.assertEqual(actual[k].dtype,torch.float32)
            self.assertTrue(torch.equal(actual[k],reference[k]))
        sum(v.sum() for v in actual.values()).backward()
        self.assertGreater(float(u.video.grad.float().abs().sum()),0.)
        self.assertGreater(float(u.audio.grad.float().abs().sum()),0.)
        for k,v in heads(u,layout,"probe_only").items():
            self.assertEqual(torch.autograd.grad(v.sum(),[u.video,u.audio],allow_unused=True,retain_graph=True),(None,None))


class ForwardComparison(unittest.TestCase):
    def record(self,value=.31):
        return dict(base=7.,event=value,event_known_bins={"head_nod":8})

    def test_small_roundoff_is_not_required_to_be_bitwise_equal(self):
        r=replay.compare_forward_records(self.record(),self.record(.3100001),self.record())
        self.assertTrue(r["passed"])
        self.assertFalse(r["exact"])
        self.assertTrue(r["counts_exact"])

    def test_original_bf16_drift_still_exceeds_new_numeric_threshold(self):
        r=replay.compare_forward_records(self.record(),self.record(.3100117),self.record())
        self.assertFalse(r["passed"])

    def test_supervision_count_mismatch_is_never_tolerated(self):
        r=self.record();r["event_known_bins"]["head_nod"]+=1
        self.assertFalse(replay.compare_forward_records(r,self.record(),self.record())["passed"])

    def test_nonfinite_or_large_loss_drift_fails(self):
        for v in (float("nan"),float("inf"),.32):
            self.assertFalse(replay.compare_forward_records(self.record(v),self.record(),self.record())["passed"])

    def test_invalid_tolerances_fail(self):
        for v in (-1.,float("inf"),float("nan")):
            with self.assertRaises(ValueError):
                replay.compare_forward_records(self.record(),self.record(),self.record(),atol=v)
