import copy
from dataclasses import dataclass
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from p2_listen.contracts import atomic_json,sha,read
from p2_listen.config import load
from p2_listen.dataset import prepare,import_labels,load_observations
from p2_listen.runtime import guided_sample,append_negative
from test_p2_listen_core import obs,label


@dataclass
class Modality:
    latent: torch.Tensor
    sigma: torch.Tensor
    timesteps: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor


class FakeRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=[]
    def forward(self,video,audio,perturbations):
        self.calls.append((video.context.clone(),audio.context.clone()))
        return torch.ones_like(video.latent)*video.context.mean(),torch.ones_like(audio.latent)*audio.context.mean()


class Sampler(unittest.TestCase):
    def test_same_u_in_both_branches_and_first_frame_preserved(self):
        u=SimpleNamespace(video=torch.ones(1,3,4),audio=torch.ones(1,2,2),video_valid=torch.ones(1,3,dtype=torch.bool),audio_valid=torch.ones(1,2,dtype=torch.bool))
        text=dict(video=torch.zeros(1,1,4),audio=torch.zeros(1,1,2),mask=torch.ones(1,1,dtype=torch.bool))
        negative=append_negative(text,u)
        video=Modality(torch.full((1,3,4),.5),torch.ones(1),torch.ones(1,3),negative["video"],negative["video_mask"])
        audio=Modality(torch.zeros(1,3,2),torch.ones(1),torch.ones(1,3),negative["audio"],negative["audio_mask"])
        prepared=SimpleNamespace(video=video,audio=audio,video_loss_mask=torch.tensor([[False,True,True]]))
        values=[]
        for cfg in (1.,7.):
            torch.manual_seed(4);renderer=FakeRenderer()
            values.append(guided_sample(renderer,prepared,negative,2,cfg,cfg,torch.tensor([1.,.5,0.])))
            for v,a in renderer.calls:
                self.assertTrue(torch.equal(v[:,-3:],u.video));self.assertTrue(torch.equal(a[:,-2:],u.audio))
        self.assertTrue(torch.equal(values[0]["video"][:,:1],video.latent[:,:1]))
        self.assertTrue(torch.allclose(values[0]["video"],values[1]["video"]))
        self.assertTrue(torch.allclose(values[0]["audio"],values[1]["audio"]))

    def test_invalid_schedule_rejected(self):
        with self.assertRaises(ValueError):
            guided_sample(None,SimpleNamespace(video=SimpleNamespace(latent=torch.zeros(1))),None,2,3,7,torch.tensor([.9,.5,0.]))


class SuppliedMedia(unittest.TestCase):
    def make_media(self,root):
        import av
        import numpy as np
        import soundfile as sf
        from PIL import Image
        speaker=root/"speaker.wav";listener=root/"listener.wav";image=root/"first.png";video=root/"listener.mp4"
        sf.write(speaker,np.zeros(32000,dtype=np.float32),16000)
        sf.write(listener,np.zeros(32000,dtype=np.float32),16000)
        pixels=np.zeros((32,32,3),dtype=np.uint8);Image.fromarray(pixels).save(image)
        with av.open(str(video),"w") as out:
            stream=out.add_stream("libx264",rate=25);stream.width=32;stream.height=32;stream.pix_fmt="yuv420p"
            for i in range(50):
                frame=av.VideoFrame.from_ndarray(pixels,format="rgb24")
                for packet in stream.encode(frame):out.mux(packet)
            for packet in stream.encode():out.mux(packet)
        o=obs();o.update(requested_duration_s=2.,speaker_audio=str(speaker),listener_image=str(image))
        o["media_hashes"]=dict(speaker_audio=sha(speaker),listener_image=sha(image))
        t=dict(schema="p2-listen-target-v1",sample_id="s1",target_video=str(video),target_audio=str(listener),
               duration_s=2.,admitted=True,media_hashes=dict(target_video=sha(video),target_audio=sha(listener)))
        return o,t

    def test_prepare_import_and_source_only_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);o,t=self.make_media(root)
            c=load();c["output"]=str(root/"experiment");c["data"]["size"]=32
            atomic_json(root/"observations.json",[o]);atomic_json(root/"targets.json",[t])
            report=prepare(c,root/"observations.json",root/"targets.json")
            self.assertEqual(report["counts"]["train"],1)
            l=label();l["duration_s"]=2.;l["coverage"]={k:[[0.,2.]] for k in l["coverage"]}
            l["provenance"]["media_hashes"]=t["media_hashes"]
            atomic_json(root/"labels.json",[l]);self.assertEqual(import_labels(c,root/"labels.json")["imported"],["s1"])
            Path(t["target_audio"]).unlink();Path(t["target_video"]).unlink()
            (root/"experiment/manifests/targets.json").unlink()
            (root/"experiment/labels/canonical/s1.json").unlink()
            self.assertEqual(load_observations(c)[0]["sample_id"],"s1")

    def test_unadmitted_media_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);o,t=self.make_media(root);t["admitted"]=False
            atomic_json(root/"o.json",[o]);atomic_json(root/"t.json",[t])
            c=load();c["output"]=str(root/"experiment")
            with self.assertRaises(ValueError):prepare(c,root/"o.json",root/"t.json")
            self.assertFalse((root/"experiment").exists())


if __name__=="__main__":unittest.main()
