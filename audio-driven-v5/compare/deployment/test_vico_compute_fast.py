"""Regression checks for invalidation, fallback, export failure and resume."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from vico_compute_fast import install_dualtalk
from vico_more_common import save,sha
from vico_gpavatar_render_fast import reusable,pipeline


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
    def forward(self,x,frame_num):return SimpleNamespace(last_hidden_state=(x.mean(1)[:,None,None]+self.weight).expand(-1,frame_num,2))


class Joint(torch.nn.Module):
    def __init__(self):
        super().__init__();self.audio_encoder1=Encoder();self.audio_encoder2=Encoder()
        self.audio_projection=torch.nn.Linear(2,2);self.blendshape_encoder=torch.nn.Identity()
    def forward(self,a,b,c):
        import math
        n=math.ceil(a.shape[1]/16000*25)
        return self.audio_projection(self.audio_encoder1(a,n).last_hidden_state),self.audio_projection(self.audio_encoder2(b,n).last_hidden_state),c


class ComputeTests(unittest.TestCase):
    def setup_model(self):
        joint=Joint().eval();adapter=install_dualtalk(SimpleNamespace(joint_encoder=joint))
        return joint,adapter,(torch.zeros(1,1600),torch.ones(1,1600),torch.ones(1,3,2))

    def test_equal_and_weight_replacement_invalidates(self):
        j,a,args=self.setup_model()
        with torch.no_grad():
            reference=a.original(*args);j(*args);pred=j(*args)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(reference,pred)))
            self.assertEqual((a.hits,a.misses),(1,1))
            j.audio_encoder1.weight.add_(1)
            self.assertTrue(torch.equal(j(*args)[0],a.original(*args)[0]))
            self.assertEqual((a.hits,a.misses),(0,1))
            j.audio_encoder1.weight=torch.nn.Parameter(torch.full((1,),4.))
            self.assertTrue(torch.equal(j(*args)[0],a.original(*args)[0]))
            self.assertEqual((a.hits,a.misses),(0,1))

    def test_fallback_and_bounded_cache(self):
        j,a,args=self.setup_model();a.max_entries=2
        with torch.no_grad():
            j(args[1],args[1],args[2]);self.assertEqual(a.fallbacks,1)
            for n in (1600,3200,4800):j(torch.zeros(1,n),args[1],args[2])
            self.assertEqual(len(a.cache),2)
            j.train();j(*args);self.assertEqual(a.fallbacks,2)
        j.eval();j(*args);self.assertEqual(a.fallbacks,3)

    def test_corrupt_video_invalidates_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files={}
            for name in ('generated','raw','score'):
                path=root/(name+'.mp4');path.write_bytes(b'video');files[str(path)]=sha(path)
            receipt=root/'complete.json';save(receipt,dict(protocol_sha256='p',motion_sha256='m',frames=3,files=files))
            self.assertIsNotNone(reusable(receipt,'p','m',3))
            self.assertIsNone(reusable(receipt,'p','other',3))
            (root/'score.mp4').write_bytes(b'corrupt')
            self.assertIsNone(reusable(receipt,'p','m',3))

    def test_export_failure_aborts_all_pipes(self):
        import vico_gpavatar_render_fast as module
        created=[]
        class Pipe:
            def __init__(self,path,size):self.aborted=False;created.append(self)
            def write(self,pixels):raise RuntimeError('encoder failure')
            def abort(self):self.aborted=True
        gp=SimpleNamespace(render_motion=lambda *_:torch.zeros(1,3,512,512))
        with tempfile.TemporaryDirectory() as tmp,patch.object(module,'VideoPipe',Pipe):
            with self.assertRaisesRegex(RuntimeError,'encoder failure'):
                pipeline(gp,None,None,torch.zeros(3,56),range(3),Path(tmp),Path(tmp)/'score.mp4')
        self.assertEqual(len(created),3);self.assertTrue(all(p.aborted for p in created))

    def test_dualtalk_invalid_link_detached_without_archive_write(self):
        from vico_inference_fast import clean_dualtalk_partial,verified_records
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'archive.npz';archive.write_bytes(b'keep')
            dest=root/'predictions/clip.npz';dest.parent.mkdir();dest.symlink_to(archive)
            save(dest.with_suffix('.json'),dict(id='clip',path=str(archive),sha256=sha(archive)))
            clean_dualtalk_partial(root,'new-checkpoint',[dict(id='clip')])
            self.assertFalse(dest.is_symlink());self.assertEqual(archive.read_bytes(),b'keep')
            marker=root/'COMPLETE.json';marker.write_text('{broken')
            self.assertIsNone(verified_records(marker,['clip']))


if __name__=='__main__':unittest.main()
