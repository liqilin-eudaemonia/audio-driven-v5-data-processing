"""Independent numeric references; requires the optional evaluation dependencies."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import tempfile
import json
import contextlib
import io
import cv2
import numpy as np
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio
from .core import pixel_scores, frechet, response_scores


class ReferenceTests(unittest.TestCase):
    def test_psnr_and_independent_ssim(self):
        rng = np.random.default_rng(122)
        for size in [(32,48), (111,103), (224,224)]:
            a = rng.integers(0,256,(*size,3),dtype=np.uint8)
            b = np.clip(a.astype(int)+rng.integers(-25,26,a.shape),0,255).astype(np.uint8)
            scores = pixel_scores(a,b)
            self.assertAlmostEqual(scores['psnr'],peak_signal_noise_ratio(a,b,data_range=255),places=12)
            # Independent Wang SSIM with OpenCV filtering, valid convolution region.
            x,y = a.astype(float),b.astype(float)
            k = cv2.getGaussianKernel(11,1.5); window = k@k.T
            blur = lambda z: cv2.filter2D(z,-1,window)[5:-5,5:-5]
            mx,my = blur(x),blur(y)
            vx,vy,xy = blur(x*x)-mx*mx,blur(y*y)-my*my,blur(x*y)-mx*my
            c1,c2 = (0.01*255)**2,(0.03*255)**2
            reference = ((2*mx*my+c1)*(2*xy+c2)/((mx*mx+my*my+c1)*(vx+vy+c2))).mean()
            self.assertAlmostEqual(scores['ssim'],reference,places=12)

    def test_frechet_against_scipy_and_official_fid(self):
        from pytorch_fid.fid_score import calculate_frechet_distance
        rng = np.random.default_rng(13)
        for n,m,d in [(4,6,20),(30,25,8),(8,20,12),(64,48,32),(2,2,128)]:
            a,b = rng.normal(size=(n,d)),rng.normal(size=(m,d))+.3
            for population in (False,True):
                c1=np.cov(a,rowvar=False,ddof=0 if population else 1)
                c2=np.cov(b,rowvar=False,ddof=0 if population else 1)
                root=sqrtm(c1@c2)
                reference=float(np.sum((a.mean(0)-b.mean(0))**2)+np.trace(c1)+np.trace(c2)-2*np.trace(root).real)
                # sqrtm is less accurate on singular, high-dimensional covariance.
                np.testing.assert_allclose(frechet(a,b,population),reference,rtol=1e-7,atol=1e-6)
                if n==m==2:
                    # Rank-one closed form independently verifies our result to
                    # tight tolerance, rather than trusting noisy sqrtm zeros.
                    da,db=a[0]-a[1],b[0]-b[1]
                    denom=4 if population else 2
                    analytic=np.sum((a.mean(0)-b.mean(0))**2)+(da@da+db@db-2*abs(da@db))/denom
                    self.assertAlmostEqual(frechet(a,b,population),analytic,places=10)
                if not population:
                    official=calculate_frechet_distance(a.mean(0),c1,b.mean(0),c2)
                    np.testing.assert_allclose(frechet(a,b),official,rtol=1e-7,atol=1e-6)

    def test_response_against_explicit_paper_sum(self):
        rng=np.random.default_rng(14)
        g,p=rng.normal(size=(40,106,2)),rng.normal(size=(40,106,2))*2
        intervals=[[0,.8],[.8,2.]]; fps=20
        # Copy indices from paper, not from the implementation constant.
        indices=[43,48,49,50,51,101,102,103,104,105,35,36,37,39,41,42,89,90,91,93,95,96]
        total=0
        for start,end in intervals:
            frames=[j for j in range(len(g)) if start<=j/fps<end]
            motions=[]
            for track in (g,p):
                motion=sum(float(np.sqrt(sum((track[j+1,i]-track[j,i])**2)))
                           for j in frames[:-1] for i in indices)/((len(frames)-1)*len(indices))
                motions.append(motion)
            total+=(end-start)*abs(motions[0]-motions[1])
        reference=total/sum(e-s for s,e in intervals)
        self.assertAlmostEqual(response_scores(g,p,fps,intervals)['delta'],reference,places=12)

    def test_runner_aggregation_unequal_clips(self):
        from .run import evaluate
        pairs=[]; detections=[]
        for n,gain,labels,target in [(9,2,[0,1,2]*3,[0,1,2]*3),(3,4,[0]*3,[1]*3)]:
            frames=np.zeros((n,16,16,3),np.uint8)
            pairs.append((frames,frames,dict(fps=3,aligned_frames=n,gt_frames=n,generated_frames=n)))
            for scale,emotions in [(1,labels),(gain,target)]:
                points=np.zeros((n,106,2));points[:,:,0]=np.arange(n)[:,None]*scale
                boxes=np.tile([0.,0.,224.,224.],(n,1))
                detections.append((points,boxes,np.array(emotions),[1]*n))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=root/'input.json'
            manifest.write_text(json.dumps([dict(id=str(i),gt='gt.mp4',generated='p.mp4') for i in range(2)]))
            args=SimpleNamespace(threads=2,metrics='emo_acc,emo_div,delta_resp',batch_size=2,
                fid_frames=32,fvd_frames=32,fvd_stride=1,manifest=manifest,output=root/'out',
                assets=None,device='cpu',whole_clip_listening=True,allow_partial=False,face_padding=0)
            with patch('evaluation.video.paired_metrics.run.load_pair',side_effect=pairs), \
                 patch('evaluation.video.paired_metrics.backends.Faces') as factory, \
                 contextlib.redirect_stdout(io.StringIO()):
                factory.return_value.execution = dict(face_device="cpu", emotion_device="cpu")
                factory.return_value.extract.side_effect=detections
                self.assertEqual(evaluate(args),0)
            summary=json.loads((root/'out/summary.json').read_text())
            self.assertEqual(summary['metrics']['emo_acc'],.75) # 9/12, not (1+0)/2
            self.assertEqual(summary['metrics']['emo_div_gt'],2) # (3+1)/2
            self.assertEqual(summary['metrics']['emo_div_generated'],2)
            self.assertEqual(summary['metrics']['delta_resp'],1.5) # (3*1+1*3)/4


if __name__=='__main__':
    unittest.main()
