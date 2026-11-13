import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import listen_v8_head_shoulders as c

class HeadShoulderTests(unittest.TestCase):
    def geometry(self,face=190):
        k=np.ones((100,133,3));k[:,:,2]=.99
        k[:,23:91,0]=np.tile(np.linspace(480-face/2,480+face/2,17),4)
        k[:,23:91,1]=np.repeat(np.linspace(180,180+face,4),17)
        k[:,5,:2]=(280,440);k[:,6,:2]=(680,440)
        return dict(fps=25,width=1080,height=1080,arrays={'boxes_and_keypoints:keypoints':k})
    def test_includes_shoulders_without_padding_or_small_output_face(self):
        r=self.geometry();old=dict(x=280,y=100,size=420,valid_fraction=1.,face_width=190.,face_height=190.,wide_crop=False)
        with patch.object(c,'BASE_CROP',return_value=old):crop=c.crop(r,0,3)
        self.assertGreaterEqual(crop['size'],512);self.assertGreaterEqual(crop['output_face_short_p10_px'],128)
        self.assertLessEqual(crop['x'],280);self.assertGreaterEqual(crop['x']+crop['size'],680)
        self.assertLessEqual(crop['x']+crop['size'],1080)
    def test_small_face_not_rescued_by_large_box(self):
        with patch.object(c,'BASE_CROP',return_value=dict(size=300)):
            with self.assertRaisesRegex(ValueError,'native_face_below_160'):c.crop(self.geometry(120),0,3)
    def test_existing_large_crop_unchanged(self):
        old=dict(size=500,wide_crop=False)
        with patch.object(c,'BASE_CROP',return_value=old):self.assertIs(c.crop({},0,3),old)

if __name__=='__main__':unittest.main()
