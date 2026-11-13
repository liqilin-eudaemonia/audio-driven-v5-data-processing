import numpy as np
from p2_glare_rgb.detector import rgb_emotions,policy
from p2_glare_eval.common import ROOT,read

def test_rgb_boxes_and_original_bgr_emotion_branch():
 bgr=np.array([[[10,20,90]]],dtype=np.uint8);boxes=[[0,0,1,1]]
 class Fake:
  def find_faces(self,image):np.testing.assert_array_equal(image,np.array([[[90,20,10]]],dtype=np.uint8));return boxes
  def detect_emotions(self,image,face_rectangles):assert image is bgr and face_rectangles is boxes;return ['score']
 assert rgb_emotions(Fake(),bgr)==['score']
 np.testing.assert_array_equal(bgr,np.array([[[10,20,90]]],dtype=np.uint8))

def test_empty_boxes_explicitly_prevent_fer_fallback():
 class Fake:
  def find_faces(self,image):return []
  def detect_emotions(self,image,face_rectangles=None):assert face_rectangles==[];return []
 assert rgb_emotions(Fake(),np.zeros((1,1,3),dtype=np.uint8))==[]

def test_no_threshold_or_formula_relaxation():
 old=read(ROOT/'configs/p2_glare_detector_v1.json');new=policy()
 for key in old:
  if key not in ['schema','fer_input']:assert new[key]==old[key]
 assert new['required_frame_coverage']==1.0
 assert new['missing_frame_policy']=='reject_clip_no_interpolation_no_zero_fill'
