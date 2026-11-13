import sys,tempfile,unittest
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from listen_benchmark.shared_visual_decode import save_native,load_native
class SharedTests(unittest.TestCase):
 def test_roundtrip_and_source_binding(self):
  with tempfile.TemporaryDirectory() as d:
   frames=[np.zeros((512,512,3),dtype=np.uint8),np.ones((512,512,3),dtype=np.uint8)];src={'sample_id':'a'}
   save_native(d,src,frames,[0.,.04],'original');arr,t,reader=load_native(d,src)
   self.assertTrue(np.array_equal(arr,np.stack(frames)));self.assertEqual(t,[0.,.04]);self.assertEqual(reader,'original')
   with self.assertRaises(ValueError):load_native(d,{'sample_id':'b'})
 def test_corrupted_pixels_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   save_native(d,{},[np.zeros((512,512,3),dtype=np.uint8)],[0.],'original')
   data=np.load(Path(d)/'rgb.npy',mmap_mode='r+');data[0,0,0,0]=1;data.flush()
   with self.assertRaises(ValueError):load_native(d,{})
 def test_nonmonotonic_timestamps_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):save_native(d,{},[np.zeros((512,512,3),dtype=np.uint8)]*2,[.1,.1],'original')
if __name__=='__main__':unittest.main()
