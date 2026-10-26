import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

from p3_listen.common import digest, sha, write
from p3_listen.features import cache_dir, load_native, paths, save


class InferenceIsolationTests(unittest.TestCase):
    def test_prediction_works_when_gt_feature_file_is_absent(self):
        row=dict(sample_id="sample",split="dev")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=cache_dir(root,row)
            marker=paths(root,row)["marker"]
            write(marker,{"input":"fixed"})
            layout=dict(prefix=torch.zeros(1,2,4),prefix_positions=torch.zeros(3,1,2,dtype=torch.long),
                        prefix_valid=torch.ones(1,2,dtype=torch.bool),coordinates=torch.zeros(1,2,3),
                        modalities=torch.tensor([[0,1]]),valid=torch.ones(1,2,dtype=torch.bool))
            save(folder/"layout.pt",layout)
            write(folder/"native.json",dict(contract=dict(row=digest(row),native=digest({}),media=sha(marker)),
                files={"layout.pt":sha(folder/"layout.pt"),"targets.pt":"deliberately absent"}))
            with patch("p3_listen.features.native_contract",return_value={}):
                value=load_native(root,row,{},targets=False)
                self.assertEqual(tuple(value.valid.shape),(1,2))
                with self.assertRaises(FileNotFoundError):
                    load_native(root,row,{},targets=True)


if __name__=="__main__":unittest.main()
