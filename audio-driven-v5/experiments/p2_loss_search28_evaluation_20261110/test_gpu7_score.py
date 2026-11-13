import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import gpu7_score as driver


class GPU7Checks(unittest.TestCase):
    def test_four_shards_preserve_all_items_and_can_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder);(work/'shards').mkdir()
            items=[dict(key='seamless/'+str(i),geometry=dict(native_frames=25,width=512,height=512))for i in range(64)]
            (work/'shared_items.json').write_text(json.dumps(items))
            (work/'cluster.json').write_text(json.dumps(dict(nodes=[dict(host=driver.HOST,gpus=[0,1,2,3])])) )
            p=dict(workers=28,total_samples=64,seed=137,generation=dict(steps=30))
            with patch.object(driver.E,'prepare',return_value=copy.deepcopy(p)):
                configured=driver.configure(work)
            shards=[x for i in range(4)for x in json.loads((work/f'shards/{i:02d}.json').read_text())]
            self.assertEqual({x['key']for x in shards},{x['key']for x in items})
            self.assertEqual(len(shards),64)
            self.assertEqual(configured['workers'],4)
            self.assertEqual(configured['seed'],137)
            with patch.object(driver.E,'prepare',return_value=configured):
                self.assertEqual(driver.configure(work),configured)

    def test_reject_other_host_gpu_or_uuid(self):
        p=dict(scoring_device=dict(host=driver.HOST,node='node11',physical_gpu=7,uuid=driver.UUID),execution_code={})
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='7'),patch.object(driver.E,'code_identity',return_value={}):
            with patch.object(driver.subprocess,'check_output',side_effect=['172.29.189.119',driver.UUID]):
                driver.validate_device(p)
            with patch.object(driver.subprocess,'check_output',return_value='172.29.252.37'):
                with self.assertRaises(ValueError):driver.validate_device(p)
            with patch.object(driver.subprocess,'check_output',side_effect=['172.29.189.119','wrong UUID']):
                with self.assertRaises(ValueError):driver.validate_device(p)
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'),patch.object(driver.subprocess,'check_output',return_value=driver.HOST):
            with self.assertRaises(ValueError):driver.validate_device(p)

    def test_every_remote_stage_uses_only_node11_gpu7(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            (work/'cluster.json').write_text(json.dumps(dict(nodes=[dict(name='node11',host=driver.HOST)])))
            (work/'protocol.json').write_text(json.dumps(dict(arms=[dict(name='test')])) )
            for stage in ('render','decode','metric','events'):
                with patch('listen_cluster.transport',side_effect=lambda c,n,a:(n,a)),patch('stage_driver.command')as command:
                    driver.remote(work,stage)
                    node,args=command.call_args.args[0]
                    self.assertEqual(node['host'],driver.HOST)
                    self.assertEqual([x for x in args if x.startswith('CUDA_VISIBLE_DEVICES=')],['CUDA_VISIBLE_DEVICES=7'])
                    self.assertTrue(command.call_args.kwargs['stdin_guard'])


if __name__=='__main__':unittest.main()
