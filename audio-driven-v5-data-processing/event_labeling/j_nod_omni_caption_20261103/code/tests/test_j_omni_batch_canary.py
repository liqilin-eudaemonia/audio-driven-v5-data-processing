import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write
from listen_benchmark.j_omni_batch_canary import rollback,request


class CanaryTests(unittest.TestCase):
    def test_failure_reverts_only_its_engine_and_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=Path(tmp)
            original=dict(engine_max_num_seqs={'pilot':4,'other':1},startup_inference_benchmarks={'pilot':'trial','other':'unrelated'},preparation_profiles={'pilot':{'workers':8}},pipeline_id='unchanged')
            write(w/'config.json',original)
            rollback(w,'pilot',w/'trial',RuntimeError('simulated engine failure'))
            got=read(w/'config.json')
            self.assertEqual(got['engine_max_num_seqs'],{'other':1})
            self.assertEqual(got['startup_inference_benchmarks'],{'other':'unrelated'})
            self.assertEqual(got['preparation_profiles'],original['preparation_profiles'])
            self.assertEqual(got['pipeline_id'],'unchanged')
            self.assertTrue(read(w/'trial/error.json')['engine_reverted_to_single_request'])

    def test_pair_uuid_changes_no_media_prompt_or_processor_arguments(self):
        processor=SimpleNamespace(image_processor=SimpleNamespace(temporal_patch_size=2),apply_chat_template=lambda *args,**kwargs:'same rendered prompt')
        prepared=dict(frames=object(),audio=object(),prompt='same actual description request')
        baseline=request(processor,prepared)
        one=request(processor,prepared,'one');two=request(processor,prepared,'two')
        self.assertEqual(one.pop('multi_modal_uuids'),{'audio':['one'],'video':['one']})
        self.assertEqual(two.pop('multi_modal_uuids'),{'audio':['two'],'video':['two']})
        self.assertEqual(one,baseline);self.assertEqual(two,baseline)
        self.assertIs(one['multi_modal_data']['audio'][0],prepared['audio'])
        self.assertIs(one['multi_modal_data']['video'][0],prepared['frames'])


if __name__=='__main__':unittest.main()
