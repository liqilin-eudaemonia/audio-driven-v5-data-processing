"""Bind the native Feature model to the same current AV baseline/test inputs."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('feature_eval', ROOT/'scripts/p2_current_feature6000_sft2752_eval_20261108.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FeatureEvalTests(unittest.TestCase):
    def test_frozen_same_inputs_and_single_feature_increment(self):
        p = module.read(module.WORK/'protocol.json')
        av = module.read(module.AV_RESULT/'protocol.json')
        for key in ['datasets', 'generation', 'seed', 'planner_adapter_identity', 'shared_items_sha256']:
            self.assertEqual(p[key], av[key])
        a = av['checkpoint_identity']['experiment']
        f = p['checkpoint_identity']['experiment']
        for key in ['training', 'connector', 'text_lora', 'qwen_lora', 'current_cache_index', 'sample_list']:
            self.assertEqual(f[key], a[key])
        self.assertEqual(f['loss_weights'], dict(av=1., event=0., video_feature=.025, audio_feature=.025))
        self.assertFalse(any(k.startswith('whisper') for k in f))
        self.assertEqual(p['arms'][0]['kind'], 'current500h_av_feature')
        self.assertEqual(module.read(module.WORK/'cluster.json')['nodes'][0]['gpus'], [7])

    def test_inference_sampler_unchanged(self):
        change = next(x for x in module.ADAPTATIONS if x['function']=='render')
        self.assertEqual(len(change['replacements']), 1)
        self.assertEqual(change['replacements'][0]['old'], 'decoders=0, text_targets=len(model.text_targets), qwen_lora=0')
        self.assertEqual(len(module.read(module.WORK/'REUSED_CONDITIONS.json')['plans']), 400)


if __name__ == '__main__':
    unittest.main()
