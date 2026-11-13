import os
import unittest
from unittest.mock import patch

import torch

from p2_listen.numerics import POLICY, configure_training_numerics, validate_numerics


class TrainingNumerics(unittest.TestCase):
    def setUp(self):
        self.flags=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)

    def tearDown(self):
        enabled,warn,benchmark,deterministic=self.flags
        torch.use_deterministic_algorithms(enabled,warn_only=warn)
        torch.backends.cudnn.benchmark=benchmark
        torch.backends.cudnn.deterministic=deterministic

    def test_configures_determinism_before_cuda_without_allocating_context(self):
        with patch.dict(os.environ,clear=True),patch('torch.cuda.is_initialized',return_value=False),patch('torch.cuda.init') as init:
            record=configure_training_numerics()
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
            self.assertEqual(record['policy'],POLICY)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertEqual(record,configure_training_numerics())
            init.assert_not_called()

    def test_existing_supported_workspace_is_preserved(self):
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':16:8'}):
            self.assertEqual(configure_training_numerics()['cublas_workspace_config'],':16:8')

    def test_late_or_invalid_configuration_is_not_silently_accepted(self):
        with patch.dict(os.environ,clear=True),patch('torch.cuda.is_initialized',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'before initializing CUDA'):
                configure_training_numerics()
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':'invalid'}):
            with self.assertRaises(ValueError): configure_training_numerics()

    def test_changed_or_missing_checkpoint_numerics_refuses_replay(self):
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
            record=configure_training_numerics()
        validate_numerics(record,dict(record))
        for saved in (None,dict(record,warn_only=True),dict(record,cublas_workspace_config=':16:8'),dict(record,policy='old')):
            with self.assertRaisesRegex(ValueError,'numerics differ'):validate_numerics(saved,record)


if __name__=='__main__':unittest.main()
