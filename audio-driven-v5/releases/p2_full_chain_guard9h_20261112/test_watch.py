import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('chain_guard',Path(__file__).with_name('watch.py'))
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class GuardChecks(unittest.TestCase):
    def test_exact_worker_ownership(self):
        for arm in ['no_whisper','full']:
            script=str(M.info(arm)['script'])
            args=[script,'train']+(['--arm','no_whisper'] if arm=='no_whisper' else [])
            self.assertTrue(M.owned({'argv':args},arm,'train'))
            self.assertFalse(M.owned({'argv':['other.py','train',script+'suffix']},arm,'train'))
            self.assertFalse(M.owned({'argv':[script,'node']},arm,'train'))
        self.assertFalse(M.owned({'argv':[str(M.info('no_whisper')['script']),'train','--arm','no_feature']},'no_whisper','train'))

    def test_reservation_and_recipe(self):
        self.assertEqual([n['gpus'] for n in M.Q.CL['nodes']], [list(range(8))]*3+[list(range(4))])
        for arm in ['no_whisper','full']:
            self.assertTrue(M.contract(arm)['success'])

    def test_incomplete_observation_cannot_kill(self):
        with patch.object(M.os,'kill') as kill:
            with self.assertRaises(AssertionError):M.terminate_owned('full',{'observed':False},None)
            kill.assert_not_called()

    def test_user_stop_blocks_recovery_before_kill(self):
        stop=M.info('full')['out']/'STOP'
        original=Path.exists
        with patch.object(Path,'exists',lambda p: True if p==stop else original(p)),patch.object(M,'terminate_owned') as terminate:
            with self.assertRaisesRegex(RuntimeError,'User STOP'):M.restart('full',{},dict(owner=None))
            terminate.assert_not_called()

    def test_dead_process_not_treated_as_live(self):
        self.assertFalse(M.live(None))
        self.assertFalse(M.live({'pid':999999999,'start':'0','cmd':'absent'}))

if __name__=='__main__':unittest.main()
