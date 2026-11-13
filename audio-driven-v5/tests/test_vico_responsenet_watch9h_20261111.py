import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('new_watch',ROOT/'scripts/p2_vico_responsenet_watch9h_20261111.py')
W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)


class RecoverySafety(unittest.TestCase):
    def check_no_launch(self,setup):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            mode=dict(work=work,script=work/'owned_eval.py',args=['controller'],runs=[work])
            with patch.dict(W.MODES,{'test':mode}),patch.object(W,'WORK',work),\
                 patch.object(W,'controller',return_value=dict(live=False)),\
                 patch.object(W,'live_children',return_value=[]),\
                 patch.object(W.subprocess,'Popen') as start:
                setup(work)
                W.recover('test',dict(end_at=W.time.time()+3600),{})
                start.assert_not_called()

    def test_explicit_stop_wins(self):
        self.check_no_launch(lambda work:(work/'STOP').touch())

    def test_finished_evaluation_never_restarts(self):
        self.check_no_launch(lambda work:(work/'COMPLETE.json').write_text('{}'))

    def test_live_controller_never_restarts(self):
        with patch.object(W,'controller',return_value=dict(live=True)),patch.object(W.subprocess,'Popen') as start:
            W.recover('responsenet',dict(end_at=W.time.time()+100),{})
            start.assert_not_called()

    def test_orphaned_worker_is_preserved(self):
        with patch.object(W,'controller',return_value=dict(live=False)),\
             patch.object(W,'live_children',return_value=[dict(pid=123,live=True)]),\
             patch.object(W.subprocess,'Popen') as start:
            W.recover('responsenet',dict(end_at=W.time.time()+100),{})
            start.assert_not_called()

    def test_no_restart_past_deadline(self):
        with patch.object(W,'controller',side_effect=AssertionError('No probe past deadline')),patch.object(W.subprocess,'Popen') as start:
            W.recover('responsenet',dict(end_at=W.time.time()-1),{})
            start.assert_not_called()

    def test_pid_reuse_does_not_match_foreign_process(self):
        self.assertFalse(W.identity(os.getpid(),['nonexistent_owned_eval_script.py'])['live'])
        self.assertFalse(W.identity(99999999,['owned_eval.py'])['live'])

    def test_streaming_recovery_requires_its_own_handoff_sentinel(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder);(work/'STOP').write_text('{"reason":"user_stop"}')
            mode=dict(work=work,script=work/'stream.py',args=['controller'],runs=[work])
            with patch.dict(W.MODES,{'streaming':mode}),patch.object(W,'WORK',work/'monitor'),\
                 patch.object(W,'controller',side_effect=AssertionError('User stop must prevent recovery')):
                W.recover('streaming',dict(end_at=W.time.time()+100),{})

    def test_explicit_stream_stop_wins_over_internal_sentinel(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder);(work/'STOP').write_text('{"owner_script":"stream.py"}');(work/'STREAM_STOP').touch()
            mode=dict(work=work,script=work/'stream.py',args=['controller'],runs=[work])
            with patch.dict(W.MODES,{'streaming':mode}),patch.object(W,'WORK',work/'monitor'),\
                 patch.object(W,'controller',side_effect=AssertionError('Do not probe after stream stop')):
                W.recover('streaming',dict(end_at=W.time.time()+100),{})


if __name__=='__main__':unittest.main()
