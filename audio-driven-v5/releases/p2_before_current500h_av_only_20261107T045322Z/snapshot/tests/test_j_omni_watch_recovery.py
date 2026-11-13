"""Metadata misses must not create duplicate watchdogs or imply process exits."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('watch_recovery_test',ROOT/'scripts/listen_j_omni_watch_window.py')
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)


class RecoveryTests(unittest.TestCase):
    def helper(self, reader=None, live=None, function_name='watchdog_status'):
        living=set(live or [])
        def spawn(*args):
            living.add(42)
            return dict(pid=42,start_ticks='t',cmd=['owned-watchdog'])
        launch=Mock(side_effect=spawn)
        ns=dict(read=reader or watch.read,write=watch.write,acquire=watch.acquire,
                alive=lambda r:bool(r and r.get('pid') in living and r.get('start_ticks')=='t'),
                spawn=launch,time=SimpleNamespace(sleep=lambda _:None))
        fn=next(n for n in ast.parse(watch.REMOTE).body if isinstance(n,ast.FunctionDef) and n.name==function_name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'remote-recovery','exec'),ns)
        return ns[function_name],launch

    def test_controller_metadata_retry_preserves_actual_identity(self):
        record=dict(pid=20,start_ticks='t')
        reader=Mock(side_effect=[FileNotFoundError('transient'),record])
        check,_=self.helper(reader,live=[20],function_name='read_process_record')
        self.assertEqual(check(Path('controller.json')),record)
        self.assertEqual(reader.call_count,2)

    def test_controller_read_failure_is_not_reported_as_exit(self):
        check,_=self.helper(Mock(side_effect=OSError('read failed')),function_name='read_process_record')
        with self.assertRaisesRegex(RuntimeError,'observation unavailable'):
            check(Path('controller.json'))

    def test_transient_metadata_miss_keeps_live_watchdog(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);watch.write(folder/'watchdog.json',dict(pid=20,start_ticks='t'))
            reader=Mock(side_effect=[FileNotFoundError('transient'),dict(pid=20,start_ticks='t')])
            check,launch=self.helper(reader,live=[20])
            record,child,obs=check(folder,[],{},'test',True)
            self.assertEqual(record['pid'],20);self.assertTrue(obs['alive'])
            self.assertIsNone(child);launch.assert_not_called()

    def test_held_owner_lock_defers_recovery_even_without_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);owner=watch.acquire(folder/'watchdog.lock',False)
            try:
                check,launch=self.helper();_,child,obs=check(folder,[],{},'test',True)
                self.assertEqual(obs['reason'],'owner_lock_held');self.assertIsNone(child)
                launch.assert_not_called()
            finally:owner.close()

    def test_confirmed_absence_recovers_once_across_startup_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);check,launch=self.helper()
            _,child,obs=check(folder,[],{},'test',True)
            self.assertEqual(child['pid'],42);self.assertEqual(obs['reason'],'restoration_launched')
            _,child,obs=check(folder,[],{},'test',True)
            self.assertIsNone(child);self.assertEqual(obs['reason'],'pending_launch_alive')
            self.assertEqual(launch.call_count,1)

    def test_unreadable_metadata_is_not_a_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            check,launch=self.helper(Mock(side_effect=OSError('read failed')))
            _,child,obs=check(Path(directory),[],{},'test',True)
            self.assertIsNone(child);self.assertEqual(obs['reason'],'metadata_unavailable')
            launch.assert_not_called()

    def test_disabled_restoration_does_not_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            check,launch=self.helper();_,child,obs=check(Path(directory),[],{},'test',False)
            self.assertIsNone(child);self.assertEqual(obs['reason'],'restoration_disabled')
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
