"""Unavailable observations must never become zero counters or kill evidence."""
import contextlib
import importlib.util
import io
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('watch_observation_test', ROOT / 'scripts/listen_j_omni_watch_window.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class ObservationTests(unittest.TestCase):
    def run_probe(self, *, failures=0, mismatch=False, stalled=False):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            node = work / 'nodes/node17'
            heartbeat = work / 'workers/node17-gpu-0.json'
            watch.write(work / 'config.json', {'nodes': {'node17': {'hostname': socket.gethostname(), 'gpu_ids': [0]}}})
            watch.write(node / 'watchdog.json', {'pid': 20})
            watch.write(node / 'supervisor.json', {'pid': 30, 'children': {'gpu-0': {'pid': 10, 'start_ticks': 't'}}})
            progress = 100 if stalled else 1950
            watch.write(heartbeat, {'pid': 99 if mismatch else 10, 'stage': 'inference', 'completed_clips': 12,
                                    'heartbeat': 1999, 'last_progress_at': progress})
            pending = {'at': 1700, 'progress': 100, 'progress_token': [10, 't', 100, 12, None, None, None],
                       'owned_tree': [{'pid': 10, 'start_ticks': 't'}, {'pid': 11, 'start_ticks': 'u'}]}
            watch.write(node / 'window_guard_state.json', {'suspects': {}, 'pending': {'0': pending}})
            reads = [0]

            def read(path):
                if Path(path) == heartbeat:
                    reads[0] += 1
                    if reads[0] <= failures:
                        raise FileNotFoundError('transient heartbeat observation')
                return watch.read(path)

            spawn = Mock()
            modules = {
                'listen_benchmark.j_annotation': SimpleNamespace(read=read, write=watch.write),
                'listen_j_annotation_supervisor': SimpleNamespace(alive=lambda x: bool(x), identity=lambda p: 't', spawn=spawn),
                'listen_j_omni': SimpleNamespace(environment=lambda: {}, complete=lambda w: False),
            }
            output = io.StringIO()
            with patch.dict(sys.modules, modules), patch('time.time', return_value=2000), patch('time.sleep'), \
                 patch('subprocess.check_output', return_value='0, 120000, 0\n'), patch('os.kill') as kill, \
                 contextlib.redirect_stdout(output):
                error = None
                try:
                    exec(watch.REMOTE, {'ROOT': str(ROOT), 'WORK': str(work), 'NODE': 'node17'})
                except RuntimeError as caught:
                    error = caught
                result = json.loads(output.getvalue()) if output.getvalue() else None
                return SimpleNamespace(error=error, result=result, reads=reads[0], kills=kill.call_args_list,
                                       spawns=spawn.call_count, state=watch.read(node / 'window_guard_state.json'))

    def test_transient_missing_file_is_retried_and_keeps_real_counter(self):
        result = self.run_probe(failures=1)
        self.assertIsNone(result.error)
        self.assertEqual(result.reads, 2)
        self.assertEqual(result.result['workers'][0]['completed'], 12)
        self.assertEqual(result.kills, [])

    def test_persistent_missing_file_fails_observation_before_any_recovery(self):
        result = self.run_probe(failures=3)
        self.assertIn('heartbeat observation unavailable', str(result.error))
        self.assertEqual(result.reads, 3)
        self.assertIsNone(result.result)
        self.assertEqual(result.kills, [])
        self.assertEqual(result.spawns, 0)
        self.assertIn('0', result.state['pending'])

    def test_mismatching_heartbeat_defers_pending_escalation(self):
        result = self.run_probe(mismatch=True)
        self.assertIsNone(result.error)
        self.assertFalse(result.result['workers'][0]['matching_heartbeat'])
        self.assertEqual(result.kills, [])
        self.assertIn('0', result.state['pending'])

    def test_confirmed_unchanged_stall_can_still_escalate_owned_tree(self):
        result = self.run_probe(stalled=True)
        self.assertIsNone(result.error)
        self.assertEqual([call.args[0] for call in result.kills], [11, 10])
        self.assertEqual(result.result['actions'][0]['action'], 'kill_confirmed_stalled_owned_tree')
        self.assertNotIn('0', result.state['pending'])


if __name__ == '__main__':
    unittest.main()
