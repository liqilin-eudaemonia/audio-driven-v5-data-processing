import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('capacity', ROOT / 'scripts/listen_j_expand_local_capacity.py')
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


class CapacityTests(unittest.TestCase):
    def test_gpu_with_resident_process_is_not_free_even_with_low_memory(self):
        with patch.object(capacity.subprocess, 'check_output', side_effect=[
                '3, empty, 1\n4, occupied, 1\n5, large, 54000\n', 'occupied, 123\n']):
            self.assertEqual(capacity.free_devices(), {3: True, 4: False, 5: False})

    def test_activation_keeps_existing_execution_and_updates_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp);work = base / 'work';quality = base / 'quality'
            capacity.write(work / 'config.json', {'gpu_ids': [0, 1, 2, 3],
                'gpu_execution_overrides': {'0': {'memory': .8}}, 'rolling_canary_gpus': [0, 1, 2, 3]})
            capacity.write(work / 'resource_tuning.json', {'decode_workers': 16, 'decode_slots': 56})
            capacity.write(quality / 'node8/original_config.json', {
                'gpu_execution_overrides': {str(i): {'memory': .8, 'decode_backend': 'process'} for i in range(8)}})
            scope = {'nodes': {n: {'gpu_ids': list(range(8 if n != 'node12' else 4))}
                                for n in ('node17', 'node8', 'node12')}}
            for p in (base / 'nodes.json', quality / 'annotation_nodes.json'):
                capacity.write(p, scope)
            with patch.multiple(capacity, BASE=base, WORK=work, QUALITY=quality):
                self.assertEqual(capacity.activate([4, 5, 6, 7]), list(range(8)))
            cfg = capacity.read(work / 'config.json')
            self.assertEqual(cfg['gpu_execution_overrides']['0'], {'memory': .8})
            self.assertEqual(cfg['nod_workers'], 80)
            self.assertEqual(capacity.read(work / 'resource_tuning.json'), {'decode_workers': 16, 'decode_slots': 112})
            self.assertEqual(capacity.read(base / 'nodes.json')['total_gpus'], 24)


if __name__ == '__main__':
    unittest.main()
