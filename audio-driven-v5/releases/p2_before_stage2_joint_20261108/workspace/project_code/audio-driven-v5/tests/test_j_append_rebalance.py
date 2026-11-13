import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rebalance', ROOT / 'experiments/listen_j_annotation_three_nodes_20261030/expansion_24/rebalance.py')
rebalance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebalance)


class AppendTests(unittest.TestCase):
    def test_live_destination_offsets_and_completed_prefix_preserved_on_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); dest = root / 'dest'; stage = root / 'stage'; tx = root / 'tx'
            for p in (dest, stage, tx): p.mkdir()
            original = b'{"job_id":"done","count":2}\n'
            extra = b'{"job_id":"new","count":3}\n'
            (dest / 'ready.jsonl').write_bytes(original)
            (stage / 'ready.jsonl').write_bytes(extra)
            rebalance.write(dest / 'jobs/new.json', {'rows': []})
            state = {'offset': len(original), 'completed_clips': 2, 'inflight': {}, 'failed': {}}
            for branch in ('vlm', 'nod'): rebalance.write(dest / 'queues' / branch / 'state.json', state)
            rebalance.append_staged(dest, stage, tx)
            # Simulate a crash after journal replacement but before receipt commit.
            receipt = rebalance.read(tx / 'append.json'); receipt['state'] = 'prepared'
            rebalance.write(tx / 'append.json', receipt)
            rebalance.append_staged(dest, stage, tx)
            self.assertEqual((dest / 'ready.jsonl').read_bytes(), original + extra)
            for branch in ('vlm', 'nod'):
                self.assertEqual(rebalance.read(dest / 'queues' / branch / 'state.json'), state)
            self.assertEqual(rebalance.read(tx / 'append.json')['state'], 'committed')

    def test_duplicate_retry_journal_counted_once(self):
        self.assertEqual(rebalance.entries(b'{"job_id":"x","count":21}\n' * 2), {'x': 21})


if __name__ == '__main__': unittest.main()
