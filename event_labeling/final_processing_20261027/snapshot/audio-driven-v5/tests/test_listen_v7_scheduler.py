import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_listen_v7_fast as fast


class SharedQueueTests(unittest.TestCase):
    def test_concurrent_claims_exactly_once_and_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=fast.SCHED;fast.SCHED=Path(tmp)
            try:
                queue=[f'pair_{i}' for i in range(137)]
                fast.bulk.write(fast.SCHED/'state.json',dict(cursor=0,total=len(queue)))
                def consume(rank):
                    output=[]
                    while True:
                        pair=fast.claim(queue,rank)
                        if pair is None:return output
                        output.append(pair)
                with ThreadPoolExecutor(max_workers=24) as pool:groups=list(pool.map(consume,range(24)))
                claimed=[p for rows in groups for p in rows]
                self.assertEqual(len(claimed),len(set(claimed)))
                self.assertEqual(set(claimed),set(queue))
                self.assertEqual(len(list((fast.SCHED/'claims').glob('*.json'))),len(queue))
                self.assertIsNone(fast.claim(queue,0))
                self.assertEqual(json.loads((fast.SCHED/'state.json').read_text())['cursor'],len(queue))
            finally:fast.SCHED=old


if __name__=='__main__':unittest.main()
