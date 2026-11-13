"""Recovery evidence must survive later healthy polls and guardian restarts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('test_watch_window_module',ROOT/'scripts/listen_j_omni_watch_window.py')
watch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class JournalTests(unittest.TestCase):
    def test_between_report_recovery_survives_healthy_polls_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            watch.write(work/'config.json',{'nodes':{'node17':{'host':'unused'}}})
            watch.write(work/'queues/caption/state.json',{'completed_clips':1,'failed':{}})
            clock=[10000.]
            calls=[0]
            def probe(*args):
                calls[0]+=1
                return {'node':'node17','actions':[{'action':'restore_watchdog'}] if calls[0]==2 else []}
            def sleep(seconds):
                clock[0]+=seconds
            with patch.object(watch,'probe',side_effect=probe),patch.object(watch.time,'time',side_effect=lambda:clock[0]),patch.object(watch.time,'sleep',side_effect=sleep):
                watch.main(work,10000.,10065.)
                folder=work/'monitoring_10h'
                first=(folder/'health.jsonl').read_bytes()
                records=[json.loads(line) for line in first.splitlines()]
                self.assertEqual(len(records),4)
                self.assertEqual(records[1]['nodes']['node17']['actions'],[{'action':'restore_watchdog'}])
                self.assertEqual(watch.read(folder/'latest.json')['nodes']['node17']['actions'],[])
                self.assertEqual(watch.read(folder/'report-00.json')['nodes']['node17']['actions'],[])
                self.assertEqual(watch.read(folder/'state.json')['health_journal_started_at'],10000.)
                watch.main(work,10000.,10065.)
                second=(folder/'health.jsonl').read_bytes()
                self.assertTrue(second.startswith(first))
                self.assertEqual(len(second.splitlines()),5)
                self.assertEqual(watch.read(folder/'state.json')['health_journal_started_at'],10000.)


if __name__=='__main__':
    unittest.main()
