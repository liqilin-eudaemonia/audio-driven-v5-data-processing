import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_listen_v8_full as full


class SchedulingTests(unittest.TestCase):
    def test_pending_review_blocks_full_run_but_allows_bounded_validation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(full,'OUT',Path(tmp)):
            full.write(Path(tmp)/'review_hold.json',dict(state='awaiting_user_review'))
            with self.assertRaisesRegex(RuntimeError,'on hold'):
                full.check_review_hold(0)
            full.check_review_hold(128)
            full.write(Path(tmp)/'review_hold.json',dict(state='accepted'))
            full.check_review_hold(0)

    def test_defaults_bound_both_processing_stages(self):
        args=full.argument_parser().parse_args(['detach'])
        self.assertEqual(args.workers,8)
        self.assertEqual(args.workers*args.prep,16)
        self.assertEqual(args.render_workers,16)

    def test_detach_preserves_explicit_concurrency_and_benchmark_limit(self):
        args=full.argument_parser().parse_args(['detach','--workers','8','--prep','1',
            '--render-workers','4','--limit','128','--no-render'])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'logs').mkdir()
            with patch.object(full,'OUT',out),patch.object(full,'verify'), \
                    patch.object(full.subprocess,'Popen') as popen,patch('builtins.print'):
                popen.return_value.pid=123
                full.detach(args)
                cmd=popen.call_args.args[0]
                for flag,value in [('--workers','8'),('--prep','1'),('--render-workers','4'),('--limit','128')]:
                    self.assertEqual(cmd[cmd.index(flag)+1],value)
                self.assertIn('--no-render',cmd)


if __name__=='__main__':unittest.main()
