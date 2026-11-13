from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.visual_media_coverage import validate_terminal_support


class CoverageTests(unittest.TestCase):
    def test_actual_variable_duration_frame_covers_requested_end(self):
        validate_terminal_support(6.700000000000049,6.645,29.996753246753247,54)

    def test_preexisting_small_rounding_tolerance_is_retained(self):
        validate_terminal_support(6.63,6.645,30,54)

    def test_real_missing_support_empty_frames_and_invalid_times_fail(self):
        for args in [(6.5,6.645,30,54),(6.7,6.645,30,0),(6.7,6.645,30,257),(None,6.645,30,54),(6.7,6.645,0,54),(float('nan'),6.645,30,54)]:
            with self.subTest(args=args),self.assertRaises(ValueError):validate_terminal_support(*args)


if __name__=='__main__':unittest.main()
