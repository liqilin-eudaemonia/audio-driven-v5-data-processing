import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glare_prosody_audit import parse_reply,frame_targets


class ProsodyAuditTests(unittest.TestCase):
    def test_explicit_null_is_a_known_negative(self):
        result=parse_reply(' \nNULL\n',2.)
        self.assertEqual(result['status'],'explicit_null')
        scores,valid=frame_targets(result,50)
        self.assertFalse(scores.any());self.assertTrue(valid.all())

    def test_bad_replies_never_become_null(self):
        for raw in ['', 'No clear events.', 'NULL\n0.00, 1.00, 0.80',
                    '-0.10, 0.20, 0.90','0.10, 3.00, 0.90','1.00, 0.00, 0.80',
                    '0.00, 1.00, 1.20','0.00, 1.00, nan',
                    '0.00, 1.00, 0.80\nextra explanation',
                    '0.00, 1.00, 0.80\n0.50, 1.50, 0.60',
                    '0.001, 0.004, 0.80']:
            with self.subTest(raw=raw):
                result=parse_reply(raw,2.)
                self.assertFalse(result['accepted']);self.assertIsNone(result['normalized_text'])
                self.assertEqual(result['raw_text'],raw)
                with self.assertRaises(ValueError):frame_targets(result,50)

    def test_below_threshold_is_distinct(self):
        result=parse_reply('0.00, 1.00, 0.30',2.)
        self.assertEqual(result['status'],'below_threshold')
        self.assertEqual(len(result['below_threshold']),1)
        self.assertFalse(frame_targets(result,50)[0].any())

    def test_native_rounding_and_endpoint_grid(self):
        result=parse_reply('0.201, 0.809, 0.754',2.)
        scores,valid=frame_targets(result,50)
        self.assertEqual(result['normalized_text'],'0.20, 0.81, 0.75')
        self.assertTrue((scores[5:21]==.75).all())
        self.assertFalse(scores[:5].any());self.assertFalse(scores[21:].any())

    def test_valid_rows_and_threshold_filter(self):
        result=parse_reply('0.00, 0.50, 0.40\n0.50, 1.00, 0.80',2.)
        self.assertEqual(result['status'],'events')
        self.assertEqual(len(result['events']),1);self.assertEqual(len(result['below_threshold']),1)

    def test_token_limit_cannot_create_known_negative(self):
        for raw in ['NULL','0.00, 1.00, 0.80','0.00,']:
            result=parse_reply(raw,2.,truncated=True)
            self.assertEqual(result['status'],'truncated');self.assertFalse(result['accepted'])

    def test_invalid_duration(self):
        for duration in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):parse_reply('NULL',duration)

    def test_unambiguous_format_repairs_preserve_numbers(self):
        variants=['0.00, 0.50, 0.80 0.60, 1.00, 0.70',
                  r'0.00, 0.50, 0.80\n0.60, 1.00, 0.70',
                  r'0.00, 0.50, 0.80\\n0.60, 1.00, 0.70',
                  '```csv\n0.00, 0.50, 0.80\n0.60, 1.00, 0.70\n```']
        for raw in variants:
            with self.subTest(raw=raw):
                parsed=parse_reply(raw,2.)
                self.assertTrue(parsed['accepted']);self.assertTrue(parsed['format_repairs'])
                self.assertEqual(parsed['normalized_text'],'0.00, 0.50, 0.80\n0.60, 1.00, 0.70')

    def test_numeric_garbage_cannot_be_reflowed(self):
        for raw in ['0.00, 1.00, 0.80 extra','0.00, 1.00, 0.80 NULL',
                    'I heard 0.00, 1.00, 0.80','0.00, 1.00, 0.80 123',
                    '0.00, 1.00, 0.80 -0.1, 1.5, 0.9']:
            self.assertFalse(parse_reply(raw,2.)['accepted'])


if __name__=='__main__':unittest.main()
