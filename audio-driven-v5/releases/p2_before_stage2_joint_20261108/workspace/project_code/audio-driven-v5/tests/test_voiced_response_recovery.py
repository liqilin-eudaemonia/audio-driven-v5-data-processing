"""Contracts specific to voiced recovery; media truth is not inferred here."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import recover_voiced_response_sft as recovery


class RecoveryTests(unittest.TestCase):
    def answer(self, **changes):
        data = dict(keep=True, vocal_type='speech', issue='none', quote='says "Yeah"', reason='Explicit listener speech.')
        data.update(changes)
        return json.dumps(data)

    def test_old_visual_veto_is_removed_only_for_recovery(self):
        text = 'Wearing a red shirt against a gray background, she says "Yeah".'
        original = recovery.base.rule_evidence
        self.assertTrue(original(text))
        try:
            recovery.configure_policy()
            self.assertEqual(recovery.base.rule_evidence(text), [])
            self.assertTrue(recovery.parse_answer(self.answer(), 'stop', text)['keep'])
        finally:
            recovery.base.rule_evidence = original

    def test_keep_requires_positive_vocal_type_and_exact_evidence(self):
        text = 'She says "Yeah" and then remains silent.'
        for change in [dict(vocal_type='none'), dict(vocal_type='uncertain'), dict(issue='contradiction'), dict(quote='says "No"')]:
            with self.assertRaises(ValueError):
                recovery.parse_answer(self.answer(**change), 'stop', text)
        with self.assertRaises(ValueError):
            recovery.parse_answer(self.answer(), 'length', text)
        result = recovery.parse_answer(self.answer(quote='SAYS "Yeah"'), 'stop', text)
        self.assertEqual(result['quote'], 'says "Yeah"')

    def test_nonzero_waveform_is_only_candidate_gate(self):
        def row(flag):
            return dict(provenance=dict(processed_listener_waveform_nonzero=flag))
        self.assertFalse(recovery.candidate(row(False)))
        self.assertTrue(recovery.candidate(row(True)))
        text = 'She remains silent while another person speaks.'
        result = recovery.parse_answer(self.answer(keep=False, vocal_type='none', issue='no_listener_vocalization', quote='She remains silent'), 'stop', text)
        self.assertFalse(result['keep'])

    def test_external_evidence_is_rejected_even_for_negative_behavior_claims(self):
        self.assertTrue(recovery.rule_evidence('Supplementary data indicates no nodding. She says Yeah.'))
        self.assertTrue(recovery.rule_evidence('The nod is not visually confirmed. He laughs.'))
        self.assertEqual(recovery.rule_evidence('She wears red against a gray background and says Yeah.'), [])
        self.assertEqual(recovery.rule_evidence('He speaks, with no other vocalizations besides his speech.'), [])

    def test_merge_preserves_rows_and_refuses_duplicate_ids(self):
        with (recovery.base.SOURCE / 'train.jsonl').open('rb') as f:
            rows = [f.readline(), f.readline()]
        for duplicate in [False, True]:
            with tempfile.TemporaryDirectory() as tmp:
                old, work = Path(tmp) / 'old', Path(tmp) / 'new'
                for parent, raw in [(old, rows[0]), (work, rows[0] if duplicate else rows[1])]:
                    pool = parent / 'filtered'
                    pool.mkdir(parents=True)
                    (pool / 'train.jsonl').write_bytes(raw)
                    (pool / 'dev.jsonl').write_bytes(b'')
                    (pool / 'crop_config.json').write_text('{}')
                    recovery.base.save(pool / 'summary.json', dict(
                        splits={'train': {'counts': {'keep': 1}}, 'dev': {'counts': {'keep': 0}}},
                        manifests={f'{s}.jsonl': {'sha256': recovery.base.sha(pool / f'{s}.jsonl')} for s in ['train', 'dev']}))
                    recovery.base.save(parent / 'COMPLETE.json', dict(status='complete', summary_sha256=recovery.base.sha(pool / 'summary.json')))
                recovery.base.save(work / 'config.json', dict(previous_filter=str(old)))
                if duplicate:
                    with self.assertRaises(recovery.sqlite3.IntegrityError):
                        recovery.merge_retained(work)
                    self.assertFalse((work / 'MERGED_COMPLETE.json').exists())
                else:
                    recovery.merge_retained(work)
                    self.assertEqual((work / 'combined/train.jsonl').read_bytes(), b''.join(rows))
                    with recovery.sqlite3.connect(work / 'combined/index.sqlite') as db:
                        locations = db.execute('SELECT split,offset,bytes FROM samples ORDER BY offset').fetchall()
                    self.assertEqual(locations, [('train', 0, len(rows[0])), ('train', len(rows[0]), len(rows[1]))])


if __name__ == '__main__':
    unittest.main()
