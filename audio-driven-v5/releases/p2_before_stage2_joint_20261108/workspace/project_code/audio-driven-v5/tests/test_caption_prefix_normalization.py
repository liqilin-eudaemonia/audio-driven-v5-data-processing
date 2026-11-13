import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from listen_response_sft.data import validated_target_text
from fix_test_caption_prefixes import clean_row


class PrefixTests(unittest.TestCase):
    def test_only_leading_prefix_is_removed(self):
        original = 'Assistant: She says "Assistant: yes" and nods twice.'
        row = dict(target=dict(format='original_caption_verbatim', text=original), inputs={'unchanged': True})
        cleaned = clean_row(row)
        self.assertEqual(cleaned['target']['text'], 'She says "Assistant: yes" and nods twice.')
        self.assertEqual(cleaned['inputs'], row['inputs'])
        self.assertEqual(row['target']['text'], original)
        self.assertEqual(validated_target_text(cleaned['target'], original), cleaned['target']['text'])
        self.assertIsNone(clean_row(dict(target=dict(text='She says Assistant: yes.'))))

    def test_reader_rejects_other_edits_and_forged_provenance(self):
        original = 'Assistant: She says Yeah twice.'
        target = clean_row(dict(target=dict(text=original)))['target']
        for field, value in [('text', 'She remains silent.'), ('original_text', 'Assistant: She remains silent.'), ('removed_prefix', 'Assistant')]:
            bad = copy.deepcopy(target)
            bad[field] = value
            with self.assertRaises(ValueError):
                validated_target_text(bad, original)

    def test_existing_verbatim_contract_remains_strict(self):
        target = dict(format='original_caption_verbatim', text='She nods.')
        self.assertEqual(validated_target_text(target, 'She nods.'), 'She nods.')
        with self.assertRaises(ValueError):
            validated_target_text(target, 'She smiles.')
        with self.assertRaises(ValueError):
            validated_target_text(dict(format='arbitrary_rewrite', text='She nods.'), 'She nods.')


if __name__ == '__main__':
    unittest.main()
