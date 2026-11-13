import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import plan_listen_v8 as planner
import run_listen_v8_full as full


class SourceFailureTests(unittest.TestCase):
    def test_empty_video_stream_is_explicit_source_failure(self):
        source=dict(archive='/test/data.tar',member='person.mp4',offset=512,byte_size=261)
        with patch.object(planner.base.media,'run',return_value=b'{"streams": []}'):
            with self.assertRaisesRegex(ValueError,'source_missing_video_stream:') as ctx:
                planner.video_stream(source)
        self.assertEqual(json.loads(str(ctx.exception).split(': ',1)[1]),source)

    def test_successful_legacy_load_does_not_reprobe(self):
        loaded=object()
        with patch.object(planner.base,'load_person',return_value=loaded), patch.object(planner,'video_stream') as probe:
            self.assertIs(planner.load_person({}),loaded)
            probe.assert_not_called()

    def test_unrelated_index_error_is_not_classified_as_missing_video(self):
        archive=MagicMock()
        member=archive.__enter__.return_value.getmember.return_value
        member.name='person.mp4';member.offset_data=512;member.size=1000
        with patch.object(planner.base,'load_person',side_effect=IndexError('other bug')), \
                patch.object(planner.tarfile,'open',return_value=archive), \
                patch.object(planner,'video_stream',return_value={'codec_type':'video'}):
            with self.assertRaisesRegex(IndexError,'other bug'):
                planner.load_person(dict(id='person',archive='/test/data.tar'))

    def test_missing_video_is_audited_without_normalization(self):
        with patch.object(full.planner,'inspect_pair',side_effect=ValueError('source_missing_video_stream: evidence')), \
                patch.object(full,'write') as write,patch.object(full.legacy,'normalize') as normalize:
            audit,norm,_=full.prepare(dict(pair='missing'))
        self.assertFalse(audit['ok']);self.assertEqual(audit['clips'],[])
        self.assertTrue(audit['unresolved'][0]['reason'].startswith('source_missing_video_stream:'))
        self.assertIsNone(norm);normalize.assert_not_called();write.assert_called_once()

    def test_unknown_preparation_errors_still_fail(self):
        with patch.object(full.planner,'inspect_pair',side_effect=ValueError('unknown bug')):
            with self.assertRaisesRegex(ValueError,'unknown bug'):
                full.prepare(dict(pair='broken'))


if __name__=='__main__':unittest.main()
