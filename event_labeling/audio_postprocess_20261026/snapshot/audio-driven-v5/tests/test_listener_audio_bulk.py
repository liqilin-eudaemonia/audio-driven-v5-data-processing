import unittest
from listen_audio_postprocess.bulk import official

class OfficialSourceContracts(unittest.TestCase):
    def test_missing_metadata_is_not_silence(self):
        for raw in [{},{'metadata:transcript':[]},{'metadata:transcript':[], 'metadata:vad':None}]:
            with self.assertRaisesRegex(ValueError,'Missing official'):official(raw,10,2)
    def test_present_empty_fields_are_valid_silence(self):
        result=official({'metadata:transcript':[],'metadata:vad':[]},10,2)
        self.assertEqual([result[k] for k in ('words','sentences','vad')],[[],[],[]])
    def test_relative_timestamps_and_neighbor_context(self):
        raw={'metadata:transcript':[{'start':10.3,'end':10.8,'transcript':'Yeah','words':[{'word':'Yeah','start':10.3,'end':10.8}]}], 'metadata:vad':[{'start':9.9,'end':10.4},{'start':100,'end':101}]}
        result=official(raw,10,2)
        self.assertEqual(result['words'][0]['start'],.3)
        self.assertEqual(result['words'][0]['text'],'Yeah')
        self.assertEqual(len(result['vad']),1)
        self.assertEqual(result['vad'][0]['start'],-.1)

if __name__=='__main__':unittest.main()
