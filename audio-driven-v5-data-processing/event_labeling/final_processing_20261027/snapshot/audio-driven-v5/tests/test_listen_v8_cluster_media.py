import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import render_listen_v8_cluster as media
class MediaTests(unittest.TestCase):
    def test_only_release_candidates_export_even_when_old_qc_differs(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(media,'OUT',Path(tmp)):
            p=Path(tmp);(p/'results').mkdir();(p/'render_results').mkdir()
            rows=[dict(sample_id='approved',effective_decision='review',release={'pool':'candidate'}),dict(sample_id='noise',effective_decision='keep',release={'pool':'repair_reserve'}),dict(sample_id='role',effective_decision='keep',release={'pool':'role_review'})]
            media.write(p/'results/pair.json',dict(clips=rows))
            with patch.object(media.renderer,'export',side_effect=lambda row,verified:dict(sample_id=row['sample_id'])) as export:
                result=media.export_pair('pair')
                self.assertEqual(export.call_count,1);self.assertEqual(export.call_args.args[0]['effective_decision'],'candidate')
                self.assertEqual([r['sample_id'] for r in result['clips']],['approved'])
                self.assertEqual(result['clips'][0]['release']['pool'],'candidate')
    def test_claims_exhaust_and_stop(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(media,'SCHED',Path(tmp)),patch.object(media,'_QUEUE',None):
            p=Path(tmp);media.write(p/'queue.json',dict(pairs=['a','b']));media.write(p/'state.json',dict(cursor=0,total=2));media.write(p/'control.json',dict(stop=False))
            self.assertEqual(media.claim(),'a');media.write(p/'control.json',dict(stop=True));self.assertIsNone(media.claim())
            media.write(p/'control.json',dict(stop=False));self.assertEqual(media.claim(),'b');self.assertIsNone(media.claim())
if __name__=='__main__':unittest.main()
