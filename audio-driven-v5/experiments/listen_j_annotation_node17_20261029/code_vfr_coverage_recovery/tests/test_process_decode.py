from pathlib import Path
import os,pickle,sys,tempfile,unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.process_decode import read_decoded_reply,ProcessMediaDecoder,init_cpu_decoder
from listen_benchmark.visual_media_cache import cache_file


class ProcessDecodeTests(unittest.TestCase):
    def test_cached_reply_preserves_media_and_rejects_wrong_identity_or_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'archive';archive.write_bytes(b'bytes')
            row={'sample_id':'clip','source':{'source':{'archive':str(archive)}}}
            path=cache_file(row['source'],root);media={8:{'images':['unchanged'],'meta':{'frames':[1,2]}}}
            path.write_bytes(pickle.dumps(media))
            reply={'sample_id':'clip','cache_path':str(path),'decoder_cuda_visible_devices':'','decoder_pid':123,'metrics':{'decode_s':2}}
            result=read_decoded_reply(row,reply,root)
            self.assertEqual(result[:2],(row,media));self.assertEqual(result[2]['decode_backend'],'process')
            for change in [{'sample_id':'other'},{'cache_path':str(root/'untrusted')},{'decoder_cuda_visible_devices':'0'}]:
                with self.subTest(change=change),self.assertRaises(ValueError):
                    read_decoded_reply(row,dict(reply,**change),root)

    def test_input_failure_stays_explicit_and_does_not_load_cache(self):
        row={'sample_id':'clip'}
        reply={'sample_id':'clip','decoder_cuda_visible_devices':'','error':'decode failed'}
        with patch('pickle.load',side_effect=AssertionError('must not load')):
            self.assertEqual(read_decoded_reply(row,reply,'unused'),(row,None,{'error':'decode failed'}))

    def test_spawn_pool_reused_and_replaced_only_for_changed_capacity(self):
        instances=[]
        class Pool:
            def __init__(self,**kwargs):self.kwargs=kwargs;self.closed=False;instances.append(self)
            def map(self,fn,tasks):
                return [{'sample_id':t[0]['sample_id'],'decoder_cuda_visible_devices':'','error':'test'} for t in tasks]
            def shutdown(self,**kwargs):self.closed=True
        with patch('listen_benchmark.process_decode.ProcessPoolExecutor',Pool):
            decoder=ProcessMediaDecoder('cache','slots',True)
            for n in [2,2,4]:list(decoder.map([{'sample_id':'clip'}],n,8))
            self.assertEqual(len(instances),2);self.assertTrue(instances[0].closed)
            self.assertEqual(instances[1].kwargs['mp_context'].get_start_method(),'spawn')
            self.assertIs(instances[1].kwargs['initializer'],init_cpu_decoder)
            decoder.close();decoder.close();self.assertTrue(instances[1].closed)
            with self.assertRaises(RuntimeError):list(decoder.map([],4,8))

    def test_cpu_initializer_hides_gpu_and_limits_math_threads(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'7','OMP_NUM_THREADS':'8'}):
            init_cpu_decoder();self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'1')


if __name__=='__main__':unittest.main()
