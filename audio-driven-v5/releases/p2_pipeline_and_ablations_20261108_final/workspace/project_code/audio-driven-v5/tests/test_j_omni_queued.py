import sys,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace as N
from concurrent.futures import Future
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_omni_queued import generate,groups,consume_source
from listen_benchmark.j_omni import EvidenceUnavailable
from listen_benchmark.j_annotation import result_path,read

class FakeLLM:
    def __init__(self,fail_once=None):self.calls=[];self.resets=0;self.fail_once=fail_once
    def reset_mm_cache(self):self.resets+=1
    def generate(self,requests,dec,use_tqdm):
        self.calls.append(requests);out=[]
        for r in requests:
            sid=r['multi_modal_data']['video'][0]
            finish='length' if sid==self.fail_once and len(self.calls)==1 else 'stop'
            out.append(N(prompt=r['prompt'],prompt_token_ids=[1,2,3],outputs=[N(text='The listener smiles '+sid+'.',finish_reason=finish,token_ids=[4,5])]))
        return out

P=N(image_processor=N(temporal_patch_size=2),apply_chat_template=lambda messages,**kw:messages[0]['content'][0]['text'])
C=dict(decoding_attempts=[dict(temperature=0),dict(temperature=0,repetition_penalty=1.12)],pipeline_id='same')
def rec(sid):return dict(row=dict(sample_id=sid,source=dict(sample_id=sid)),prepared=dict(prompt='identical prompt',frames=sid,audio='same silence',evidence={},recipe={},_preparation_metrics=dict(cuda_visible_devices='',seconds=1,pid=1)),wait_s=0)

class QueuedTests(unittest.TestCase):
    def test_pairing_and_resume_preserve_sample_identity(self):
        with tempfile.TemporaryDirectory() as d:
            llm=FakeLLM();rs=[rec(str(i)) for i in range(4)]
            out=generate(llm,P,rs,C,d,sampling_factory=lambda x:x)
            self.assertEqual([o['attempts'][-1]['caption'] for o in out],['The listener smiles '+str(i)+'.' for i in range(4)])
            ids=[r['multi_modal_uuids'] for r in llm.calls[0]]
            self.assertEqual(len({x['audio'][0] for x in ids}),4)
            self.assertTrue(all(x['audio']==x['video'] for x in ids));self.assertEqual(llm.resets,1)
            self.assertEqual(generate(llm,P,rs,C,d,sampling_factory=lambda x:x),out);self.assertEqual(len(llm.calls),1)
            rs[0]['prepared']['prompt']='changed input'
            with self.assertRaises(ValueError):generate(llm,P,rs,C,d,sampling_factory=lambda x:x)
            self.assertEqual(len(llm.calls),1)

    def test_retry_only_failed_item_and_resume_cached_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            llm=FakeLLM('b');rs=[rec('a'),rec('b')]
            out=generate(llm,P,rs,C,d,sampling_factory=lambda x:x)
            self.assertEqual([len(x) for x in llm.calls],[2,1]);self.assertEqual(llm.calls[1][0]['multi_modal_data']['video'][0],'b')
            self.assertEqual([len(x['attempts']) for x in out],[1,2]);self.assertFalse(out[1]['attempts'][-1]['failures'])
            again=generate(llm,P,rs,C,d,sampling_factory=lambda x:x);self.assertEqual(out,again);self.assertEqual(len(llm.calls),2)

    def test_short_tail_and_stop_are_not_filled_with_duplicates(self):
        stop=threading.Event();self.assertEqual(list(groups(iter(range(7)),4,stop)),[[0,1,2,3],[4,5,6]])
        stop.set();self.assertEqual(list(groups(iter(range(7)),4,stop)),[])

    def test_unavailable_evidence_is_held_and_never_sent_to_model(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            good=rec('good');bad=rec('bad');a=Future();a.set_result(good['prepared']);b=Future();b.set_exception(EvidenceUnavailable('not negative evidence'))
            stream=iter([(bad['row'],b,0),(good['row'],a,0)])
            state=dict(production_batch_size=4,preparation_wait_s=0,held_clips=0,completed_clips=0,failed_clips=0)
            fake=FakeLLM();real=generate
            with patch('listen_benchmark.j_omni_queued.generate',side_effect=lambda *args,**kw:real(*args,**kw,sampling_factory=lambda x:x)):
                consume_source(stream,fake,P,C,Path(d),'node',0,dict(backend='process'),'queue',{'good':{'decision':'retain'}},state,lambda:None,threading.Event())
            self.assertEqual(len(fake.calls[0]),1);self.assertEqual(state['held_clips'],1);self.assertEqual(state['completed_clips'],1)
            self.assertFalse(result_path(d,'caption','bad').exists());self.assertTrue(result_path(d,'held','bad').exists())
            out=read(result_path(d,'caption','good'));self.assertEqual(out['source'],good['row']['source']);self.assertEqual(out['engine_max_num_seqs'],1);self.assertEqual(out['submitted_requests'],1)

class ReadyGroupTests(unittest.TestCase):
    def test_ready_item_does_not_wait_for_a_slow_partner(self):
        import time
        from listen_benchmark.j_omni_queued import ready_groups
        release=threading.Event();finished=threading.Event()
        def source():
            try:
                yield 0
                release.wait(3)
                yield 1
                yield 2
            finally:finished.set()
        stream=source();g=ready_groups(stream,4,threading.Event());tick=time.monotonic()
        try:
            first=next(g);self.assertEqual(first,[0]);self.assertLess(time.monotonic()-tick,.5)
            release.set();self.assertEqual([x for batch in g for x in batch],[1,2])
        finally:release.set();g.close();stream.close()
        self.assertTrue(finished.is_set())

    def test_reader_error_propagates_and_reader_joins_before_source_close(self):
        from listen_benchmark.j_omni_queued import ready_groups
        def source():
            yield 1
            raise ValueError('real decoder failure')
        stream=source();g=ready_groups(stream,4,threading.Event())
        self.assertEqual(next(g),[1])
        with self.assertRaisesRegex(ValueError,'real decoder failure'):next(g)
        g.close();stream.close()

if __name__=='__main__':unittest.main()
