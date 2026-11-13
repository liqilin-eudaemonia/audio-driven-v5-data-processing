"""Execution-only scheduling must preserve membership and surface failures."""
import importlib.util
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('teacher_cluster',Path(__file__).resolve().parents[1]/'scripts/p2_j_teacher_cluster.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Reader:
    source=SimpleNamespace(manifest={})
    def item(self,sid):return {'sample_id':sid}
    def load(self,sid,missing_ok=False):return {} if sid=='cached' else None


def test_ready_item_bypasses_slow_head_and_cached_is_preserved():
    class Pool:
        def __init__(self):self.futures={}
        def submit(self,fn,item,*args):
            f=Future();self.futures[item['sample_id']]=f
            if item['sample_id']=='fast':f.set_result('fast-result')
            return f
    pool=Pool();stream=module.prepared_stream_ready(Reader(),['slow','fast','cached'],pool,None,None,3)
    assert next(stream)==({'sample_id':'cached'},None)
    assert next(stream)==({'sample_id':'fast'},'fast-result')
    pool.futures['slow'].set_result('slow-result')
    assert list(stream)==[({'sample_id':'slow'},'slow-result')]


def test_failed_preparation_is_not_silently_skipped():
    class Pool:
        def submit(self,*args):
            f=Future();f.set_exception(ValueError('bad source'));return f
    with pytest.raises(ValueError,match='bad source'):
        list(module.prepared_stream_ready(Reader(),['bad'],Pool(),None,None,2))


def test_bounded_submission_and_complete_membership():
    class Pool:
        def __init__(self):self.submitted=[]
        def submit(self,fn,item,*args):
            self.submitted.append(item['sample_id']);f=Future();f.set_result(item['sample_id']);return f
    pool=Pool();stream=module.prepared_stream_ready(Reader(),[str(i) for i in range(20)],pool,None,None,4)
    first=next(stream);assert len(pool.submitted)<=5
    rows=[first]+list(stream)
    assert sorted(int(item['sample_id']) for item,value in rows)==list(range(20))
    assert all(item['sample_id']==value for item,value in rows)
