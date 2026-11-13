import hashlib,json,sys,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_testsets_eval.shared_items import all_items,claim_items
from p2_testsets_eval.runtime import complete


def setup(tmp_path,n=24):
 items=[dict(key='responsenet/'+str(i),id=str(i),geometry=dict(native_frames=8*i+1)) for i in range(n)]
 path=tmp_path/'items.json';path.write_text(json.dumps(items))
 return items,dict(shared_item_queue=True,shared_items_manifest=str(path),shared_items_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),total_samples=n)


def commit(path):
 path.mkdir(parents=True,exist_ok=True);f=path/'payload.pt';f.write_bytes(b'fixed');complete(path,[f])


def test_multi_gpu_claims_once_with_uneven_worker_speeds(tmp_path):
 items,p=setup(tmp_path);arm=dict(name='a');calls=[];mu=threading.Lock();barrier=threading.Barrier(8)
 def worker(rank):
  barrier.wait()
  for item in claim_items(tmp_path,[],p,arm,'render'):
   with mu:calls.append(item['key'])
   time.sleep(.001*(rank+1))
   commit(tmp_path/'latents/a'/item['key'])
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
 assert len(calls)==len(set(calls))==len(items)
 assert list(claim_items(tmp_path,[],p,arm,'render'))==[]


def test_completed_media_skip_and_exception_releases_claim(tmp_path):
 items,p=setup(tmp_path,2);arm=dict(name='a')
 commit(tmp_path/'generated/a'/items[0]['key'])
 iterator=claim_items(tmp_path,[],p,arm,'render');item=next(iterator);assert item['key']==items[1]['key']
 iterator.close()
 assert [x['key'] for x in claim_items(tmp_path,[],p,arm,'render')]==[items[1]['key']]
 assert list(claim_items(tmp_path,[],p,arm,'decode'))==[]
 commit(tmp_path/'latents/a'/items[1]['key'])
 assert [x['key'] for x in claim_items(tmp_path,[],p,arm,'decode')]==[items[1]['key']]


def test_fixed_candidate_list_rejects_mutation_and_default_unchanged(tmp_path):
 items,p=setup(tmp_path)
 assert all_items(tmp_path,[],p)[0]['geometry']['native_frames']==max(x['geometry']['native_frames'] for x in items)
 assert all_items(tmp_path,items,{}) is items
 Path(p['shared_items_manifest']).write_text('[]')
 with pytest.raises(ValueError,match='changed'):all_items(tmp_path,[],p)


def test_drain_request_finishes_current_sample_then_stops(tmp_path):
 items,p=setup(tmp_path,3);arm=dict(name='a')
 iterator=claim_items(tmp_path,[],p,arm,'render')
 first=next(iterator)
 (tmp_path/'STOP').write_text('drain')
 commit(tmp_path/'latents/a'/first['key'])
 assert list(iterator)==[]
 (tmp_path/'STOP').unlink()
 assert len(list(claim_items(tmp_path,[],p,arm,'render')))==2
