"""Only current Qwen/Gemma/AV readers; no label, Whisper or teacher reader."""
from pathlib import Path
import csv,threading
from collections import Counter
from p2_listen.contracts import read,sha,digest
ACTIVE=('qwen_hidden','gemma_text','av_vae')


def checked_index(c):
 index=read(c['current_cache_index'])
 if (index['status']!='complete' or index['samples']!=284218
     or index['source_split_counts']!={'train':281060,'dev':3158}):
  raise ValueError('Require the final train+dev500h selection; exclude test')
 ids=None
 if c['j_cache']!=index['caches']['qwen_hidden']['work']:
  raise ValueError('Condition cache differs from the final index')
 for name in ACTIVE:
  cache=index['caches'][name];p=Path(cache['work']);protocol=read(p/'protocol.json')
  if (sha(cache['complete'])!=cache['complete_sha256'] or sha(p/'protocol.json')!=cache['protocol_sha256']
      or cache['samples']!=index['samples']):raise ValueError('Current active cache changed: '+name)
  if ids is None:ids=protocol['sample_ids']
  if protocol['sample_ids']!=ids:raise ValueError('Active cache membership/order differs')
 if read(Path(index['caches']['qwen_hidden']['work'])/'protocol.json')['selection_sha256']!=index['selection_sha256']:
  raise ValueError('Latest selection binding differs')
 return index


def open_reader(index,verify=True):
 from p2_j_current_av_cache import Reader as AVReader
 from p2_j_current_gemma_cache import Reader as TextReader
 from p2_j_current_qwen_hidden import Reader as HReader
 from p2_plan_cache.runtime import dependencies
 from p2_whisper_av.compat import canonical_dependencies
 caches=index['caches'];av=AVReader(caches['av_vae']['work'],verify=verify)
 text=TextReader(caches['gemma_text']['work'],verify=False)
 if verify:
  if canonical_dependencies(dependencies(text.config,'text'))!=canonical_dependencies(text.protocol['dependencies']):
   raise ValueError('Gemma assets changed beyond audited byte-identical relocation')
  av.source.db.close();av.source.stream.close();av.source=HReader(caches['qwen_hidden']['work'],verify=True)
 av.text=text
 return av


class Dataset:
 def __init__(self,c,verify=True):
  self.spec=c;self.index=checked_index(c);self.reader=open_reader(self.index,verify)
  self.owner=threading.get_ident();self.local=threading.local()
  path=Path(c['sample_list'])
  if sha(path)!=c['duration_sampling']['sample_list_sha256']:raise ValueError('Sample list changed')
  with path.open()as f:self.rows=list(csv.DictReader(f))
  ids=[r['sample_id']for r in self.rows]
  if (ids!=self.reader.protocol['sample_ids'] or len(ids)!=len(set(ids))
      or dict(Counter(r['split']for r in self.rows))!=self.index['source_split_counts']):
   raise ValueError('AV-only lost source split, membership or order')

 def _reader(self):
  if threading.get_ident()==self.owner:return self.reader
  if not hasattr(self.local,'reader'):self.local.reader=open_reader(self.index,False)
  return self.local.reader

 def sample(self,index,device='cpu'):
  sid=self.rows[index]['sample_id'];r=self._reader();cached=r.training_sample(sid,'guided',device)
  if cached['metadata']['split']!=self.rows[index]['split']:raise ValueError('Source split differs')
  return dict(sample_id=sid,source_split=cached['metadata']['split'],condition=cached['condition'],
   layout=cached['layout'],text=cached['text'],av=dict(av=cached['av'],support=cached['support']),teacher=None)


def data_identity(c):
 index=checked_index(c)
 return dict(selection_sha256=index['selection_sha256'],samples=index['samples'],hours=index['hours'],
  source_split_counts=index['source_split_counts'],sample_list_sha256=sha(c['sample_list']),
  active_caches={k:index['caches'][k]for k in ACTIVE},objective='video_av+audio_av',
  text_policy='latest full caption verbatim, shared Qwen/Gemma prompt',qwen_hidden_source='base_pre_sft')


def data_config(c):
 index=checked_index(c)
 return read(Path(index['caches']['av_vae']['work'])/'protocol.json')['config']


def to_device(value,device):
 from p2_listen.cache import tree_to
 condition=value['condition'].to(device);result=tree_to(dict(value,condition=None),device)
 result['condition']=condition
 return result
