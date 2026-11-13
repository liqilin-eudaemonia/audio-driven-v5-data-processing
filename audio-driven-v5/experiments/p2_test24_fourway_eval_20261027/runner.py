"""Strict historical-source restoration for source-only evaluation, never checkpoint rewriting."""
import sys,runpy,importlib.util,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
args=sys.argv[1:];arm=args.pop(0);sys.argv=[str(ROOT/'scripts/p2_glare_eval.py'),*args]
setup=runpy.run_path(str(ROOT/'scripts/p2_eval_generate.py'));setup['bootstrap'](args)
import p2_listen.config as config
from p2_glare_eval import generation as gen
from p2_glare_eval.common import read,sha
restoration=read(OUT/'source_restoration.json')[arm]
# Every relocated file must be byte-identical to the checkpoint's original source hash.
for original,item in restoration.items():
 if sha(item['path'])!=item['sha256']:raise ValueError('Restored source tampered: '+original)
raw_sha=config.sha
config.sha=lambda path:raw_sha(restoration.get(str(path),{}).get('path',path))
for original,item in restoration.items():
 if original.endswith('/src/p2_listen/training_data.py'):
  name='p2_listen.training_data';spec=importlib.util.spec_from_file_location(name,item['path']);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
def checked_files(expected):
 for name,value in expected.items():
  target=restoration.get(name,{}).get('path',name)
  if sha(target)!=value:raise ValueError('File changed: '+name)
gen.checked_files=checked_files
runpy.run_path(str(ROOT/'scripts/p2_glare_eval.py'),run_name='__main__')
