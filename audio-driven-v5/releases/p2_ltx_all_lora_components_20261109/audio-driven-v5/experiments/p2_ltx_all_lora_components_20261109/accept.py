from pathlib import Path
import sys,time,shutil,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read,sha,atomic_json
from p2_ltx_all_lora_components.config import source_identity,load
from p2_ltx_all_lora_components.data import data_identity
from p2_ltx_all_lora_components.training import check_preflight
from p2_ltx_all_lora_components.schedule import Schedule
W=Path(__file__).parent
old=read(W/'ACTIVE_SOURCE_BEFORE.json');assert all(sha(p)==h for p,h in old.items())
suite=ET.parse(W/'tests.xml').getroot().find('testsuite');assert int(suite.attrib['tests'])==23 and int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
src=source_identity();arms={}
full=read(ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108/preflight.json')
for arm in ['no_feature','no_whisper']:
 out=W/arm;c=load(out/'config.json');pre=read(out/'preflight.json');check_preflight(c,pre,src,data_identity(c))
 assert pre['samples']==284218 and len(pre['canary_reads'])==32 and {v['split']for v in pre['canary_reads']}=={'train','dev'}
 assert pre['statistics']==full['statistics']
 assert sum(v['enabled']for v in pre['statistics'].values())==18
 assert not(out/'run.json').exists()and not(out/'status.json').exists()
 log=(out/'COLD_TRAIN.log').read_text();assert log.rstrip().endswith('RuntimeError: Real training requires GPU; CPU tests use a tiny native model')
 schedule=Schedule(c);assert schedule.size==284218 and len(schedule.pairs)==284239
 if arm=='no_feature':old_pairs=schedule.pairs
 else:assert schedule.pairs==old_pairs
 arms[arm]=dict(config_sha256=sha(out/'config.json'),preflight_sha256=sha(out/'preflight.json'),meta=read(out/'META_COVERAGE.json'),real_cache_reads=32,classes=19,enabled_classes=18,cold_entry_passed=True,gpu_training_started=False)
result=dict(success=True,at=time.time(),tests=23,arms=arms,source=src,active_source_unchanged=len(old),real_gpu_validation_pending=True,queue_not_modified=True,sample_noise_plan_equal=True)
atomic_json(W/'PREPARATION_ACCEPTED.json',result)
dest=ROOT/'releases/p2_ltx_all_lora_components_20261109';files={Path(p)for p in src}
files.update(p for p in W.rglob('*')if p.is_file()and '__pycache__'not in p.parts)
files.add(ROOT/'tests/test_p2_ltx_all_lora_components.py');manifest={}
for p in sorted(files):
 q=dest/p.relative_to(ROOT.parent);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);h=sha(p);assert sha(q)==h;manifest[str(p)]=h
atomic_json(dest/'MANIFEST.json',dict(at=time.time(),files=manifest))
print(dict(success=True,tests=23,real_reads=64,active_source_unchanged=len(old),backup_files=len(files),gpu_started=False))
