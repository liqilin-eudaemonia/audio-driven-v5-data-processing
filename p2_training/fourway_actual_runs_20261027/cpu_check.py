"""CPU regression from one isolated arm snapshot; never load pretrained weights."""
import os,sys,json
from pathlib import Path
B=Path(__file__).resolve().parent;arm=sys.argv[1];assert arm in ('E0','E1','E0F','E1F')
snapshot=B/'arms'/arm/'snapshot';root=snapshot/'audio-driven-v5'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
release='p2_feature_v1' if arm.endswith('F') else ('p2_preflight_v1' if arm=='E0' else None)
paths=([root/'releases'/release/'src'] if release else [])+[root/'src',root/'tests',snapshot/'audio-driven-v4/src']+[snapshot/'Omni-LTX-stage2/packages'/name/'src' for name in ['ltx-core','ltx-pipelines','ltx-trainer']]
sys.path[:0]=list(map(str,paths));os.chdir(root)
import p2_listen
from p2_listen import config,model,training
import omni_av_p2,ltx_core
origins={m.__name__:str(Path(m.__file__).resolve()) for m in [p2_listen,config,model,training,omni_av_p2,ltx_core]}
assert all(Path(p).is_relative_to(snapshot) for p in origins.values()),origins
import pytest
class Report:
 def pytest_sessionfinish(self,session,exitstatus):
  out=B/'validation'/arm;out.mkdir(parents=True,exist_ok=True)
  reporter=session.config.pluginmanager.get_plugin('terminalreporter')
  counts={key:len(value) for key,value in reporter.stats.items() if key in ('passed','failed','error','skipped','deselected')}
  (out/'cpu_tests.json').write_text(json.dumps(dict(arm=arm,exit_code=int(exitstatus),counts=counts,origins=origins,real_models_loaded=False,training_started=False),indent=2)+'\n')
args=['-q','-p','no:cacheprovider',*map(str,sorted((root/'tests').glob('test_p2_listen_*.py')))]
if release:args.append(str(root/'releases/p2_preflight_v1/tests/test_preflight.py'))
if arm.endswith('F'):
 args += [str(root/'releases/p2_feature_v1/tests/test_feature.py'),str(root/'experiments/p2_e1_feature01_train50h_20261026/tests/test_e1_feature.py'),'-k','not pair_is_only_arm_change and not launcher_uses_separate_e1_output_and_same_config and not release_preserved']
raise SystemExit(pytest.main(args,plugins=[Report()]))
