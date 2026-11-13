"""Refresh explanatory notes, publish, and perform artifact/browser acceptance."""
import ctypes
assert ctypes.CDLL(None,use_errno=True).prctl(41,1,0,0,0)==0
import json,os,subprocess,sys,time
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
deadline=time.monotonic()+1800
while 'READY FOR BROWSER' not in (EXP/'finish.log').read_text():
 if time.monotonic()>deadline:raise RuntimeError('Export/publish incomplete; inspect finish.log')
 time.sleep(5)
path=EXP/'review/localbc/manifest.json';m=json.loads(path.read_text())
for d in m['pairs']:
 for c in d['comparison_cases']:
  if c['id']=='change4':c['note']='新版从 359.759 秒开始，比旧版提前 341 毫秒，补回 359.819–359.959 秒的 it’s。原始词对齐和不完整句提示仍保留；新起点进入对方 VAD 活动区（359.680–359.950 秒），仍需试听两人重叠声音，不能仅凭补回单词认定更好。'
  if c['id']=='change3':c['note']='重点复核：B 的 Okay. 在源时间 213.343–213.963 秒。后面的 Well 在 213.983 秒开始，仅隔 20 毫秒。新版把 Okay 独立归为反馈后，未纳入前一段，也未纳入随后由 B 主讲的片段。旧版保留，新版遗漏；这是新规则的衔接缺口，不应算作改进。'
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
for script in ['validate_data.py','publish.py']:
 subprocess.run([sys.executable,str(EXP/script)],cwd=ROOT,check=True)
env=dict(os.environ,PYTHONPATH='/tmp/v8_browser_deps',LD_LIBRARY_PATH='/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu')
subprocess.run([sys.executable,str(EXP/'check_browser.py')],cwd=ROOT,env=env,check=True)
print('ACCEPTANCE COMPLETE',flush=True)
