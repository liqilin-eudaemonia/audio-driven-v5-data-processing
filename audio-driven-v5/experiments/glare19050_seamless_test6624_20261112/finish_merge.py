import runpy,sys,time,json,subprocess,traceback
from pathlib import Path
w=Path(__file__).resolve().parent;root=w.parents[1]
sys.path[:0]=[str(root/'scripts'),str(root/'src'),str(root.parent/'OmniResponse')]
try:
 runpy.run_path(str(w/'verify_reader_compatibility.py'),run_name='__main__')
 import eval_glare19050_seamless_test_20261112 as evaluation
 evaluation.native.merge()
 subprocess.run([sys.executable,str(root/'scripts/render_glare_checkpoint_comparison.py')],cwd=root,check=True)
 print('COMPLETE',flush=True)
except BaseException:
 (w/'MERGE_RECOVERY_FAILURE.json').write_text(json.dumps(dict(at=time.time(),traceback=traceback.format_exc()),indent=2)+'\n')
 raise
