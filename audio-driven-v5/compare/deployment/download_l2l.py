import gdown,tarfile,requests
from pathlib import Path
base=Path(__file__).resolve().parents[1]/'learning2listen/src'
for name,ident in [('conan','1HlGLMPcshqwdmQvryKPVsYvd9oL2yGML'),('fallon','1_d4D6T9qflgd15uA3xhtp9wvchWbg9Da'),('stephen','1gXt2pjpnPItCIfINKCToBacoTI-cVs0W'),('trevor','1M5y5J3NKhMbzIaU58_Gz8yOFQZuOVhCn')]:
 target=base/'models'/f'{name}_models.tar';target.parent.mkdir(exist_ok=True)
 try:
  if not target.exists():gdown.download(id=ident,output=str(target),quiet=False)
  with tarfile.open(target) as t:t.extractall(base)
 except Exception as e:print(name,type(e).__name__,str(e),flush=True)
for target in (base/'configs').glob('*_configs.tar'):
 with tarfile.open(target) as t:t.extractall(base)
import subprocess,sys
subprocess.run([sys.executable,str(Path(__file__).with_name('download_l2l_test.py'))],check=True)
