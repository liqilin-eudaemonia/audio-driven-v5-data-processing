"""Conservative task ownership, independent of the Python environment/model path."""
from pathlib import Path

GENERIC={'python','python3','python3.13','bash','sh','zsh','env','uv','torchrun','accelerate','vllm'}
INFRA=('nvidia-','nv-hostengine','Xorg','systemd','sshd')
def inside(path,root):
 try:return Path(path).resolve().is_relative_to(Path(root).resolve())
 except (OSError,ValueError):return False

def task_paths(argv,cwd):
 found=[]
 if argv:
  exe=Path(argv[0]);exe=exe if exe.is_absolute()else Path(cwd)/exe
  if exe.name not in GENERIC and exe.is_file()and not any(x in exe.parts for x in ['site-packages','.venv','venv']):found.append(str(exe.resolve()))
 for token in argv:
  if not token.endswith(('.py','.sh'))or any(x in token for x in ['\n','\x00']):continue
  p=Path(token);p=p if p.is_absolute()else Path(cwd)/p
  try:p=p.resolve()
  except OSError:continue
  if any(x in p.parts for x in ['site-packages','.venv','venv']):continue
  if p.is_file():found.append(str(p))
 return found

def classify(chain,root,complete,protected=False):
 if protected:return 'owned','exact_training_pid_start_ticks'
 if chain and chain[0].get('comm','').startswith(INFRA):return 'infrastructure','protected_gpu_service'
 for r in chain:
  if r.get('cwd')and inside(r['cwd'],root):return 'owned','workspace_cwd'
  if any(inside(p,root)for p in r.get('task_paths',[])):return 'owned','workspace_task_or_launcher_script'
 if complete and chain and not any(r.get('error')for r in chain):return 'foreign','complete_ancestry_outside_workspace'
 return 'unknown','incomplete_process_attribution'
