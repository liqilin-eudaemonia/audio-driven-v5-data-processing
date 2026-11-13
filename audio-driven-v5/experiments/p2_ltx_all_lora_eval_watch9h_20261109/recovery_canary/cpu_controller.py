from pathlib import Path
import json,os,time
p=Path(__file__).parent
(p/"RESTARTED.json").write_text(json.dumps(dict(pid=os.getpid(),at=time.time(),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),cpu_only=True)))
