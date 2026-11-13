import json,time,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
previous=None
with (r/'resources.jsonl').open('a',buffering=1) as out:
 while not (r/'STOP_MONITOR').exists():
  t=time.time();ticks=[int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
  cpu=None if previous is None else 100*(1-(ticks[3]+ticks[4]-previous[3]-previous[4])/(sum(ticks)-sum(previous)))
  previous=ticks
  try:g=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader,nounits'],text=True,timeout=8)
  except Exception as e:g=repr(e)
  out.write(json.dumps({'time':t,'cpu_percent':cpu,'gpus':g})+'\n');time.sleep(5)
