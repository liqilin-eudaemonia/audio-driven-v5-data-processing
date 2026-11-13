from pathlib import Path
import time,sys
p=Path(sys.argv[1])
for i in range(1200):
    tmp=p.with_suffix('.tmp');tmp.write_text(str(i));tmp.replace(p);time.sleep(.1)
