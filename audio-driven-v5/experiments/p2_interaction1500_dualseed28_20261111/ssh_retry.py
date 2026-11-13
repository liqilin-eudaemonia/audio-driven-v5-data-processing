"""Retry only SSH key-exchange failures before any remote worker starts."""
import sys,subprocess,tempfile,time,select,os,stat,signal

def run(argv,guarded=False):
 fifo=guarded and stat.S_ISFIFO(os.fstat(0).st_mode)
 for attempt in range(12):
  with tempfile.TemporaryFile() as log:
   child=subprocess.Popen(argv,stdin=sys.stdin,stdout=log,stderr=subprocess.STDOUT)
   try:
    while child.poll() is None:
     if fifo and select.select([sys.stdin],[],[],.5)[0] and not os.read(0,1):
      child.terminate();child.wait(timeout=20);return 130
     if not fifo:time.sleep(.5)
    log.seek(0);content=log.read()
    retry=child.returncode==255 and b'kex_exchange_identification' in content and len(content)<3000
    if retry and attempt<11:time.sleep(min(2+attempt,8));continue
    sys.stdout.buffer.write(content);sys.stdout.buffer.flush();return child.returncode
   finally:
    if child.poll() is None:child.terminate()
 return 255
if __name__=='__main__':
 guarded=sys.argv[1:2]==['--guarded'];sys.exit(run(sys.argv[2:] if guarded else sys.argv[1:],guarded))
