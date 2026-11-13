"""Independent post-publication validator for the already-running coordinator."""
import os,sys,time,subprocess,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
def main():
 atomic_json(OUT/'validation_status.json',dict(phase='waiting_publication'))
 while True:
  status=read(OUT/'status.json')
  if status['phase']=='failed':raise RuntimeError(status['error'])
  if status['phase']=='complete':break
  time.sleep(5)
 atomic_json(OUT/'validation_status.json',dict(phase='validating'))
 view=ROOT/'experiments/p2_av_review_20261027'
 env={**os.environ,'LD_LIBRARY_PATH':'/tmp/v8_browser_libs/extracted/usr/lib/x86_64-linux-gnu','FONTCONFIG_FILE':'/tmp/p2_review_fonts.conf'}
 for script in ['browser_english_check.py','browser_full_guided_check.py','validate_full_media.py']:
  python=str(PY) if script=='validate_full_media.py' else '/tmp/p2-review-qa/bin/python'
  with (OUT/('post_'+script+'.log')).open('a') as log:subprocess.run([python,str(view/script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 atomic_json(OUT/'validation_status.json',dict(phase='complete',english_outputs=384,total_review_outputs=768,metric_modes=2,passed=True))
if __name__=='__main__':
 if len(sys.argv)>1:
  with (OUT/'validate_published.log').open('a') as log:p=subprocess.Popen([str(PY),str(__file__)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
  print(p.pid)
 else:
  try:main()
  except Exception as error:atomic_json(OUT/'validation_status.json',dict(phase='failed',error=str(error)));traceback.print_exc();raise
