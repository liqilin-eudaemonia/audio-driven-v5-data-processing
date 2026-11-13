"""Publish light progress and run final browser acceptance after the GPU job."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/common_method_comparison_20261106'
RUN=WORK/'l2l_retrained'
OUT=ROOT/'data_previews/listen_benchmark_review/common_method_comparison_20261106'


def read(p):return json.loads(p.read_text())
def save(p,d):
    tmp=p.with_suffix(p.suffix+'.watch_tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def main():
    global RUN
    if (WORK/'ACTIVE_L2L_RUN.json').exists():RUN=Path(read(WORK/'ACTIVE_L2L_RUN.json')['path'])
    deadline=time.time()+12*3600
    last_check=0.;missing_checks=0
    while time.time()<deadline:
        status=read(RUN/'status.json') if (RUN/'status.json').exists() else {}
        count=len(list((RUN/'features').glob('*/*.json')))
        complete=(RUN/'COMPLETE.json').exists()
        progress=dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            stage='complete' if complete else status.get('stage','features'),feature_pairs=count,total_pairs=397,
            training=status,physical_gpu='node11 GPU7',complete=complete)
        if (RUN/'PAUSED_AFTER_VQ.json').exists():
            pause=read(RUN/'PAUSED_AFTER_VQ.json')
            progress.update(stage='paused_after_vq',paused_by_user=True,complete=False,gpu_released=pause['gpu_released'])
            save(OUT/'training_progress.json',progress);save(WORK/'PROGRESS.json',progress)
            return 0
        failures=[]
        for path in RUN.glob('features_*.log'):
            text=path.read_text()
            if 'Traceback' in text:failures.append(dict(log=path.name,error=text.splitlines()[-1]))
        if failures:progress.update(stage='failed',failures=failures)
        if not complete and time.time()-last_check>60:
            config=read(ROOT/'configs/p2_j_qwen_whisper_av_train32_cluster_20261104.json')
            pid=read(RUN/'LAUNCH.json')['pid']
            code="from pathlib import Path;p=Path('/proc')/str("+str(int(pid))+")/'cmdline';print('alive' if p.exists() and b'vico_l2l_retrain' in p.read_bytes() else 'missing')"
            try:
                check=subprocess.run(['ssh','-i',config['ssh_key'],'-o','BatchMode=yes','-o','ConnectTimeout=10',
                    'root@172.29.189.119',shlex.join(['python3','-c',code])],capture_output=True,text=True,timeout=20)
                if check.returncode==0:missing_checks=missing_checks+1 if check.stdout.strip()=='missing' else 0
            except subprocess.TimeoutExpired:pass
            last_check=time.time()
            if missing_checks>=2:
                progress.update(stage='failed',failure='training controller exited before completion; inspect l2l_retrained/run.log')
                save(WORK/'WATCH_FAILURE.json',progress)
        save(OUT/'training_progress.json',progress);save(WORK/'PROGRESS.json',progress)
        if missing_checks>=2:return 1
        if complete:
            # The training controller publishes after writing COMPLETE. Wait for
            # its atomic data publication, avoiding concurrent builder writes.
            if read(OUT/'data.json').get('l2l_retrained') is None:
                time.sleep(15);continue
            result=subprocess.run(['/tmp/native-fps-review-qa/bin/python',str(ROOT/'scripts/check_common_method_comparison.py')],
                cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            (WORK/'final_browser_acceptance.log').write_text(result.stdout)
            if result.returncode:
                save(WORK/'WATCH_FAILURE.json',dict(stage='final_browser_acceptance',returncode=result.returncode,log='final_browser_acceptance.log'))
                return 1
            save(WORK/'COMPLETE.json',dict(status='complete',local_video_comparisons=True,common_deca=True,
                l2l_full_retraining=True,test43=True,browser_acceptance=read(WORK/'FINAL_ACCEPTANCE.json')['passed'],
                published_results_kept_separate=True))
            return 0
        time.sleep(30)
    save(WORK/'WATCH_FAILURE.json',dict(stage='monitor_timeout',training_not_stopped=True))
    return 1


if __name__=='__main__':raise SystemExit(main())
