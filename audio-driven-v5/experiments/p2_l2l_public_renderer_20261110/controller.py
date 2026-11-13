"""Complete the authorized L2L+GPAvatar generation and scoring sequentially."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import p2_l2l_gpavatar as gp


def run():
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gp.verify()
        files=[Path(__file__),WORK/'evaluate.py',WORK/'image_metrics.py',WORK/'pilot_video.py',ROOT/'scripts/p2_l2l_gpavatar.py']
        frozen={str(f):gp.sha(f) for f in files}
        policy=dict(sources=frozen,generation_protocol_sha256=gp.sha(WORK/'protocol.json'),
            gpu=7,node='node11',resume_completed_outputs=True,automatic_training=False)
        dest=WORK/'CONTROLLER_PROTOCOL.json'
        if dest.exists():assert gp.read(dest)==policy
        else:gp.save(dest,policy)
        python=ROOT/'compare/.envs/vico_more/bin/python'
        stages=[
            ('track_smoke',ROOT/'scripts/p2_l2l_gpavatar.py',['track_smoke'],WORK/'TRACK_SMOKE_COMPLETE.json',32768),
            ('smoke',ROOT/'scripts/p2_l2l_gpavatar.py',['smoke'],WORK/'SMOKE_COMPLETE.json',32768),
            ('pilot',WORK/'pilot_video.py',[],WORK/'PILOT_VIDEO_COMPLETE.json',32768),
            ('track',ROOT/'scripts/p2_l2l_gpavatar.py',['track'],WORK/'TRACK_COMPLETE.json',32768),
            ('generate',ROOT/'scripts/p2_l2l_gpavatar.py',['generate'],WORK/'RENDER_COMPLETE.json',32768),
            ('eval_prepare',WORK/'evaluate.py',['prepare'],WORK/'emoca_eval/protocol.json',0),
            ('eval_smoke',WORK/'evaluate.py',['smoke'],WORK/'emoca_eval/MEDIA_PARITY.json',32768),
            ('eval_extract',WORK/'evaluate.py',['extract'],WORK/'emoca_eval/EXTRACTION_COMPLETE.json',32768),
            ('eval_score',WORK/'evaluate.py',['score'],WORK/'emoca_eval/COMPLETE.json',0),
        ]
        for name in gp.NAMES:
            d=WORK/'image_eval'/name
            stages.append(('images_'+name,WORK/'image_metrics.py',['--work',str(d)],d/'COMPLETE.json',32768))
        for stage,script,args,marker,minimum in stages:
            if marker.exists():continue
            for f,h in frozen.items():assert gp.sha(f)==h,f
            assert gp.sha(WORK/'protocol.json')==policy['generation_protocol_sha256']
            if stage in ['pilot','generate']:
                assert gp.read(WORK/'SMOKE_COMPLETE.json')['passed']
                assert gp.read(WORK/'RENDER_CONTROL_PARITY.json')['passed']
                assert gp.read(WORK/'VISUAL_ACCEPTED.json')['passed']
                assert gp.read(WORK/'VISUAL_ACCEPTED.json')['protocol_sha256']==gp.sha(WORK/'protocol.json')
            if minimum:
                while True:
                    info=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
                    assert info[0].strip()==gp.UUID
                    if int(info[1])>=minimum:break
                    gp.save(WORK/'status.json',dict(stage='waiting_capacity',next_stage=stage,free_mib=int(info[1]),required_mib=minimum,at=time.time()))
                    time.sleep(30)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='7' if minimum else '',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                PYTHONUNBUFFERED='1',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='9.0',
                TORCH_EXTENSIONS_DIR=str(WORK/'torch_extensions'),TOKENIZERS_PARALLELISM='false')
            with (WORK/(stage+'.log')).open('ab') as output:
                proc=subprocess.Popen([str(python),str(script),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
                gp.save(WORK/'ACTIVE.json',dict(stage=stage,pid=proc.pid,log=str(WORK/(stage+'.log')),at=time.time()))
                code=proc.wait()
            gp.save(WORK/(stage+'_exit.json'),dict(returncode=code,at=time.time()))
            if code or not marker.exists():
                gp.save(WORK/'status.json',dict(stage='failed',failed_stage=stage,returncode=code))
                raise RuntimeError('Stage failed: '+stage)
        paths=[WORK/'RENDER_COMPLETE.json',WORK/'emoca_eval/COMPLETE.json',*[WORK/'image_eval'/n/'COMPLETE.json' for n in gp.NAMES]]
        gp.save(WORK/'PIPELINE_COMPLETE.json',dict(passed=True,outputs=172,pairs=43,
            completion_receipts={str(p):gp.sha(p) for p in paths},protocol_sha256=gp.sha(WORK/'protocol.json'),
            label='Public L2L4 person checkpoints + public GPAvatar, not original L2L renderer'))
        gp.save(WORK/'status.json',dict(stage='complete',outputs=172,pairs=43,methods=4,at=time.time()))


if __name__=='__main__':run()
