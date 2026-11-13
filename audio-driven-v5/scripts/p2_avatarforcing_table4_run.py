"""Sequential new Table4 check; never resumes the paused DIM best10 queue."""
import fcntl
import os
from pathlib import Path
import subprocess
import time

from p2_avatarforcing_table4_generate import ROOT,WORK,UUID,read,save,sha


def run():
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=[Path(__file__),ROOT/'scripts/p2_avatarforcing_table4_generate.py',ROOT/'scripts/p2_spectre_coefficients.py',ROOT/'scripts/p2_avatarforcing_table4_score.py']
        policy=dict(files={str(p):sha(p) for p in files},generation_protocol_sha256=sha(WORK/'protocol.json'),
            spectre_weight_sha256=sha(WORK/'assets/spectre_model.tar'),node11_gpu=7,
            runtime_overlay={str(p):sha(p) for p in sorted((WORK/'runtime_py').rglob('*.py'))})
        dest=WORK/'CONTROLLER_PROTOCOL.json'
        if dest.exists():assert read(dest)==policy
        else:save(dest,policy)
        assert read(WORK/'GENERATION_SMOKE.json')['passed']
        assert read(WORK/'SPECTRE_PARITY.json')['passed']
        assert read(WORK/'GENERATION_SMOKE.json')['retained_for_formal']
        for stage,python,script,args,marker,required in [
            ('generation',ROOT/'compare/.envs/avatarforcing/bin/python',ROOT/'scripts/p2_avatarforcing_table4_generate.py',['generate'],'GENERATION_COMPLETE.json',49152),
            ('spectre',ROOT/'compare/.envs/vico_more/bin/python',ROOT/'scripts/p2_spectre_coefficients.py',['extract'],'EXTRACTION_COMPLETE.json',32768),
            ('score',ROOT/'compare/.envs/vico_more/bin/python',ROOT/'scripts/p2_avatarforcing_table4_score.py',[],'COMPLETE.json',0)]:
            if (WORK/marker).exists():continue
            while required:
                row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
                assert row[0].strip()==UUID
                free=int(row[1])
                if free>=required:break
                save(WORK/'status.json',dict(stage='waiting_capacity',next_stage=stage,free_mib=free,required_mib=required,at=time.time()))
                time.sleep(30)
            for path,h in policy['files'].items():assert sha(path)==h,path
            assert sha(WORK/'protocol.json')==policy['generation_protocol_sha256']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='7' if required else '',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                PYTHONPATH=str(WORK/'runtime_py') if stage=='generation' else '',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
            log=WORK/(stage+'_formal.log')
            with log.open('ab') as output:
                proc=subprocess.Popen([str(python),str(script),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
                save(WORK/'ACTIVE.json',dict(pid=proc.pid,stage=stage,log=str(log),at=time.time()))
                code=proc.wait()
            save(WORK/(stage+'_exit.json'),dict(returncode=code,at=time.time(),log=str(log)))
            if code or not (WORK/marker).exists():
                save(WORK/'status.json',dict(stage='failed',failed_stage=stage,returncode=code,log=str(log)))
                raise RuntimeError('Stage failed: '+str(log))
        save(WORK/'CONTROLLER_COMPLETE.json',dict(passed_execution=True,results_sha256=sha(WORK/'RESULTS.json'),
            matched_cells=read(WORK/'RESULTS.json')['matched_cells_at_published_precision'],gpu=7,
            exact_private_protocol_confirmed=False))


if __name__=='__main__':run()
