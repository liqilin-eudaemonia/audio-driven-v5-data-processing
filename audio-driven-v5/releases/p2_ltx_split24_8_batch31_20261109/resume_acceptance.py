"""Resume acceptance after pre-execution SSH failure; never remigrate live outputs."""
import fcntl,json,time,traceback
import control as C

def main():
    with(C.WORK/'controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);C.source_ready()
        C.write(C.WORK/'CONTROLLER.json',C.record())
        migration={a:C.read(C.path(a,True)/'MIGRATION.json')for a in C.ARMS}
        assert migration['full']['step']==6000 and migration['noevent']['step']==110
        attempts=0;begin=time.time()
        while time.time()-begin<3600:
            if not C.alive(C.read(C.WORK/'noevent_LAUNCH_CONTROLLER.json')):
                raise RuntimeError('NoEvent launcher stopped; do not restart it silently')
            if not C.alive(C.read(C.WORK/'full_LAUNCH_CONTROLLER.json')):
                out=C.path('full',True);logs=sorted(out.glob('nodes_*/node11.log'),key=lambda p:p.stat().st_mtime)
                if not logs or 'kex_exchange_identification'not in logs[-1].read_text()or list((out/'rank_logs').glob('worker_*.json')):
                    raise RuntimeError('Full failure is not a proven pre-worker SSH handshake failure')
                if attempts>=8:raise RuntimeError('Full SSH launch retries exhausted')
                time.sleep(5*(attempts+1));p=C.spawn('full');attempts+=1
                C.emit('retrying_full8_pre_worker_ssh',attempt=attempts,launcher_pid=p.pid)
                time.sleep(5);continue
            nr=C.legacy.rows_after(C.path('noevent',True),migration['noevent']['step'])
            fr=C.legacy.rows_after(C.path('full',True),migration['full']['step'])
            C.legacy.check_rows(nr,False);C.legacy.check_rows(fr,True)
            if len(nr)>=3 and len(fr)>=3:
                ns=C.probe('noevent');fs=C.probe('full')
                if ns['healthy']and fs['healthy']:
                    canaries={a:C.read(C.path(a,True)/'LONGEST_UNEVEN_CHECK.json')for a in C.ARMS}
                    assert all(v['success']and v['global_batch']==31 for v in canaries.values())
                    result=dict(success=True,at=time.time(),global_batch=31,noevent_step=nr[-1]['step'],full_step=fr[-1]['step'],
                        noevent_cluster=ns,full_cluster=fs,noevent_migration=migration['noevent'],full_migration=migration['full'],
                        canaries=canaries,noevent_seconds=[r['seconds']for r in nr],full_seconds=[r['seconds']for r in fr],
                        recovered_pre_worker_ssh_failure=True,noevent_never_restarted=True)
                    C.write(C.WORK/'SPLIT_ACCEPTED.json',result)
                    C.write(C.WORK/'SSH_FAILURE_RECOVERED.json',dict(at=time.time(),resolved_error='ERROR_927994.json',extra_attempts=attempts,training_code_unchanged=True))
                    C.emit('both_running_accepted',noevent_step=nr[-1]['step'],full_step=fr[-1]['step']);break
            time.sleep(10)
        else:raise TimeoutError('Acceptance after SSH recovery')
        # The independent read-only five-minute observer now checks both trainers.

if __name__=='__main__':
    try:main()
    except Exception as exc:
        C.write(C.WORK/'RECOVERY_ERROR.json',dict(at=time.time(),error=repr(exc),traceback=traceback.format_exc()));raise
