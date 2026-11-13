"""Read-only hourly monitoring; never signal trainers or foreign processes."""
from pathlib import Path
import fcntl,json,statistics,time,traceback
import control as C

def sample():
    report=dict(at=time.time(),interval_seconds=3600,state=C.read(C.WORK/'STATE.json'),arms={})
    if not(C.WORK/'SPLIT_ACCEPTED.json').exists():
        report['waiting_for_acceptance']=True
        report['controller_alive']=C.alive(C.read(C.WORK/'CONTROLLER.json'))
        report['controller_errors']=[str(p)for p in C.WORK.glob('ERROR_*.json')]
        return report
    for arm in C.ARMS:
        out=C.path(arm,True);status=C.read(out/'status.json');base=C.read(out/'MIGRATION.json')['step']
        rows=C.legacy.rows_after(out,base);recent=rows[-20:];flags=[]
        try:C.legacy.check_rows(recent,arm=='full')
        except Exception as exc:flags.append(repr(exc))
        saving=status['step']%500==0 and not(out/f"step_{status['step']:06d}.complete.json").exists()
        age=time.time()-(out/'metrics.jsonl').stat().st_mtime
        mean=statistics.mean(r['seconds']for r in recent)if recent else None
        if not status['complete']and age>max(1800 if saving else 600,10*(mean or 0)):flags.append('progress_stale')
        placement=None
        if not status['complete']:
            placement=C.probe(arm)
            if not placement['healthy']:flags.append('placement_observation_not_healthy')
        report['arms'][arm]=dict(status=status,mean_recent_seconds=mean,recent_updates=len(recent),
                                metrics_age_seconds=age,saving=saving,flags=flags,placement=placement)
    return report

def main():
    with(C.WORK/'watch5m.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        C.write(C.WORK/'WATCH5M_CONTROLLER.json',dict(**C.record(),interval_seconds=3600,read_only=True))
        while not(C.WORK/'STOP_WATCH5M').exists():
            time.sleep(3600)
            try:report=sample()
            except Exception as exc:report=dict(at=time.time(),observation_error=repr(exc),traceback=traceback.format_exc(),training_not_modified=True)
            C.write(C.WORK/'WATCH5M_LATEST.json',report)
            with(C.WORK/'watch5m.jsonl').open('a')as f:f.write(json.dumps(report)+'\n')
            if report.get('arms')and all(a['status']['complete']for a in report['arms'].values()):return

if __name__=='__main__':main()
