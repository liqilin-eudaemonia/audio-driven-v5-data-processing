"""Audited one-time repair of the nine failures from the interrupted V8 run.

Run from any directory with the project Python. Never starts the full queue.
Original errors, code and contract are retained under this experiment.
"""
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_listen_v8_full as full
from p3_listen.common import read,write,sha

REVISION='v8-full-2-error-repair-20261023'
CHANGED={'scripts/listen_policy_v8.py','scripts/plan_listen_v8.py','scripts/run_listen_v8_full.py'}


def main():
    out=full.OUT
    assert out==ROOT/'data_corpora/seamless_listen_v8_full'
    with (out/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read(HERE/'before/data_corpora/seamless_listen_v8_full/runtime.json')
        for pid in [previous['supervisor_pid'],*previous['pids']]:
            cmd=Path(f'/proc/{pid}/cmdline')
            if cmd.exists() and b'run_listen_v8_full.py' in cmd.read_bytes():
                raise RuntimeError(f'Original V8 process still alive: {pid}')
        regression=read(HERE/'accepted_sample_regression.json')
        assert len(regression)==24 and all(r['unchanged'] for r in regression)
        failed=read(HERE/'failed_pair_validation.json')
        assert len(failed)==9
        assert sum(r['status']=='planned' for r in failed)==3
        assert sum(r['status']=='source_missing_video_stream' for r in failed)==6
        contract=read(out/'contract.json')
        if contract['revision']!=REVISION:
            parent=HERE/'before/data_corpora/seamless_listen_v8_full/contract.json'
            assert sha(out/'contract.json')==sha(parent)
            changed={name for name,h in contract['code_sha256'].items() if sha(ROOT/name)!=h}
            assert changed==CHANGED,changed
            for name in CHANGED:
                assert sha(HERE/'before'/name)==contract['code_sha256'][name]
            for name,h in contract['input_sha256'].items():assert sha(out/name)==h
            assert sha(full.FEEDBACK)==contract['feedback_source_sha256']
            assert sha(ROOT/'data_reviews/listener_noise_rejections.json')==contract['legacy_feedback_sha256']
            reused=[]
            for path in sorted((out/'results').glob('*.json')):
                result=read(path);assert result['complete']
                assert all(a['end']<=b['start']+1e-6 for a,b in zip(result['clips'],result['clips'][1:]))
                reused.append(dict(pair=result['pair'],sha256=sha(path)))
            write(HERE/'retained_parent_results.json',dict(parent_contract_sha256=sha(parent),rows=reused))
            contract.update(revision=REVISION,utc=full.now(),repair=dict(
                parent_contract=str(parent.relative_to(ROOT)),parent_contract_sha256=sha(parent),
                retained_results_manifest=str((HERE/'retained_parent_results.json').relative_to(ROOT)),
                retained_results_manifest_sha256=sha(HERE/'retained_parent_results.json'),
                changed_files=sorted(CHANGED),
                compatibility='Reject only feedback heads that violate the existing non-overlap invariant; '
                    'classify verified absent video streams as invalid sources. QC and policy thresholds unchanged.',
                accepted_sample_regression_sha256=sha(HERE/'accepted_sample_regression.json')))
            for name in CHANGED:contract['code_sha256'][name]=sha(ROOT/name)
            write(out/'contract.json',contract)
        full.verify()
        write(out/'runtime.json',dict(state='repairing_failed_pairs',utc=full.now(),
            repair_pid=os.getpid(),full_queue_running=False,training_started=False))
        full.configure_normalization()
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        lookup={r['pair']:r for r in read(out/'catalog.json')['rows']}
        rejects=read(out/'human_rejections.json')['rows'];positives=read(out/'human_retained.json')['rows']
        ast=panns=None;repaired=[]
        for failure in failed:
            pair=failure['pair'];item=lookup[pair];destination=out/'results'/(pair+'.json')
            if destination.exists():
                existing=read(destination)
                assert existing.get('code_revision')==REVISION and existing['complete']
                repaired.append(dict(pair=pair,status='already_repaired',clips=len(existing['clips'])))
                continue
            try:
                audit,norm,_=full.prepare(item)
                if audit['clips'] and ast is None:
                    ast=full.legacy.EventModel(full.legacy.MODEL,batch_size=contract['ast_batch'])
                    panns=full.legacy.PannsModel(batch_size=contract['panns_batch'])
                clips=[full.screen(row,norm,ast,panns,rejects,positives) for row in audit['clips']]
                write(destination,dict(pair=pair,split=item['split'],group_id=item['group_id'],complete=True,
                    utc=full.now(),clips=clips,rejected=audit['unresolved'],source_ok=audit['ok'],
                    source_seconds=audit.get('source_seconds'),training_ready=False,code_revision=REVISION))
                error=out/'errors'/(pair+'.json')
                if error.exists():
                    assert (HERE/'original_errors'/error.name).exists()
                    error.unlink()
                result=dict(pair=pair,status='screened' if audit['ok'] else 'source_invalid',clips=len(clips))
                repaired.append(result);print(json.dumps(result),flush=True)
            except Exception as exc:
                write(out/'errors'/(pair+'.json'),dict(pair=pair,utc=full.now(),error=repr(exc),traceback=traceback.format_exc()))
                write(out/'runtime.json',dict(state='interrupted',utc=full.now(),repair_state='failed',
                    full_queue_running=False,training_started=False))
                raise
        summary=full.report(export=False)
        retained=read(HERE/'retained_parent_results.json')['rows']
        assert all(sha(out/'results'/(r['pair']+'.json'))==r['sha256'] for r in retained)
        write(HERE/'repair_result.json',dict(utc=full.now(),revision=REVISION,repaired=repaired,
            retained_parent_results_unchanged=len(retained),summary=summary))
        write(out/'runtime.json',dict(state='interrupted',utc=full.now(),repair_state='completed',
            full_queue_running=False,last_original_activity_utc='2026-10-22T13:34:23Z',
            interruption_cause='unknown; original supervisor and workers absent; no supervisor exit record',
            summary=summary,training_started=False))
        print(json.dumps(dict(repair_complete=True,complete_pairs=summary['complete_pairs'],
            error_pairs=summary['error_pairs'],pending_pairs=summary['pending_pairs'])),flush=True)


if __name__=='__main__':main()
