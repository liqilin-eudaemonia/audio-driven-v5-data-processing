"""Audit the original completed window and real continuing training before acceptance."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import common as C
import guardian as G


def identity(snapshot):
    return {(n['node'], p['pid'], p['start_ticks'])
            for n in snapshot['nodes'] for p in n['workers']}


def snapshot_ok(snapshot, expected):
    assert C.observed(snapshot) and snapshot['healthy31']
    assert snapshot['work'] == str(C.OUT)
    assert snapshot['reserved_gpu'] == 'node11:0'
    assert sorted(p['rank'] for n in snapshot['nodes'] for p in n['workers']) == list(range(31))
    assert {n['node'] for n in snapshot['nodes']} == set(expected)
    for node in snapshot['nodes']:
        workers = node['workers']
        assert sorted(p['gpu'] for p in workers) == expected[node['node']]
        assert sorted(p['local'] for p in workers) == list(range(len(workers)))
        for p in workers:
            assert p['pid'] > 0 and p['start_ticks'].isdigit() and p['gpu_context_present']
            assert p['hca'] == '=mlx5_0:1'
            assert p['memory_cap_gib'] == ('108' if node['node'] == 'node11' else '134')


def checkpoint_ok(marker):
    path = marker.with_suffix('').with_suffix('.pt')
    receipt = C.read(WORK / 'checkpoints' / (path.stem + '.json'))
    value = C.read(marker)
    stat = path.stat()
    assert receipt['success'] and receipt['path'] == str(path)
    assert receipt['step'] == value['step'] and receipt['sha256'] == value['sha256']
    assert receipt['bytes'] == stat.st_size and receipt['mtime_ns'] == stat.st_mtime_ns
    assert receipt['inode'] == stat.st_ino and receipt['rank_rng'] == 31
    assert receipt['model_and_adam_finite'] and receipt['cpu_rng_restoration_checked']
    assert receipt['cuda_initialized'] is False
    return receipt


def main():
    protocol = C.read(WORK / 'protocol.json')
    assert protocol['start'] == 1794151525 and protocol['deadline'] == 1794183925
    assert protocol['deadline'] - protocol['start'] == 9 * 3600
    assert time.time() >= protocol['deadline']
    finish = C.read(WORK / 'WINDOW_FINISHED.json')
    assert finish['start'] == protocol['start'] and finish['deadline'] == protocol['deadline']
    assert finish['at'] >= protocol['deadline'] and finish['training_stopped'] is False
    cluster = C.read(C.OUT / 'cluster.json')
    expected = {n['name']: n['gpus'] for n in cluster['nodes']}
    assert set(expected) == {'node8', 'node12', 'node4', 'node11'}
    assert expected['node11'] == list(range(1, 8))
    assert all(g == list(range(8)) for n, g in expected.items() if n != 'node11')

    first = C.read(WORK / 'FINAL_FIRST_PROBE.json')
    assert first['at'] >= protocol['deadline']
    snapshot_ok(first['cluster'], expected)
    launcher = C.read(WORK / 'CURRENT_LAUNCHER.json')
    until = time.time() + 50
    while time.time() < until:
        assert C.alive(launcher), 'Training launcher is no longer live'
        health = C.health()
        assert not health['flags'], health
        if health['step'] >= first['health']['step'] + 3:
            break
        time.sleep(5)
    current = C.probe()
    health = C.health()
    snapshot_ok(current, expected)
    assert health['step'] >= first['health']['step'] + 3 and not health['flags']
    assert identity(current) == identity(first['cluster'])
    assert identity(current) == identity(C.read(WORK / 'agent_reports' / 'recovery_3040.json')['cluster'])
    assert C.alive(launcher) and not C.stop_intent()
    contract = C.contract(True)
    node11 = next(n for n in cluster['nodes'] if n['name'] == 'node11')
    own = [p['pid'] for n in current['nodes'] if n['node'] == 'node11' for p in n['workers']]
    code = "import subprocess,json\n" + "own=" + repr(own) + "\n" + """def query(fields):
 return subprocess.check_output(['nvidia-smi',*fields,'--format=csv,noheader,nounits'],text=True).splitlines()
uuid0=next(line.split(',')[1].strip()for line in query(['--query-gpu=index,uuid'])if line.split(',')[0].strip()=='0')
apps=[line.split(',')for line in query(['--query-compute-apps=pid,gpu_uuid'])]
on_reserved=[int(a[0].strip())for a in apps if a[0].strip().isdigit()and int(a[0].strip())in own and a[1].strip()==uuid0]
print(json.dumps(dict(gpu=0,uuid=uuid0,training_pids_on_reserved=on_reserved)))
"""
    reserved = subprocess.run(C.transport(cluster, node11, [cluster['python'], '-c', code]),
                              capture_output=True, text=True, timeout=30, check=True)
    reserved = json.loads(reserved.stdout)
    assert not reserved['training_pids_on_reserved']

    reports = []
    for slot in range(10):
        path = WORK / 'reports' / f'hour_{slot:02d}.json'
        r = C.read(path)
        assert r['slot'] == slot and r['at'] >= protocol['start'] + slot * 3600
        if slot < 9:
            assert r['at'] < protocol['start'] + (slot + 1) * 3600
        snapshot_ok(r['cluster'], expected)
        assert not r['health']['flags'] and r['contract']['success']
        assert r['contract']['data_checked'] and r['contract']['source_files'] == 282
        assert r['contract']['config_sha256'] == protocol['config_sha256']
        assert r['contract']['cluster_sha256'] == protocol['cluster_sha256']
        reports.append(dict(slot=slot, at=r['at'], step=r['health']['step'], path=str(path)))

    rows = [json.loads(line) for line in (C.OUT / 'metrics.jsonl').read_text().splitlines()]
    steps = [r['step'] for r in rows]
    assert steps == list(range(min(steps), max(steps) + 1)), 'Missing or repeated metric steps'
    for r in rows:
        keys = ['av', 'av_video', 'av_audio', 'event', 'feature_video', 'feature_audio', 'weighted_total']
        assert all(math.isfinite(r[k]) for k in keys), r['step']
        assert all(math.isfinite(v) for v in r['grad_norm'].values()), r['step']
        want = r['av'] + .05 * r['event'] + .025 * (r['feature_video'] + r['feature_audio'])
        assert abs(r['weighted_total'] - want) <= 1e-8, r['step']

    latest_path, latest = G.checked_latest()
    assert latest['step'] >= 4000
    checkpoints = [checkpoint_ok(m) for m in sorted(C.OUT.glob('step_*.complete.json'))
                   if m.stat().st_mtime >= protocol['start']]
    recoveries = []
    for incident in sorted(WORK.glob('incident_*')):
        restart = C.read(incident / 'RESTART.json')
        accepted_path = incident / 'RESUMED_ACCEPTED.json'
        accepted = C.read(accepted_path)
        used = C.read(incident / 'CHECKPOINT_USED.json')
        assert used['path'] == restart['checkpoint'] and used['step'] == restart['step']
        assert checkpoint_ok(Path(used['path']).with_suffix('.complete.json')) == used
        assert accepted['restarted'] and accepted['graceful']
        assert accepted['step'] == restart['step'] and accepted['verified_step'] >= restart['step'] + 3
        assert protocol['start'] <= restart['at'] <= accepted_path.stat().st_mtime <= protocol['deadline']
        termination = C.read(incident / 'termination.json')
        assert all(not n['results'] for n in termination), 'Inspect any forced worker signals'
        recoveries.append(dict(incident=str(incident), saved_step=restart['step'],
                               checkpoint=used, verified_step=accepted['verified_step'],
                               resumed_at=accepted_path.stat().st_mtime,
                               reason=restart['reason'], graceful=True, forced_worker_signals=0))
    assert [r['saved_step'] for r in recoveries] == [1909, 3040]
    assert not C.alive(C.read(WORK / 'CURRENT_REPAIR.json'))
    assert not any((WORK / name).exists() for name in ['NEEDS_AGENT.json', 'RECOVERY_ERROR.json', 'GUARDIAN_ERROR.json'])

    hashes = C.read(WORK / 'OBSERVER_SOURCE.json')
    assert all(C.sha(WORK / name) == value for name, value in hashes.items())
    release = C.read(C.ROOT / 'releases/p2_full_lora_guard9h_20261108/presence_guard_revision_20261108/MANIFEST.json')
    assert all(C.sha(C.ROOT / name) == value for name, value in release['files'].items())
    assert all(C.sha(WORK / Path(name).name) == value for name, value in release['files'].items())
    tests = ET.parse(WORK / 'tests.xml').getroot().find('testsuite')
    assert int(tests.attrib['tests']) == 19
    assert all(int(tests.attrib[k]) == 0 for k in ['failures', 'errors', 'skipped'])
    assert not C.alive(C.read(WORK / 'GUARDIAN.json'))
    assert not C.alive(C.read(WORK / 'SUPERVISOR.json'))
    exited = C.read(WORK / 'SUPERVISOR_EXITED.json')
    assert exited['at'] >= protocol['deadline'] and exited['training_stopped'] is False
    speed = G.speed_baseline()()
    assert speed['ratio'] is not None and speed['ratio'] < 1.5
    checks = [json.loads(line) for line in (WORK / 'agent_checks.jsonl').read_text().splitlines()]
    final = dict(success=True, at=time.time(), original_protocol=protocol,
                 elapsed_hours=(time.time() - protocol['start']) / 3600,
                 window_finished=finish, hourly_reports=reports,
                 final_cluster=current, final_health=health,
                 post_window_first_probe=str(WORK / 'FINAL_FIRST_PROBE.json'),
                 post_window_fresh_updates=health['step'] - first['health']['step'],
                 training_continues=True, training_launcher=launcher,
                 final_step=max(steps), recent50_seconds=statistics.mean(r['seconds'] for r in rows[-50:]),
                 duration_adjusted_speed=speed, contract=contract,
                 all_metrics_checked=len(rows), metrics_contiguous=True,
                 all_losses_and_gradient_norms_finite=True, all_loss_formulas_correct=True,
                 latest_checkpoint=latest, window_checkpoint_receipts=checkpoints,
                 recoveries=recoveries, performance_policy=str(WORK / 'SHARED_COMPUTE_POLICY.md'),
                 reserved_gpu='node11:0', reserved_gpu_actual_probe=reserved, observer_source=hashes,
                 auditor_source_sha256=C.sha(Path(__file__)), current_observer_matches_tested_backup=True,
                 safeguard_tests_passed=19, backup_files_verified=len(release['files']),
                 live_agent_check_count=len(checks), observers_exited_as_scheduled=True,
                 requirement_checks=dict(original_nine_hour_window=True, hourly_four_node_checks=True,
                     completed_stateful_recoveries=True, current_speed_stable=True,
                     frozen_code_config_data=True, checkpoints_verified=True,
                     reserved_gpu_preserved=True, no_pending_fault=True,
                     training_progresses_after_window=True))
    C.atomic_json(WORK / 'FINAL_ACCEPTED.json', final)
    print(json.dumps(dict(success=True, step=final['final_step'], recent50_seconds=final['recent50_seconds'],
                         elapsed_hours=final['elapsed_hours'], hourly_reports=len(reports),
                         verified_checkpoints=[r['step'] for r in checkpoints],
                         recoveries=[r['saved_step'] for r in recoveries],
                         post_window_fresh_updates=final['post_window_fresh_updates'],
                         training_continues=True)))


if __name__ == '__main__':
    main()
