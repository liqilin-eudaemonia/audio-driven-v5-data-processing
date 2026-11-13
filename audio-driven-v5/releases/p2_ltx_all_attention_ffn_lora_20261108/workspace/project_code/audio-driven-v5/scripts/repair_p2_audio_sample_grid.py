"""CPU-only repair of one-sample resampling excess in this feature run.

Keep the original WAV and target/labels. Publish a float32 16-kHz speaker
derivative only when the existing native reader exceeds its round(d*16000)
contract by exactly one sample. Locks exclude active GPU/media jobs. A journal
repairs interrupted manifest/marker commits before releasing the job again.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_fusion import atomic,read,lock,file_sha

POLICY='speaker-resample-round-grid-v1'

def json_sha(value):
    return hashlib.sha256((json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n').encode()).hexdigest()

def normalize_speaker(run,row):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    obs=row['observation'];sid=row['sample_id'];d=obs['requested_duration_s']
    # Integer 16-kHz endpoints already satisfy the native reader exactly.
    if abs(d*16000-round(d*16000))<1e-5:return None
    source=Path(obs['speaker_audio']);info=sf.info(source)
    n=round(d*info.samplerate);expected=round(d*16000)
    actual=(n*16000+info.samplerate-1)//info.samplerate
    if actual==expected:return None
    if actual!=expected+1 or info.frames!=n or info.channels!=1:
        raise ValueError('Not a verified one-sample resampling excess')
    dest=run/'media'/sid
    if source.resolve()!= (dest/'speaker.wav').resolve():raise ValueError('Unexpected source speaker path')
    if file_sha(source)!=obs['media_hashes']['speaker_audio']:raise ValueError('Speaker source hash differs')
    if any((run/f'cache/{part}/{sid}/complete.json').exists() for part in ('input','teacher','vae','events')):
        raise ValueError('Refusing to change an observation with completed feature caches')
    with sf.SoundFile(source) as stream:wave=stream.read(n,dtype='float32',always_2d=True).mean(-1)
    common=math.gcd(info.samplerate,16000)
    original=resample_poly(wave,16000//common,info.samplerate//common).astype(np.float32)
    if len(original)!=actual:raise ValueError('Resampler length differs')
    fixed=original[:expected]
    target=dest/'speaker_16k_exact.wav';temp=dest/'speaker_16k_exact.pending.wav'
    sf.write(temp,fixed,16000,subtype='FLOAT')
    with temp.open('rb') as stream:os.fsync(stream.fileno())
    os.replace(temp,target)
    check,rate=sf.read(target,dtype='float32')
    if rate!=16000 or not np.array_equal(check,fixed):raise ValueError('Derived speaker waveform differs')
    new=copy.deepcopy(obs);new['speaker_audio']=str(target);new['media_hashes']['speaker_audio']=file_sha(target)
    audit=dict(policy=POLICY,sample_id=sid,original_observation=obs,derived_observation=new,
               source_rate=info.samplerate,source_samples=n,resampled_samples=actual,retained_samples=expected,
               removed_tail_samples=1,maximum_grid_difference_s=1/16000,raw_audio_modified=False)
    return new,audit

def repair_job(run,jid):
    run=Path(run);folder=run/'audio_grid';journal=folder/f'jobs/{jid}.json'
    with lock(run/f'locks/worker-features-{jid}.lock') as gpu_lock,lock(run/f'locks/media-job-{jid}.lock') as media_lock:
        if not gpu_lock or not media_lock:return None
        marker=read(run/f'state/media/{jid}.json')
        if not marker:return None
        if (run/f'state/features/{jid}.json').exists():return dict(done=True,repaired=0)
        path=run/f'manifests/shards/{jid}.json';tx=read(journal)
        if tx and tx.get('committed'):return dict(done=True,repaired=0)
        if not tx:
            if file_sha(path)!=marker['manifest_sha256']:raise ValueError('Unjournaled media manifest mismatch')
            rows=read(path);before=copy.deepcopy(rows);repairs=[]
            for row in rows:
                normalized=normalize_speaker(run,row)
                if normalized:
                    row['observation'],audit=normalized;repairs.append(audit)
            if not repairs:return dict(done=True,repaired=0)
            tx=dict(policy=POLICY,job_id=jid,original_marker=marker,original_rows=before,rows=rows,
                    new_manifest_sha256=json_sha(rows),repairs=repairs,utc=time.time())
            atomic(journal,tx)
        if file_sha(path) not in (tx['original_marker']['manifest_sha256'],tx['new_manifest_sha256']):
            raise ValueError('Manifest changed outside audio-grid transaction')
        for audit in tx['repairs']:
            sid=audit['sample_id'];dest=run/'media'/sid;obs=audit['derived_observation']
            if file_sha(obs['speaker_audio'])!=obs['media_hashes']['speaker_audio']:raise ValueError('Derived audio hash differs')
            value=read(dest/'COMPLETE.json')
            value['observation']=obs;value['meta']['sha256']['speaker_16k_exact.wav']=obs['media_hashes']['speaker_audio']
            value['meta']['speaker_grid_adapter']=audit
            atomic(dest/'media.json',value['meta']);atomic(dest/'COMPLETE.json',value)
        atomic(path,tx['rows'])
        atomic(run/f'state/media/{jid}.json',dict(tx['original_marker'],manifest_sha256=tx['new_manifest_sha256'],audio_grid_policy=POLICY))
        atomic(journal,dict(tx,committed=True,committed_at=time.time()))
        return dict(done=True,repaired=len(tx['repairs']))

def loop(run):
    seen={};checked=0;repaired=0
    with lock(run/'audio_grid/worker.lock') as got:
        if not got:return
        while not (run/'STOP').exists() and not (run/'COMPLETE.json').exists():
            for p in (run/'state/media').glob('*.json'):
                stamp=p.stat().st_mtime_ns
                if seen.get(p.stem)==stamp:continue
                try:
                    result=repair_job(run,p.stem)
                    if result:
                        seen[p.stem]=p.stat().st_mtime_ns;checked+=1;repaired+=result['repaired']
                        alert=run/f'alerts/audio-grid-{p.stem}.json'
                        if alert.exists():atomic(alert,dict(read(alert),resolved_at=time.time()))
                except Exception as exc:
                    atomic(run/f'alerts/audio-grid-{p.stem}.json',dict(utc=time.time(),error=repr(exc),job_id=p.stem))
                atomic(run/'audio_grid/heartbeat.json',dict(pid=os.getpid(),heartbeat=time.time(),checked_jobs=checked,repaired_samples=repaired))
            atomic(run/'audio_grid/heartbeat.json',dict(pid=os.getpid(),heartbeat=time.time(),checked_jobs=checked,repaired_samples=repaired))
            time.sleep(10)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--watchdog',action='store_true');a=p.parse_args();run=a.run.resolve()
    if not a.watchdog:loop(run);return
    with lock(run/'audio_grid/watchdog.lock') as got:
        if not got:return
        child=None
        while not (run/'STOP').exists() and not (run/'COMPLETE.json').exists():
            if child is None or child.poll() is not None:
                with (run/'audio_grid/worker.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run',str(run)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(run/'audio_grid/watchdog.json',dict(pid=os.getpid(),child_pid=child.pid,heartbeat=time.time()))
            time.sleep(10)

if __name__=='__main__':main()
