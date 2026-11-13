"""Committed J source adaptation and recoverable append-only annotation queue."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import uuid


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)


def digest(data):return hashlib.sha256(data).hexdigest()


def acquire(path,blocking=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+')
    try:fcntl.flock(f,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:f.close();return None
    return f


@contextmanager
def locked(path):
    f=acquire(path)
    try:yield
    finally:f.close()


def result_path(work,branch,sid):
    return Path(work)/branch/'results'/digest(sid.encode())[:2]/f'{sid}.json'


def ensure_decode_fd_capacity(workers,slots):
    """Each concurrent decoder may hold every slot FD while acquiring a lock."""
    required=workers*slots+256
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft==resource.RLIM_INFINITY:return {'soft':soft,'hard':hard,'required':required}
    target=max(soft,8192,required)
    if hard!=resource.RLIM_INFINITY:target=min(target,hard)
    if target<required:raise RuntimeError(f'File descriptor hard limit {hard} is below decode requirement {required}')
    if target>soft:resource.setrlimit(resource.RLIMIT_NOFILE,(target,hard))
    return {'soft':target,'hard':hard,'required':required}


def bounded_claim_groups(queue,worker,work,limit,stop):
    """Bound preparation batches without releasing an unfinished source claim.

    A source may span many groups. Only its final group may write a completion
    receipt; callers still verify every source output before doing so. Closing
    the iterator releases outstanding claims for ordinary checkpoint recovery.
    Close it only after the prefetch reader has stopped using its groups.
    """
    if limit < 1:raise ValueError('Group limit must be positive')
    work=Path(work);pending=None;owned={}
    try:
        while not stop.is_set():
            owned={k:h for k,h in owned.items() if not h.closed}
            claims=[];finalize=[];rows=[]
            while len(rows)<limit and not stop.is_set():
                if pending is None:
                    found=queue.claim(worker)
                    if found is None:break
                    job_id,handle=found;owned[job_id]=handle
                    job=read(work/'jobs'/f'{job_id}.json')
                    remaining=[r for r in job['rows']
                               if not result_path(work,queue.branch,r['sample_id']).exists()]
                    pending=((job_id,handle,job),remaining)
                claim,remaining=pending
                claims.append(claim)
                take=min(limit-len(rows),len(remaining))
                rows.extend(remaining[:take]);remaining=remaining[take:]
                if remaining:pending=(claim,remaining)
                else:finalize.append(claim);pending=None
            yield {'claims':claims,'finalize_claims':finalize,'rows':rows}
    finally:
        for handle in owned.values():handle.close()


def adapt_clip(clip,item,result_path_,result_hash,index):
    from listen_labeling.seamless import integer_crop
    sid=clip['sample_id']
    if Path(sid).name!=sid or sid in ('.','..'):raise ValueError('Unsafe sample ID')
    if clip['pair']!=item['pair'] or clip['split']!=item['split']:raise ValueError('Source/split mismatch')
    if clip['listener_id']==clip['speaker_id'] or clip['listener_index']==clip['speaker_index']:
        raise ValueError('Conflicting roles')
    source=clip['sources']['listener'];audio=clip['normalized_audio']['listener']
    if Path(source['member']).stem!=clip['listener_id'] or audio['contract']['source']!=source:
        raise ValueError('Listener audio/video provenance mismatch')
    start,end=clip['start'],clip['end']
    if not (math.isfinite(start) and math.isfinite(end) and 0<=start<end and
            abs(end-start-clip['duration_s'])<2e-6):raise ValueError('Invalid clip timing')
    geometry=clip['face_geometry']['listener']
    if geometry.get('rotation_degrees',0)!=0:raise ValueError('Nonzero source rotation')
    crop=integer_crop(clip.get('crop_proposals',{}).get('listener'),geometry['width'],geometry['height'])
    rate=audio['sample_rate']
    if not isinstance(rate,int) or rate<=0:raise ValueError('Invalid sample rate')
    duration=(round(end*rate)-round(start*rate))/rate
    if not 0<duration<=32:raise ValueError('Clip outside native256-frame budget')
    compact=dict(sample_id=sid,pair=item['pair'],split=clip['split'],group_id=clip.get('group_id'),
        listener_id=clip['listener_id'],speaker_id=clip['speaker_id'],listener_index=clip['listener_index'],
        source=source,source_start_s=start,source_end_s=end,duration_s=duration,
        normalized_audio={k:audio[k] for k in ('path','sha256','sample_rate')},video_transform=crop,
        crop_proposal=clip['crop_proposals']['listener'],boundary_activity=clip.get('boundary_activity',{}),
        boundary_review_flags=clip.get('boundary_review_flags',[]),training_admitted=False,
        source_catalog_status=item.get('source_catalog_status'),source_result=str(result_path_),
        source_result_sha256=result_hash,source_clip_index=index,
        admission_pool=clip.get('admission_pool'),audio_pool=clip['audio_final']['pool'])
    return {'sample_id':sid,'source':compact}


def committed_job(upstream,item,contract):
    directory=Path(upstream)/'results'/item['pair']
    receipt=read(directory/'complete.json');raw=(directory/'result.json').read_bytes()
    fingerprint=digest(raw)
    if receipt['contract']!=contract or receipt['result_sha256']!=fingerprint:
        raise ValueError('J completion contract/checksum mismatch')
    result=json.loads(raw)
    if not result.get('complete') or result['cluster_contract']!=contract or result['pair']!=item['pair']:
        raise ValueError('Uncommitted J result')
    rows=[];excluded=[];seen=set()
    for i,c in enumerate(result['clips']):
        if c['sample_id'] in seen:raise ValueError('Duplicate terminal clip')
        seen.add(c['sample_id'])
        if c.get('audio_final',{}).get('pool')!='keep_candidate':
            excluded.append({'sample_id':c['sample_id'],'reason':'audio_pool','pool':c.get('audio_final',{}).get('pool')});continue
        try:rows.append(adapt_clip(c,item,directory/'result.json',fingerprint,i))
        except (ValueError,KeyError,TypeError) as e:
            excluded.append({'sample_id':c['sample_id'],'reason':'input_invalid','error':str(e)})
    return {'job_id':item['pair'],'source_result_sha256':fingerprint,'source_contract':contract,
            'rows':rows,'excluded':excluded,'terminal_clips':len(result['clips']),'training_admitted':False}


class JobQueue:
    def __init__(self,work,branch):
        self.work=Path(work);self.branch=branch;self.folder=self.work/'queues'/branch
        self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'state.json'
        with locked(self.folder/'state.lock'):
            if not self.path.exists():write(self.path,{'offset':0,'inflight':{},'completed':0,'completed_clips':0,'failed':{},'attempts':{}})

    def done_path(self,job):return self.work/self.branch/'completed'/f'{job}.json'

    def claim_failed(self, job, worker):
        """Explicitly assign a failed source to a repaired worker under its claim lock."""
        with locked(self.folder/'state.lock'):
            state=read(self.path)
            if self.done_path(job).exists():return None
            if job not in state['failed']:
                raise ValueError('Recovery source is not in failed queue: '+job)
            handle=acquire(self.folder/'claims'/f'{job}.lock',False)
            if handle is None:raise RuntimeError('Recovery source already owned: '+job)
            try:
                write(self.folder/'recoveries'/f'{job}-{time.time_ns()}.json',dict(
                    job=job,worker=worker,at=time.time(),previous_failure=state['failed'][job],
                    previous_attempts=state['attempts'].get(job,0)))
                del state['failed'][job];state['attempts'].pop(job,None)
                state['inflight'][job]={'worker':worker,'pid':os.getpid(),'at':time.time()}
                write(self.path,state)
                return job,handle
            except BaseException:
                handle.close();raise

    def claim(self,worker,eligible=None):
        with locked(self.folder/'state.lock'):
            state=read(self.path)
            for job in list(state['inflight']):
                handle=acquire(self.folder/'claims'/f'{job}.lock',False)
                if handle is None:continue
                if self.done_path(job).exists():
                    del state['inflight'][job];state['completed']+=1
                    state['completed_clips']=state.get('completed_clips',0)+read(self.done_path(job)).get('clips',0)
                    handle.close();continue
                if eligible is not None and not eligible(job):
                    handle.close();continue
                state['inflight'][job]={'worker':worker,'pid':os.getpid(),'at':time.time()};write(self.path,state)
                return job,handle
            for job in list(state.get('deferred',{})):
                if self.done_path(job).exists() or job in state['failed']:
                    del state['deferred'][job];continue
                if eligible is not None and not eligible(job):continue
                handle=acquire(self.folder/'claims'/f'{job}.lock',False)
                if handle is None:continue
                del state['deferred'][job]
                state['inflight'][job]={'worker':worker,'pid':os.getpid(),'at':time.time()};write(self.path,state)
                return job,handle
            journal=self.work/'ready.jsonl'
            if not journal.exists():write(self.path,state);return None
            with journal.open() as f:
                f.seek(state['offset'])
                while True:
                    line=f.readline()
                    if not line or not line.endswith('\n'):break
                    state['offset']=f.tell();job=json.loads(line)['job_id']
                    if job in state['failed'] or job in state['inflight'] or job in state.get('deferred',{}) or self.done_path(job).exists():continue
                    if eligible is not None and not eligible(job):
                        state.setdefault('deferred',{})[job]={'at':time.time()};continue
                    handle=acquire(self.folder/'claims'/f'{job}.lock',False)
                    if handle is None:continue
                    state['inflight'][job]={'worker':worker,'pid':os.getpid(),'at':time.time()};write(self.path,state)
                    return job,handle
            write(self.path,state)
        return None

    def defer(self,job,reason='awaiting_detection'):
        """Return a claimed job to the deferred set without a completion receipt."""
        with locked(self.folder/'state.lock'):
            state=read(self.path)
            state['inflight'].pop(job,None)
            state.setdefault('deferred',{})[job]={'at':time.time(),'reason':reason}
            write(self.path,state)

    def finish(self,job,receipt=None,error=None):
        with locked(self.folder/'state.lock'):
            state=read(self.path)
            if error is not None:
                count=state['attempts'].get(job,0)+1;state['attempts'][job]=count
                if count>=3:
                    state['failed'][job]={'error':str(error),'at':time.time()};state['inflight'].pop(job,None)
            else:
                if receipt is None:raise ValueError('Completion receipt required')
                write(self.done_path(job),receipt)
                if job in state['inflight']:
                    state['completed']+=1;state['completed_clips']=state.get('completed_clips',0)+receipt.get('clips',0)
                    del state['inflight'][job]
            write(self.path,state)
