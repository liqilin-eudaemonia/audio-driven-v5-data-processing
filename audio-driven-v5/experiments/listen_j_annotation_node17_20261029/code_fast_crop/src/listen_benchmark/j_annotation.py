"""Committed J source adaptation and recoverable append-only annotation queue."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
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

    def claim(self,worker):
        with locked(self.folder/'state.lock'):
            state=read(self.path)
            for job in list(state['inflight']):
                handle=acquire(self.folder/'claims'/f'{job}.lock',False)
                if handle is None:continue
                if self.done_path(job).exists():
                    del state['inflight'][job];state['completed']+=1
                    state['completed_clips']=state.get('completed_clips',0)+read(self.done_path(job)).get('clips',0)
                    handle.close();continue
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
                    if job in state['failed'] or job in state['inflight'] or self.done_path(job).exists():continue
                    handle=acquire(self.folder/'claims'/f'{job}.lock',False)
                    if handle is None:continue
                    state['inflight'][job]={'worker':worker,'pid':os.getpid(),'at':time.time()};write(self.path,state)
                    return job,handle
            write(self.path,state)
        return None

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
