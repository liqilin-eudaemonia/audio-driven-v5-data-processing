"""Bounded cross-source preparation with claims held until persisted outputs exist."""
from collections import deque
from concurrent.futures import wait,FIRST_COMPLETED
from pathlib import Path
import threading,time,traceback
from .j_annotation import read,write,result_path
from .j_omni_prefetch import prepare_call
from .j_omni_tail import TailSharing,terminal


def finish_source(work,q,job,quality_contract=None):
    rows=job['rows'];counts={b:0 for b in ('caption','held','visual_excluded','visual_review')};bad=0
    for row in rows:
        sid=row['sample_id'];present=[b for b in counts if result_path(work,b,sid).exists()]
        if len(present)>1:raise ValueError('Conflicting terminal branches: '+sid)
        if not present:return False
        branch=present[0];counts[branch]+=1
        if branch=='caption':bad+=not read(result_path(work,branch,sid))['generation_checks_passed']
    receipt=dict(job_id=job['job_id'],clips=len(rows),caption_clips=counts['caption'],
        failed_generation_clips=bad,held_clips=counts['held'],visual_excluded_clips=counts['visual_excluded'],
        visual_review_clips=counts['visual_review'],completed_at=time.time())
    if quality_contract:receipt['visual_quality_contract']=quality_contract
    q.finish(job['job_id'],receipt=receipt);return True


class CrossSourceStream:
    def __init__(self,work,q,name,cfg,pool,resources,profile,gate,state,stop,max_sources=4,recovery=None):
        self.work=Path(work);self.q=q;self.name=name;self.cfg=cfg;self.pool=pool;self.resources=resources
        self.profile=profile;self.gate=gate;self.state=state;self.stop=stop;self.cancel=threading.Event()
        self.max_sources=max_sources;self.capacity=profile['prefetch'];self.active={};self.pending={}
        self.quality={} if gate else None;self.recovery=recovery if recovery is not None else []
        self.sequence=0;self.generator=self._stream();self.failure=None
        self.tail=TailSharing(work,q,name,cfg,state)
    def __iter__(self):return self
    def __next__(self):return next(self.generator)
    def cancel_pending(self):self.cancel.set()
    def record_failure(self,sid,exc):self.failure=(sid,exc)
    def persisted(self,sid):
        # Called by the consumer only after atomic terminal output persistence;
        # avoid polling every live clip's filesystem paths in the busy path.
        for item in list(self.active.values()):
            handle=item['clip_locks'].pop(sid,None)
            if handle is not None:handle.close();return
    def scheduling_for(self,sid):
        if not self.tail.enabled:return None
        for jid,item in list(self.active.items()):
            if sid in item['waiting']:
                return dict(protocol=self.state['tail_protocol'],role='helper' if item['helper'] else 'owner',
                            job_id=jid,source_owner=item.get('source_owner',self.name))
        raise ValueError('Missing clip scheduling provenance: '+sid)
    def close(self):
        self.cancel.set();self.generator.close();self._close_claims()
    def _claim(self):
        from .j_omni import verified_job,pending_caption_rows
        # If the original source owner exits while this worker helps it, a
        # source claim could recover that same job and replace the live packet.
        # Finish packets first, including when the upstream queue refills and
        # the normal four-source limit is restored.
        if any(item['helper'] for item in self.active.values()):return False
        claim=(self.q.claim_failed(self.recovery.pop(0),self.name) if self.recovery else
               self.q.claim(self.name,eligible=self.gate.ready if self.gate else None))
        if claim is None:return False
        jid,handle=claim
        try:
            origin=read(self.work/'job_origins'/f'{jid}.json');job=verified_job(self.work/'jobs'/f'{jid}.json')
            if origin['source_result_sha256']!=job['source_result_sha256']:raise ValueError('Completion source differs')
            quality=self.gate.decisions(job) if self.gate else None
            todo,added=pending_caption_rows(self.work,job,self.cfg['pipeline_id'],quality)
            self.state['visual_excluded_clips']+=added['exclude'];self.state['visual_review_clips']+=added['review']
            if quality is not None:
                if set(quality)&set(self.quality):raise ValueError('Duplicate samples across active sources')
                self.quality.update(quality)
            self.active[jid]=dict(handle=handle,job=job,origin=origin,rows=deque(todo),
                                  waiting={r['sample_id'] for r in todo},futures=[],clip_locks={},helper=False)
            return True
        except BaseException as exc:
            self.q.finish(jid,error=repr(exc));handle.close();raise
    def _help(self):
        from .j_omni import verified_job,pending_caption_rows
        for offer in self.tail.offers(self.active):
            jid=offer['job_id'];locks={}
            try:
                origin=read(self.work/'job_origins'/f'{jid}.json')
                job=verified_job(self.work/'jobs'/f'{jid}.json')
                if origin['source_result_sha256']!=job['source_result_sha256'] or offer['source_result_sha256']!=job['source_result_sha256']:
                    raise ValueError('Helper source provenance differs')
                wanted=set(offer['sample_ids'])
                rows=[r for r in job['rows'] if r['sample_id'] in wanted]
                if len(rows)!=len(wanted):raise ValueError('Helper packet contains unknown sample')
                quality=self.gate.decisions(job) if self.gate else None
                # Admission and provenance checks are identical to the owner.
                todo,_=pending_caption_rows(self.work,dict(job,rows=rows),self.cfg['pipeline_id'],quality)
                picked=[]
                for row in todo:
                    sid=row['sample_id']
                    if self.tail.helper_failed_path(sid).exists():continue
                    handle=self.tail.reserve(sid)
                    if handle is None:continue
                    if terminal(self.work,sid):handle.close();continue
                    locks[sid]=handle;picked.append(row)
                    if len(picked)>=self.tail.packet_size:break
                if not picked:continue
                if quality is not None:
                    selected={r['sample_id']:quality[r['sample_id']] for r in picked}
                    if set(selected)&set(self.quality):raise ValueError('Duplicate helper samples')
                    self.quality.update(selected)
                # A packet has no source claim and can never commit the source receipt.
                self.active[jid]=dict(handle=None,job=dict(job,rows=picked),origin=origin,
                    rows=deque(picked),waiting={r['sample_id'] for r in picked},futures=[],clip_locks=locks,helper=True,
                    source_owner=offer['source_owner'])
                self.state['assisted_packets']+=1;self.state['assisted_clips_claimed']+=len(picked)
                self.state['last_assisted_packet']=dict(job_id=jid,sample_ids=[r['sample_id'] for r in picked])
                locks={};return True
            finally:
                for handle in locks.values():handle.close()
        return False

    def _next_row(self,item):
        # Rotate busy rows; their helper may still be preparing/inferencing. The
        # original owner will retry them if a helper exits without an output.
        for _ in range(len(item['rows'])):
            row=item['rows'].popleft();sid=row['sample_id']
            if self.tail.enabled:
                if terminal(self.work,sid):continue
                if sid not in item['clip_locks']:
                    handle=self.tail.reserve(sid)
                    if handle is None:item['rows'].append(row);continue
                    if terminal(self.work,sid):handle.close();continue
                    item['clip_locks'][sid]=handle
            return row
        return None
    def _retire(self):
        for jid,item in list(self.active.items()):
            if item['rows'] or any(v['job']==jid for v in self.pending.values()):continue
            # Check only outputs not already observed; no result-directory scans.
            item['waiting']={sid for sid in item['waiting'] if not any(result_path(self.work,b,sid).exists()
                for b in ('caption','held','visual_excluded','visual_review'))}
            for sid in list(item['clip_locks']):
                if sid not in item['waiting']:
                    handle=item['clip_locks'].pop(sid,None)
                    if handle is not None:handle.close()
            if item['waiting']:continue
            if not item['helper'] and not finish_source(self.work,self.q,item['job'],self.gate.contract if self.gate else None):
                raise ValueError('Source terminal accounting changed')
            if self.quality is not None:
                for row in item['job']['rows']:self.quality.pop(row['sample_id'],None)
            if item['handle'] is not None:item['handle'].close()
            del self.active[jid]
            self.state['last_progress_at']=time.time()
    def _stream(self):
        try:
            while not self.stop.is_set() and not self.cancel.is_set():
                self._retire()
                limit=self.tail.limit(self.max_sources)
                while len(self.active)<limit and not self.stop.is_set() and not self.cancel.is_set():
                    if not self._claim():break
                self.tail.advertise(self.active)
                # Help only when local preparation has no work. Do not reserve
                # extra packets just to fill a queue or evict active inference.
                if not self.pending and not any(v['rows'] for v in self.active.values()) and len(self.active)<self.max_sources:
                    self._help()
                self.state['active_sources']=list(self.active)
                if not self.active:
                    if (self.work/'producer_complete.json').exists():
                        qs=read(self.q.path)
                        if not qs['inflight'] and not qs.get('deferred'):break
                    self.cancel.wait(.5);continue
                blocked=set()
                while len(self.pending)<self.capacity and not self.stop.is_set() and not self.cancel.is_set():
                    choices=[(sum(p['job']==k for p in self.pending.values()),
                              -v['rows'][0]['source']['duration_s'],k) for k,v in self.active.items() if v['rows'] and k not in blocked]
                    if not choices:break
                    _,_,jid=min(choices);item=self.active[jid];row=self._next_row(item)
                    if row is None:blocked.add(jid);continue
                    future=self.pool.pool.submit(prepare_call,row,item['origin']['old_work'],self.work,self.resources)
                    item['futures'].append(future);self.sequence+=1
                    self.pending[future]=dict(row=row,job=jid,sequence=self.sequence)
                self.tail.advertise(self.active)
                if not self.pending:
                    # Already delivered inputs may still be in the inference mailbox.
                    self.cancel.wait(.1);continue
                tick=time.monotonic();done,_=wait(self.pending,timeout=.2,return_when=FIRST_COMPLETED)
                if not done:continue
                future=min(done,key=lambda f:self.pending[f]['sequence']);item=self.pending.pop(future)
                yield item['row'],future,time.monotonic()-tick
        finally:
            # The ready-reader can exit while the GPU consumer is still saving
            # its last request group. It must not release any claims here.
            # consume_source joins that reader first; run then calls close().
            self.cancel.set()

    def _close_claims(self):
        for item in self.active.values():
            for f in item['futures']:f.cancel()
        for item in self.active.values():
            for f in item['futures']:
                try:f.result()
                except BaseException:pass
        try:
            self._retire()
            if self.failure:
                sid,exc=self.failure
                for jid,item in self.active.items():
                    if any(r['sample_id']==sid for r in item['job']['rows']):
                        if item['helper']:
                            write(self.tail.helper_failed_path(sid),dict(sample_id=sid,job_id=jid,worker=self.name,error=repr(exc),at=time.time()))
                        else:self.q.finish(jid,error=repr(exc))
                        break
        finally:
            for item in self.active.values():
                for handle in item['clip_locks'].values():handle.close()
                if item['handle'] is not None:item['handle'].close()
            self.active.clear();self.pending.clear()
            self.tail.advertise(self.active)


def run(work,q,name,cfg,pool,resources,profile,gate,state,status,stop,llm,processor,node,gpu,revision,max_clips=None,recovery=None,max_sources=4):
    from .j_omni_queued import consume_source
    while not stop.is_set():
        stream=CrossSourceStream(work,q,name,cfg,pool,resources,profile,gate,state,stop,max_sources,recovery)
        try:
            state.update(stage='preparing',cross_source_prefetch=True,max_active_sources=max_sources);status()
            consume_source(stream,llm,processor,cfg,work,node,gpu,profile,revision,stream.quality,state,status,stop,max_clips)
            break
        except Exception as exc:
            traceback.print_exc();state['last_error']=repr(exc);stream.failure=(state.get('sample_id'),exc);status()
            stop.wait(2)
        finally:stream.close()
