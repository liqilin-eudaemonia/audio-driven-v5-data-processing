"""Bounded cross-source preparation with claims held until persisted outputs exist."""
from collections import deque
from concurrent.futures import wait,FIRST_COMPLETED
from pathlib import Path
import threading,time,traceback
from .j_annotation import read,result_path
from .j_omni_prefetch import prepare_call


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
    def __iter__(self):return self
    def __next__(self):return next(self.generator)
    def cancel_pending(self):self.cancel.set()
    def record_failure(self,sid,exc):self.failure=(sid,exc)
    def close(self):self.cancel.set();self.generator.close()
    def _claim(self):
        from .j_omni import verified_job,pending_caption_rows
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
                                  waiting={r['sample_id'] for r in todo},futures=[])
            return True
        except BaseException as exc:
            self.q.finish(jid,error=repr(exc));handle.close();raise
    def _retire(self):
        for jid,item in list(self.active.items()):
            if item['rows'] or any(v['job']==jid for v in self.pending.values()):continue
            # Check only outputs not already observed; no result-directory scans.
            item['waiting']={sid for sid in item['waiting'] if not any(result_path(self.work,b,sid).exists()
                for b in ('caption','held','visual_excluded','visual_review'))}
            if item['waiting']:continue
            if not finish_source(self.work,self.q,item['job'],self.gate.contract if self.gate else None):
                raise ValueError('Source terminal accounting changed')
            if self.quality is not None:
                for row in item['job']['rows']:self.quality.pop(row['sample_id'],None)
            item['handle'].close();del self.active[jid]
            self.state['last_progress_at']=time.time()
    def _stream(self):
        try:
            while not self.stop.is_set() and not self.cancel.is_set():
                self._retire()
                while len(self.active)<self.max_sources and not self.stop.is_set() and not self.cancel.is_set():
                    if not self._claim():break
                self.state['active_sources']=list(self.active)
                if not self.active:
                    if (self.work/'producer_complete.json').exists():
                        qs=read(self.q.path)
                        if not qs['inflight'] and not qs.get('deferred'):break
                    self.cancel.wait(.5);continue
                while len(self.pending)<self.capacity and not self.stop.is_set() and not self.cancel.is_set():
                    choices=[(sum(p['job']==k for p in self.pending.values()),
                              -v['rows'][0]['source']['duration_s'],k) for k,v in self.active.items() if v['rows']]
                    if not choices:break
                    _,_,jid=min(choices);item=self.active[jid];row=item['rows'].popleft()
                    future=self.pool.pool.submit(prepare_call,row,item['origin']['old_work'],self.work,self.resources)
                    item['futures'].append(future);self.sequence+=1
                    self.pending[future]=dict(row=row,job=jid,sequence=self.sequence)
                if not self.pending:
                    # Already delivered inputs may still be in the inference mailbox.
                    self.cancel.wait(.1);continue
                tick=time.monotonic();done,_=wait(self.pending,timeout=.2,return_when=FIRST_COMPLETED)
                if not done:continue
                future=min(done,key=lambda f:self.pending[f]['sequence']);item=self.pending.pop(future)
                yield item['row'],future,time.monotonic()-tick
        finally:
            self.cancel.set()
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
                            self.q.finish(jid,error=repr(exc));break
            finally:
                for item in self.active.values():item['handle'].close()
                self.active.clear();self.pending.clear()


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
