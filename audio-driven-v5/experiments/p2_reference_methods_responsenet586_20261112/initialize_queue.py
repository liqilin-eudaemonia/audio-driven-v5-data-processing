"""Publish prepared/cached progress without allocating a GPU or claiming full completion."""
from runtime import *
from report import report,METHODS,RGB,NATIVE
assert read(WORK/'REUSE_COMPLETE.json')['passed']
assert not (WORK/'MONITOR_PROCESS.json').exists(),'Do not reset an already started run'
ss=selected();video={};unavailable={};scored={}
for m in RGB:
    receipts=[read(p) for p in (WORK/m/'responsenet').glob('*/complete.json')]
    video[m]=sum(bool(r.get('score_path')) for r in receipts);unavailable[m]=sum(r.get('status')=='unavailable' for r in receipts)
    scored[m]=sum(read(p)['clips'] for p in (WORK/'metrics'/m/'responsenet').glob('*/SUCCESS.json'))
save(WORK/'status.json',dict(stage='queued_after_seamless6624',scope=SCOPE,inputs=TOTAL,new_inputs=386,features=sum((WORK/'features/responsenet'/s['id']/'complete.json').exists() for s in ss),evaluation_gt=sum((WORK/'evaluation_gt/responsenet'/s['id']/'complete.json').exists() for s in ss),motion={m:sum((folder(m,s)/'MOTION_COMPLETE.json').exists() for s in ss) for m in ['listenformer',*NATIVE]},video=video,video_unavailable=unavailable,scored=scored,active=[],active_gpus={},failures={},predecessor=read(WORK/'WAIT_FOR.json'),cached_only=True,at=time.time()))
report();print('Queued ResponseNet586; only200-input cached outputs present; no GPU workers launched',flush=True)
