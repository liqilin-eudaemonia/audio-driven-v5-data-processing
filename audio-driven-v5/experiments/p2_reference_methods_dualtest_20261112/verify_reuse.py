from runtime import *
import numpy as np,soundfile as sf
p=read(WORK/'protocol.json');samples={s['key']:s for s in p['samples']};records=read(WORK/'DY_REUSE_CANDIDATES.json')['records']
for r in records:
    old,rate=sf.read(r['old_audio'],dtype='float32');new,nrate=sf.read(r['new_audio'],dtype='float32')
    assert rate==nrate==16000 and np.array_equal(old,new),r['key']
    assert hashlib.sha256(old.astype('<f4').tobytes()).hexdigest()==r['old_waveform_sha256']
    s=samples[r['key']];dest=folder('dystream',s);dest.mkdir(parents=True,exist_ok=True)
    score=dest/'scoring.mp4'
    if not score.exists():score.symlink_to(r['video'])
    assert sha(score)==r['video_sha256']
    publish('dystream',s,[score],reused=True,source_receipt=r['receipt'],source_receipt_sha256=r['receipt_sha256'],source_protocol=r['old_protocol'],source_protocol_sha256=r['old_protocol_sha256'],score_path=str(score),score_frames=s['frames'],score_fps=s['fps'])
save(WORK/'DY_REUSE_COMPLETE.json',dict(passed=True,clips=len(records),waveform_exact=True,at=time.time()))
print('Reused verified DyStream',len(records))
