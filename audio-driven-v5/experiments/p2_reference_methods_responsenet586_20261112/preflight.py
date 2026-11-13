"""CPU verification of full scope, original batches, pinned reader and checkpoint files."""
from runtime import *
reader=activate_readers();ss=selected();bb=groups();assert len(bb)==24 and len(bb[-1])==11
assert [s['key'] for batch in bb for s in batch]==[s['key'] for s in ss]
for s in ss:
    assert s['dataset']=='responsenet'
    assert sha(s['reference'])==s['reference_sha256'] and sha(s['audio'])==s['audio_sha256']
    assert Path(s['evaluation_reference']['gt']).is_file()
    audit=read(s['source_input_audit']);assert audit['source_only'] and not audit['listener_future_media_in_model_input']
print('All586 source-only inputs and24 batches verified',flush=True)
assets={}
for group,files in read(WORK/'MODEL_MANIFEST.json')['models'].items():
    for p,h in files.items():
        if p not in assets:assert sha(p)==h,p;assets[p]=h
print('Checkpoint assets verified:',len(assets),flush=True)
save(WORK/'MODEL_VERIFIED.json',dict(passed=True,files=assets,manifest_sha256=sha(WORK/'MODEL_MANIFEST.json'),at=time.time()))
save(WORK/'PREFLIGHT.json',dict(passed=True,total=TOTAL,batches=len(bb),last_batch=len(bb[-1]),original200_batches_unchanged=True,pinned_reader=str(reader.__file__),checkpoint_files=len(assets),at=time.time()))
