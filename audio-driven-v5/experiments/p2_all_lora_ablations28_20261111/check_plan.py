import runtime as R
from p2_ltx_all_lora_components.config import expected
from p2_ltx_all_lora_components.schedule import Schedule as Native
for arm in ['no_feature','no_whisper']:
 c=R.read(R.output(arm)/'config.json');R.validate(c)
 assert R.normalized(c)==expected(R.output(arm),arm)
 old=Native(R.normalized(c));new=R.Schedule(c)
 assert old.pairs[:186004]==new.pairs[:186004]
 for step in [1,500,6643]:
  assert new.indices(new.size,step,28,new.seed)==[new.indices(new.size,step,28,new.seed,i,28)[0]for i in range(28)]
 assert c['training']['save_every']==500
 assert R.read(R.output(arm)/'preflight.json')['source']==R.source()
r=R.noevent_runtime();c=R.read(R.NOEVENT/'config.json');r.validate(c)
assert r.source()==R.read(R.NOEVENT/'run.json')['source']
assert r.Schedule(c).bounds(6573)[1]==186024
R.atomic_json(R.W/'PLAN_ACCEPTED.json',dict(success=True,targets=R.TARGETS,reference_samples=186000,source=R.source(),fresh_sample_noise_prefix_unchanged=True,noevent_source_unchanged=True,real_cache_reads_per_new_arm=32))
print('PASS')
