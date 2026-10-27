"""Pin prepared launch code to the running E0+Feature source; never launches training."""
from control import *
old=read(REFERENCE/'release.json')
for p,h in old['files'].items():
 if sha(p)!=h:raise ValueError('E0+Feature frozen input changed: '+p)
paths=[OUT/n for n in ('control.py','controller.py','run_node.py','config.json')]
files={**old['source'],**{str(p):sha(p) for p in paths},str(ROOT/'configs/p2_listen_h200_multinode.env'):sha(ROOT/'configs/p2_listen_h200_multinode.env'),str(TRAIN/'config.json'):sha(TRAIN/'config.json')}
write(OUT/'release.json',dict(arm='E1',experiment='E1+Feature(0.1,0.1)',seed=SEED,steps=BUDGET,files=files,source=old['source'],created_at=now()))
verify_release();print('PREPARED_SOURCE_VERIFIED_NO_TRAINING_LAUNCHED')
