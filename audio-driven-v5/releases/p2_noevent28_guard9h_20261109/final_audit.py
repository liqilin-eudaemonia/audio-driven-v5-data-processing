"""Completion proof after the real nine-hour window; training continues."""
from pathlib import Path
import time,xml.etree.ElementTree as ET
import common as C
import guardian as G
W=C.WORK
p=C.read(W/'protocol.json');assert time.time()>=p['deadline']and(W/'WINDOW_FINISHED.json').exists()
reports=[]
for slot in range(10):
 r=C.read(W/'reports'/f'hour_{slot:02d}.json');assert r['slot']==slot and r['at']>=p['start']+slot*3600
 assert C.observed(r['cluster'])and r['contract']['success'];reports.append(dict(slot=slot,at=r['at'],step=r['health']['step'],healthy28=r['cluster']['healthy28'],flags=r['health']['flags']))
assert reports[-1]['healthy28']and not reports[-1]['flags']
C.contract(True);snap=C.probe();C.atomic_json(W/'FINAL_CLUSTER.json',snap);assert snap['healthy28']and C.observed(snap)
health=C.health();assert not health['flags']and not health['complete']
checkpoint,receipt=G.checked_latest();begin=health['step'];deadline=time.time()+300
while C.health()['step']<=begin:
 assert time.time()<deadline;time.sleep(10)
health=C.health();assert not health['flags'];assert C.alive(C.read(C.OUT/'CONTROLLER.json'))
speed=G.speed_baseline()()
if speed['ratio']is not None and speed['ratio']>1.5:
 C.atomic_json(W/'FINAL_UNRESOLVED_SLOWDOWN.json',dict(at=time.time(),health=health,speed=speed,reason='Known shared GPU compute remains; do not claim slowdown repaired',permission_pending=True))
 raise RuntimeError('Nine-hour observation ended with unresolved compute contention slowdown')
suite=ET.parse(W/'tests.xml').getroot().find('testsuite');assert suite.attrib['tests']=='34'and suite.attrib['failures']==suite.attrib['errors']=='0'
result=dict(success=True,at=time.time(),start=p['start'],deadline=p['deadline'],elapsed_seconds=time.time()-p['start'],hourly_reports=reports,final_health=health,latest_checkpoint=receipt,training_continues=True,reserved_gpu='node11:4-7',tests=34,training_source_unchanged=True)
C.atomic_json(W/'FINAL_ACCEPTED.json',result);print(dict(success=True,step=health['step'],elapsed_seconds=result['elapsed_seconds']),flush=True)
