"""Regression: a resumed worker must not inherit an expired GPU7 time slice."""
from runtime import *
from resource_control import refresh_idle_clocks
policy=read(WORK/'RESOURCE_POLICY.json');now=10000.;checks=[]
leases={'7':1000.};cooldown={};empty={'7':9000.}
refresh_idle_clocks({},leases,cooldown,empty,{'7':{'external':[]}},policy,now)
assert '7' not in leases;checks.append('Expired lease cleared after long idle gap')
leases['7']=now;active={'new_job':{'gpu':7,'estimate_gib':8}}
refresh_idle_clocks(active,leases,cooldown,empty,{'7':{'external':[]}},policy,now+44)
assert leases['7']==now and now+44-leases['7']<policy['lease_seconds'];checks.append('New worker survives44s instead of false lease expiry')
leases={'7':1000.};cooldown={};empty={}
refresh_idle_clocks({},leases,cooldown,empty,{'7':{'external':[]}},policy,now)
assert '7' not in leases and cooldown['7']==now+policy['release_seconds'];checks.append('Short quiet gap enforces handoff without carrying old lease into next launch')
leases={'7':9990.};empty={}
refresh_idle_clocks({},leases,{},empty,{'7':{'external':[{'pid':123}]}},policy,now)
assert '7' not in leases;checks.append('Foreground-only period clears prior background clock')
assert list(GPU_UUIDS)==[7] and policy['allowed_gpus']==[7]
os.environ['CUDA_VISIBLE_DEVICES']='4'
try:device()
except AssertionError:checks.append('GPU4 rejected before initializing CUDA')
else:raise AssertionError('GPU4 must be forbidden')
save(WORK/'LEASE_REGRESSION.json',dict(passed=True,checks=checks,code_sha256=sha(WORK/'resource_control.py'),at=time.time()))
print('Passed',len(checks),'resource scheduling regression checks')
