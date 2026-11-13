from pathlib import Path
import time,shutil,xml.etree.ElementTree as ET
import manager as X
C=X.C;W=X.W;OUT=X.OUT
# manager writes run.json only after exact tensor comparison and full SHA/finite checks.
deadline=time.time()+1800
while not(OUT/'MIGRATION.json').exists()or C.read(OUT/'run.json')['source']!=X.R.source():
 if time.time()>deadline:raise TimeoutError('Final migrated source identity never verified')
 time.sleep(5)
ready=C.read(W/'READY.json');assert ready['source']==X.R.source()==C.read(OUT/'run.json')['source']
assert C.sha(OUT/'config.json')==ready['config_sha256']and C.sha(OUT/'cluster.json')==ready['cluster_sha256']
m=C.read(OUT/'MIGRATION.json');assert m['model_and_adam_equal']and m['step']==660 and m['world']==28
assert m['destination_sha256']==C.read(OUT/'step_000660.complete.json')['sha256']
s=ET.parse(W/'tests.xml').getroot().find('testsuite');assert s.attrib['tests']=='4'and s.attrib['failures']==s.attrib['errors']=='0'
assert C.read(W/'TAIL_NOEVENT_ACCEPTED.json')['success']
assert 'preflight passed; no GPU used'in(W/'COLD.log').read_text()
assert C.alive(C.read(W/'HANDOFF_CONTROLLER.json'))and C.read(W/'HANDOFF_STATE.json')['stage']=='waiting_full7500'
C.write(W/'MIGRATION_READY.json',dict(success=True,at=time.time(),source=ready['source'],migration=m))
C.write(W/'PREPARED_ACCEPTED.json',dict(success=True,at=time.time(),tests=4,tail_three_process_passed=True,cold_runtime_passed=True,migration=m,world=28,global_batch=28,full_stop_step=7500,noevent_resume_step=660,reserved='node11 GPU4-7',handoff_controller=C.read(W/'HANDOFF_CONTROLLER.json'),handoff_sha256=C.sha(W/'handoff.py'),status='armed_waiting_full7500',gpu_startup_validation_pending=True))
dest=X.ROOT/'releases/p2_noevent28_after_full7500_20261109';files={Path(p)for p in ready['source']}
files.update(p for p in W.iterdir()if p.is_file()and p.suffix in {'.py','.md','.json','.xml'})
files.update(OUT/p for p in ['config.json','cluster.json','preflight.json','run.json','MIGRATION.json']);manifest={}
for p in sorted(files):
 q=dest/p.relative_to(X.ROOT.parent);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);h=C.sha(p);assert C.sha(q)==h;manifest[str(p)]=h
C.write(dest/'MANIFEST.json',dict(at=time.time(),files=manifest))
p=X.ROOT/'README.md';entry='**已预约完整组7500保存后切去Event全LoRA28卡（2026-11-09）：** 完整7500模型／Adam／28RNG校验、旧28rank退出后，从去Event完整660保留模型／Adam接续；同28卡，node11仅0–3、4–7预留，global batch28。原剩余sample/noise计划不变，尾批19，一轮终点10081；每500步保存、启动最长样本两次真实更新并恢复后验收前8步，再每小时观察。660迁移逐值校验、4项计划／7500门槛测试、3rank尾批归约、冷入口通过；四机单网卡／all重计算保持。24卡同744样本实测batch24较31吞吐+15.69%、耗时−13.57%，不宣称28卡全局最优。[准备与实际队列状态](experiments/p2_noevent28_after_full7500_20261109/README.md) · [挂载验收](experiments/p2_noevent28_after_full7500_20261109/PREPARED_ACCEPTED.json)。\n\n';p.write_text(entry+p.read_text());print('Prepared, migrated and armed; verified backup',len(files),flush=True)
