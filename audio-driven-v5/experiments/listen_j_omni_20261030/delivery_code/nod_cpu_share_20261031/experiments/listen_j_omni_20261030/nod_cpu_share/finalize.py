import hashlib,json,shutil,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent.parent;R=W.parents[1];sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write,locked
F=W/'nod_cpu_share';assert read(F/'audit.json')['passed']
reports={label:{n:read(F/label/f'{n}.json') for n in ['node17','node8','node4','node12']} for label in ['before','ramp1','steady1','final_window']}
metrics={label:dict(nod_source_receipt_clips_per_second=ns['node17']['nod_clips_per_second'],omni_caption_clips_per_second=sum(x['clips_per_second'] for x in ns.values()),cpu_cores={n:x['cpu_cores'] for n,x in ns.items()}) for label,ns in reports.items()}
with locked(W/'config-edit.lock'):
 c=read(W/'config.json');c['nod_cpu_sharing']=dict(config=str(F/'config.json'),handoff=str(R/'related/P2_点头检测跨机CPU分担_20261031.md'),validated_at=time.time(),effective_nod_workers=168,quality_workers_total=160,original_node8_waiting_workers=80);write(W/'config.json',c)
write(F/'accepted_configuration.json',dict(at=time.time(),status='deployed_and_verified',sharing=read(F/'config.json'),runtime=c,metrics=metrics,validation=str(F/'audit.json'),semantics_unchanged=True,goal_resumed=False,measurement_note='Different production clips in sequential windows; no same-sample speedup claim. Per-clip nod outputs and source-completion counters have different commit timing.'))
p=R/'README.md';s=p.read_text();s=s.replace('正在完成真实生产部署与测速。','已部署并验收：点头队列有效进程 168，专用画质进程 160；末两窗口点头整源提交约 6.6–7.4 段/秒，Omni 32卡持续处理。',1);s=s.replace('专用画质CPU现为256进程（node8点头队列耗尽后新增32），已验证并停止继续试参，保留后台生产与故障恢复。','早期专用画质CPU为256进程；当前按上方跨机点头方案调整为160进程，保留后台生产与故障恢复。',1);p.write_text(s)
p=W/'README.md';s=p.read_text();lead='\n**2026-10-31 跨机 CPU 点头分担已部署：** node17/node8/node4/node12 分别使用 48/48/32/40 个进程处理剩余 node17 点头队列；专用画质池分别保留 32/32/48/48。Omni 32卡和输入预取配置不变，输出继续写原目录。[交接](../../related/P2_点头检测跨机CPU分担_20261031.md) · [配置与测速](nod_cpu_share/accepted_configuration.json) · [验收](nod_cpu_share/audit.json)。此前 goal 仍暂停。\n'
lines=s.splitlines(keepends=True);lines.insert(1,lead+'\n');p.write_text(''.join(lines))
p=R/'related/P2_画质前置筛查与四机32卡运行_20261031.md';s=p.read_text();lines=s.splitlines(keepends=True);lines.insert(1,'\n> 最新资源调整：六小时巡检已结束、goal 已由用户暂停；后续用户单独授权跨机 CPU 分担点头，当前点头有效进程168、专用画质进程160，见 [跨机点头交接](P2_点头检测跨机CPU分担_20261031.md)。本文旧256进程配置保留为历史证据。\n\n');p.write_text(''.join(lines))
p=R/'related/P2_点头检测跨机CPU分担_20261031.md';s=p.read_text();s+='\n最后120秒窗口：点头整源提交约6.63段/秒，Omni四机合计约10.00段/秒；CPU平均占用分别约135/123/118/138核（每机160核配额）。画质新池刚启动时整源提交会有延迟，不能将其短窗口回执率等同于持续测量速度。所有队列失败数为0，原32个GPU进程未重启。正式配置与全部窗口见 `nod_cpu_share/accepted_configuration.json`。\n';p.write_text(s)
backup=W/'delivery_code/nod_cpu_share_20261031';paths=[R/'scripts/listen_j_nod_cpu_share.py',R/'tests/test_j_nod_cpu_share.py',R/'related/P2_点头检测跨机CPU分担_20261031.md',R/'related/P2_画质前置筛查与四机32卡运行_20261031.md',R/'README.md',W/'README.md',W/'config.json']
# Only the bounded rollout directory's direct scripts/JSON and named reports.
paths+=list(F.glob('*.py'))+list(F.glob('*.json'))
for label in reports:
 paths += [F/label/f'{n}.json' for n in reports[label]]
for n in ['node8','node4','node12']:
 for name in ['launch.json','quality_resize.json']:
  p=F/n/name
  if p.exists():paths.append(p)
manifest={}
for src in dict.fromkeys(paths):
 rel=src.relative_to(R);dst=backup/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);h=hashlib.sha256(src.read_bytes()).hexdigest();assert hashlib.sha256(dst.read_bytes()).hexdigest()==h;manifest[str(rel)]=h
write(backup/'SHA256.json',dict(at=time.time(),files=manifest));print(json.dumps(dict(status='deployed_and_verified',backup_files=len(manifest),metrics=metrics),ensure_ascii=False))
