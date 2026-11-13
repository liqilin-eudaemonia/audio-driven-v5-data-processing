#!/usr/bin/env python3
"""Node8-local queue: wait for prior evaluation, label on all 8 GPUs, export and audit."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/listen_j_upright_regen_20261105'
OUT=ROOT/'experiments/listen_j_labels_upright_20261105'
PY='/private/software/conda/envs/vllm/bin/python'
EXPORT_PY='/private/software/conda/bin/python'
OTHER='p2_seamless200_nineway_20261105'

def read(p):return json.loads(Path(p).read_bytes())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def state(stage,**kw):save(WORK/'QUEUE_STATUS.json',dict(stage=stage,heartbeat=time.time(),hostname=socket.gethostname(),pid=os.getpid(),**kw))
def gpu_state():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return [list(map(int,x.split(','))) for x in s.splitlines()]
def other_active():
 s=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
 return [x.strip()[:1600] for x in s.splitlines() if (('p2_seamless200_nineway.py' in x and ' run ' in x) or (OTHER in x and any(t in x for t in [' worker ',' controller ',' score ']))) and 'queue_j_upright_node8.py' not in x and 'def other_active' not in x]
def run_stage(name,args):
 with (WORK/(name+'.log')).open('ab') as log:
  p=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state(name,child_pid=p.pid,command=args)
  while p.poll() is None:
   state(name,child_pid=p.pid,command=args);time.sleep(10)
  if p.returncode:raise RuntimeError(name+' failed: '+str(p.returncode)+'; see '+str(WORK/(name+'.log')))

def summarize_generation():
 jobs={r['sample_id']:r for r in map(json.loads,(WORK/'regenerate_candidates.jsonl').open())}
 buckets={name:{p.stem:p for p in (WORK/name).glob('*.json')} for name in ['regenerated','regeneration_rejected','excluded_visual']}
 ids=[]
 for d in buckets.values():ids.extend(d)
 assert len(ids)==len(set(ids))==len(jobs) and set(ids)==set(jobs)
 events=[json.loads(x) for x in (WORK/'node8_gpu_telemetry.jsonl').open()];repeated=set(read(WORK/'repeated_sample_ids.json'))
 result=dict(total=len(jobs),counts={k:len(v) for k,v in buckets.items()},repeated_total=len(repeated),
  repeated_accepted=len(repeated&set(buckets['regenerated'])),repeated_excluded=len(repeated-set(buckets['regenerated'])),
  gpu_peaks={g:dict(memory_mib=max(v[1] for e in events for v in e['gpus'] if v[0]==g),utilization=max(v[2] for e in events for v in e['gpus'] if v[0]==g)) for g in range(8)},
  output_sha256={name:{sid:sha(p) for sid,p in paths.items()} for name,paths in buckets.items()},completed_at=time.time())
 save(WORK/'CAPTION_REGENERATION_COMPLETE.json',result)
 return result

def publish_docs(generation):
 d=read(OUT/'EXPORT_SUMMARY.json');c=d['counts'];a=read(OUT/'UPRIGHT_AUDIT.json');assert a['status']=='passed' and read(OUT/'EXPORT_COMPLETE.json')['status']=='complete'
 text=f'''# 新全量标签统一入口：原画转正＋重复重打标（2026-11-05）

本目录已发布完成，可作为后续行为统计和训练／测试选集的统一标签入口。主文件 [labels.jsonl](labels.jsonl)，**{c['labels']:,}条**；[UPRIGHT_AUDIT.json](UPRIGHT_AUDIT.json)独立验收通过，[EXPORT_COMPLETE.json](EXPORT_COMPLETE.json)绑定摘要与验收SHA。

自动通过及待复核均保留，保留复核标记{c['with_review_flags']:,}条。原画横置747个来源／1,699段，{c.get('orientation_restored',0):,}段已转正并重新检测、打标，{c.get('rotation_excluded',0):,}段因转正后视觉或生成仍不可用而剔除。方向正常的宽屏、正常歪头均保留。

全部36条原重复标记均已重打标：{c.get('repeated_regenerated',0)}条保留、{c.get('repeated_excluded',0)}条剔除。新标签正常结束且无病态长循环；普通复核标记继续保留。最终剔除共{c.get('excluded',0)}条，见[excluded.jsonl](excluded.jsonl)；原有1条无新标签例外保持单列，见[exceptions.jsonl](exceptions.jsonl)。[coverage.jsonl](coverage.jsonl)完整归账1,107,323个目标ID。

| split | 可用标签 |
|---|---:|
| train | {d['splits'].get('train',0):,} |
| dev | {d['splits'].get('dev',0):,} |
| test | {d['splits'].get('test',0):,} |

[replacements.jsonl](replacements.jsonl)记录每条旧／新caption及SHA；[EXPORT_SUMMARY.json](EXPORT_SUMMARY.json)给出全量文件大小、SHA和版本。未受影响的原标签逐行字节保留，改写原行0；转正／重复修复各行有自己的caption_identity_version，应逐行读取，不能假定全池仍是同一个旧cache身份。`training_admitted`仍为false，表示尚未执行新的训练选集。

转正行的source已指向独立原生裁框转正素材的TAR/NUT字节范围，时间轴、sample ID、pair、split／group、participant和净化音轨保持原样。原生分辨率未缩放；档案SHA、每段首尾PTS和代表帧像素对照均核验，新的25fps检测输入也核验。转正行的quality、face_p10／p50、crop_provenance和detector SHA来自新素材；orientation_restoration保留完整旧来源／质量及修复依据。原素材、旧caption、旧缓存没有覆盖。

node8新地址172.29.104.237的GPU0–7以8个独立模型副本运行，单卡最多4条并发、CPU预取与同次解码复用。启动、GPU实际利用率及完整打标结果见[执行记录](../listen_j_upright_regen_20261105/README.md)。

后续：使用本目录重算新caption行为统计，再按原规则选取实际训练／测试集，并为转正行重建受影响的首帧／视觉／条件缓存。历史500h清单、旧SFT选集及正在运行的实验未改写。历史未转正全量入口位于[父清单](../listen_j_labels_manifest_20261105/README.md)，用于追溯。
'''
 (OUT/'README.md').write_text(text)
 p=WORK/'README.md';old=p.read_text();marker='## 最终完成结果'
 if marker not in old:p.write_text(old+f"\n{marker}\n\n已完成八卡重打标及全量独立验收。新统一入口为[../listen_j_labels_upright_20261105/README.md](../listen_j_labels_upright_20261105/README.md)，保留{c['labels']:,}条，剔除{c.get('excluded',0)}条；原1条缺项单列。CAPTION_REGENERATION_COMPLETE.json保存逐新标签SHA与八卡实测峰值，QUEUE_STATUS.json为complete。前文等待／失败启动为历史记录。\n")
 root=ROOT/'README.md';marker='**原画转正与重复标签八卡重打标已完成（2026-11-05）：**'
 entry=marker+f" 全量新标签保留{c['labels']:,}条；747个原画横置来源／1,699段中{c.get('orientation_restored',0):,}段成功转正并重新检测打标，其余不可用段剔除；全部36条重复标记重打标，{c.get('repeated_regenerated',0)}条保留、{c.get('repeated_excluded',0)}条剔除。正常宽屏与歪头、自动通过与待复核均保留，原音频／时间／split不变。全部1,107,323个目标归账、原字节保留与新素材SHA独立验收通过。[新全量统一入口](experiments/listen_j_labels_upright_20261105/README.md) · [执行及八卡实测](experiments/listen_j_upright_regen_20261105/README.md)。\n\n"
 old=root.read_text()
 if marker not in old:
  temp=root.with_name('README.j-upright.tmp');temp.write_text(entry+old)
  if root.read_text()!=old:temp.unlink();raise RuntimeError('README changed concurrently; repeat docs publish')
  temp.replace(root)

def main():
 assert socket.gethostname()=='dedicated-developjob-zz-dev-machine-1-wx3im'
 lock=(WORK/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(WORK/'input_preflight.json')['status']=='passed'
 assert read(WORK/'VOCAL_ORIGINAL_ARCHIVE_PREFLIGHT.json')['status']=='passed'
 save(WORK/'QUEUE_LAUNCHED.json',dict(hostname=socket.gethostname(),pid=os.getpid(),started_at=time.time(),wait_for_other_evaluation=True,other_evaluation=OTHER,script_sha256=sha(Path(__file__))))
 if (OUT/'EXPORT_COMPLETE.json').exists():
  publish_docs(read(WORK/'CAPTION_REGENERATION_COMPLETE.json'));state('complete',manifest=str(OUT/'labels.jsonl'));return
 while True:
  if (WORK/'STOP_QUEUE').exists():state('stopped_before_launch');return
  gpus=gpu_state();other=other_active()
  if len(gpus)==8 and all(v[1]<2000 for v in gpus) and not other:break
  state('waiting_for_node8',gpus=gpus,other_evaluation_processes=other,policy='Wait for current evaluation to finish; do not take its between-stage gaps')
  time.sleep(15)
 state('launching_eight_gpu_regeneration')
 run_stage('node8_controller',[PY,str(ROOT/'scripts/regenerate_j_upright_captions_v4.py'),'launch','--max-num-seqs','4'])
 generation=summarize_generation()
 run_stage('export_unified_manifest',[EXPORT_PY,str(ROOT/'scripts/export_j_upright_labels.py'),'export'])
 run_stage('audit_unified_manifest',[EXPORT_PY,str(ROOT/'scripts/export_j_upright_labels.py'),'audit'])
 publish_docs(generation)
 state('complete',manifest=str(OUT/'labels.jsonl'),audit=str(OUT/'UPRIGHT_AUDIT.json'),counts=read(OUT/'EXPORT_SUMMARY.json')['counts'])

if __name__=='__main__':
 try:main()
 except Exception as e:
  state('failed_requires_fix',error=repr(e));raise
