#!/usr/bin/env python3
"""Capture processing source and receipts only; never execute a pipeline."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime,timezone

DEST=Path(__file__).resolve().parent
REPO=DEST.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('/private/liqi_workspace/project_code/audio-driven-v5'))
    a=parser.parse_args();source=a.source.resolve();project=source.parent
    if (DEST/'BACKUP_MANIFEST.json').exists():raise SystemExit('Sealed snapshot already exists; use a new backup version')
    files={};chosen=set()
    def add(p):
        p=Path(p)
        if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:chosen.add(p)
    packages=['listen_benchmark','listen_labeling','listen_audio_full','listen_audio_postprocess','p2_listen','p3_listen']
    for name in packages:
        for p in (source/'src'/name).rglob('*.py'):add(p)
    # Carry forward source entry points already backed up for the upstream
    # segmentation/labeling pipeline, at their CURRENT bytes.
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0')
    old_index={}
    for n in tracked:
        p=REPO/n
        if n and p.is_file() and p.suffix in {'.py','.sh','.json','.md'}:
            old_index.setdefault(sha(p),[]).append(n)
        if n.startswith(('scripts/','src/','tests/','configs/')) and Path(n).suffix in {'.py','.sh','.json','.md'}:add(source/n)
    for p in (source/'scripts').glob('*'):
        if p.suffix=='.py' and (any(k in p.name for k in ['listen','postprocess','p2_full_features']) or p.name in ['select_processed_test_panel.py','report_open_backchannel_times.py','report_postprocessed_backchannels.py','render_backchannel_corpus_report.py','prepare_open_verbatim_subset.py','select_open_verbatim_subset.py','audit_open_verbatim_subset.py']):add(p)
    for p in (source/'tests').rglob('*.py'):
        if any(k in str(p.relative_to(source/'tests')) for k in ['listen','postprocessed','processed_test','open_verbatim','inference_isolation','p2_full_feature']):add(p)
    for p in (source/'configs').glob('*.json'):
        if 'listen' in p.name:add(p)
    for p in (source/'related').glob('P2_*.md'):
        if any(k in p.name for k in ['打标','整合','后处理','静音','原文','音频时间','中文','特征抽取','backchannel','300h','最终数据集']):add(p)
    add(source/'README.md')
    runs=['listen_raw_crop384_20261024T131300Z','listen_full_fusion_crop384_20261025','p2_features_full_crop384_20261025',
          'p2_full_audio_postprocess_crop384_20261026','p2_train50h_postprocessed_reselected_20261027',
          'p2_train300h_seven_20261027','p2_train300h_open_20261027','p2_test24_backchannel_face_20261027','p2_test48_backchannel_face_20261027']
    allowed={'README.md','config.json','media_cache_config.json','p2_config.json','COMPLETE.json','dataset_contract.json',
             'summary.json','validation.json','selection_policy.json','split_audit.json','code_manifest.json','scope.json',
             'source_shards.json','launch.json','launch_validation.json','scope_audit.json','capacity_preset.json','PREPARE_COMPLETE.json'}
    external_refs=[]
    for run in runs:
        base=source/'experiments'/run
        for p in base.iterdir():
            if p.is_file() and (p.name in allowed or p.suffix in {'.py','.sh'}) and p.stat().st_size<5_000_000:add(p)
        for n in ['view/config.json','view/audio_postprocess/policy_manifest.json','audio_postprocess/policy_manifest.json']:add(base/n)
        code_manifest=base/'code_manifest.json'
        if code_manifest.exists():
            cm=json.loads(code_manifest.read_text())
            for n in cm:
                p=Path(n) if Path(n).is_absolute() else source/n
                if p.suffix in {'.py','.sh'}:add(p)
        marker=base/'COMPLETE.json'
        if marker.exists():
            external_refs.append(dict(run=str(base),complete_sha256=sha(marker),
                dataset_files_included=False,source_completion=json.loads(marker.read_text())))
    # All dependencies already curated by the earlier audio-processing backup,
    # refreshed from the actual working tree and recorded by hash.
    old=REPO/'event_labeling/audio_postprocess_20261026'
    old_manifest=json.loads((old/'BACKUP_MANIFEST.json').read_text())
    for n,row in old_manifest['files'].items():
        if not n.startswith('snapshot/audio-driven-v5/'):
            p=Path(row['source'])
            if not p.is_file():raise RuntimeError('Missing dependency '+str(p))
            add(p)
    for p in sorted(chosen):
        rel=Path('snapshot')/p.relative_to(project);out=DEST/rel
        out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
        files[str(rel)]=dict(source=str(p),bytes=out.stat().st_size,sha256=sha(out),source_sha256=sha(p))
    # Freeze membership compactly, while retaining exact original byte hashes.
    for run in runs:
        p=source/'experiments'/run/'sample_ids.txt'
        if not p.exists():continue
        out=DEST/'membership'/f'{run}.sample_ids.txt.gz';out.parent.mkdir(exist_ok=True)
        out.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
        files[str(out.relative_to(DEST))]=dict(source=str(p),bytes=out.stat().st_size,sha256=sha(out),source_sha256=sha(p),encoding='gzip')
    # Preserve the exact 74 Python files used by the completed full processing.
    full=source/'experiments/p2_full_audio_postprocess_crop384_20261026'
    frozen=json.loads((full/'code_manifest.json').read_text());bindings=[]
    with tarfile.open(full/'code_snapshot.tar.gz') as archive:
        for original,h in frozen.items():
            p=Path(original)
            if p.suffix!='.py':continue  # SSH host-key file is environment data.
            name=str(p.relative_to(source));raw=archive.extractfile(name).read()
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Frozen source mismatch '+name)
            out=DEST/'production_full_snapshot/audio-driven-v5'/name
            out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
            rel=str(out.relative_to(DEST));files[rel]=dict(source=str(full/'code_snapshot.tar.gz')+'::'+name,bytes=len(raw),sha256=h)
            bindings.append(dict(original=original,backup=rel,sha256=h,current_matches=sha(p)==h))
    evidence=DEST/'evidence'
    write(evidence/'production_full_bindings.json',bindings)
    stage_bindings=[]
    historical_by_hash={}
    for code_root in (source/'experiments').glob('*/code'):
        if not code_root.parent.name.startswith(('listen','p2_features')):continue
        for p in code_root.rglob('*.py'):
            if p.is_file():historical_by_hash.setdefault(sha(p),p)
    for run in runs[:3]:
        base=source/'experiments'/run;cm=json.loads((base/'code_manifest.json').read_text())
        frozen_by_hash={sha(p):p for p in (base/'code').rglob('*') if p.is_file() and p.suffix in {'.py','.sh'}}
        for original,h in cm.items():
            p=Path(original) if Path(original).is_absolute() else source/original
            if p.suffix not in {'.py','.sh'}:continue
            choices=[p]+([frozen_by_hash[h]] if h in frozen_by_hash else [])+([historical_by_hash[h]] if h in historical_by_hash else [])+[REPO/n for n in old_index.get(h,[])]
            found=next((q for q in choices if q.is_file() and sha(q)==h),None)
            if found is None:raise RuntimeError('Missing exact execution source: '+original)
            out=DEST/'production_stage_snapshots'/run/'snapshot'/p.relative_to(project)
            out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(found,out)
            rel=str(out.relative_to(DEST));files[rel]=dict(source=str(found),bytes=out.stat().st_size,sha256=h)
            stage_bindings.append(dict(run=run,original=original,backup=rel,sha256=h,current_matches=p.exists() and sha(p)==h))
    write(evidence/'production_stage_bindings.json',stage_bindings)
    write(evidence/'dataset_references.json',external_refs)
    audit=[dict(source=str(p.relative_to(source)),sha256=sha(p),previous_exact_backups=old_index.get(sha(p),[]))
           for p in sorted(chosen) if p.is_relative_to(source) and p.suffix=='.py']
    write(evidence/'prior_code_coverage.json',dict(files=audit,checked=len(audit),previous_exact=sum(bool(r['previous_exact_backups']) for r in audit)))
    write(DEST/'BACKUP_MANIFEST.json',dict(schema='p2-final-processing-code-backup-v1',created_at=datetime.now(timezone.utc).isoformat(),
        source_project=str(source),files=files,omitted=['media','weights','checkpoints','latent payloads','full annotation tables','credentials'],
        production_frozen_python_files=len(bindings),other_production_frozen_files=len(stage_bindings),new_semantic_inventory_used=False))
    print(json.dumps(dict(files=len(files),bytes=sum(r['bytes'] for r in files.values()),python_checked=len(audit),
                         missing_previous_exact=sum(not r['previous_exact_backups'] for r in audit),frozen_files=len(bindings))))

if __name__=='__main__':main()
