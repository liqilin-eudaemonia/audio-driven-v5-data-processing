"""Rebuild source coverage from every original TAR header, then isolate evaluation."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from p3_listen.common import read, write, sha, digest

SOURCE = Path('/public/0_DATA/2_Avatar/seamless_interaction')
OUT = ROOT/'data_corpora/seamless_listen_v7_full'
EXTENSIONS = {'.json', '.mp4', '.npz', '.wav'}


def inspect(path):
    path = Path(path)
    stat = path.stat()
    marker = OUT/'inventory'/(digest(str(path))+'.json')
    if marker.exists():
        old = read(marker)
        if old['bytes'] == stat.st_size and old['mtime_ns'] == stat.st_mtime_ns:
            return old
    files = defaultdict(set)
    with tarfile.open(path, 'r:') as tar:
        for member in tar:
            if member.isfile() and Path(member.name).suffix in EXTENSIONS:
                files[str(Path(member.name).with_suffix(''))].add(Path(member.name).suffix)
    relative = path.relative_to(SOURCE).parts
    complete = sorted(k for k,v in files.items() if v == EXTENSIONS)
    row = dict(archive=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
        label=relative[0], split=relative[1], extras='extras' in relative,
        complete_ids=complete, incomplete={k:sorted(v) for k,v in files.items() if v != EXTENSIONS},
        extension_counts=dict(Counter(ext for values in files.values() for ext in values)))
    write(marker,row)
    return row


def build(workers=32):
    import prepare_listen_300h as base
    paths=sorted(SOURCE.rglob('*.tar'))
    write(OUT/'source_archives.json', [str(p) for p in paths])
    records=defaultdict(list); anomalies=[]; counts=Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,row in enumerate(pool.map(inspect,paths),1):
            counts['archives']+=1
            counts['extras_archives' if row['extras'] else 'ordinary_archives']+=1
            counts['incomplete_participant_records']+=len(row['incomplete'])
            for ident in row['complete_ids']:
                if row['extras'] or '/' in ident:
                    anomalies.append(dict(reason='complete_record_requires_layout_review',archive=row['archive'],id=ident))
                    continue
                records[(row['label'],row['split'],ident)].append(row['archive'])
            if i%250==0 or i==len(paths):
                write(OUT/'inventory_progress.json',dict(completed=i,total=len(paths),counts=dict(counts)))
                print(json.dumps(dict(inventoried=i,total=len(paths))),flush=True)
    interactions=defaultdict(list)
    for (label,split,ident),archives in records.items():
        try:
            base.person_keys(ident)
        except ValueError:
            anomalies.append(dict(reason='invalid_id',id=ident,archives=archives));continue
        if len(archives)!=1:
            anomalies.append(dict(reason='duplicate_participant_archive',id=ident,archives=archives));continue
        interactions[(label,split,ident.rsplit('_',1)[0])].append(dict(
            id=ident,p=ident.rsplit('_',1)[1],archive=archives[0],label=label,split=split,complete=True))
    pairs=[]
    for (label,split,interaction),people in sorted(interactions.items()):
        if len(people)!=2:
            anomalies.append(dict(reason='not_two_complete_participants',label=label,split=split,
                                  interaction=interaction,participants=people));continue
        a,b=sorted(people,key=lambda p:p['id'])
        pairs.append(dict(interaction=interaction,label=label,split=split,a=a,b=b))
    # A repeated interaction across splits must never overwrite a result marker.
    names=Counter(p['label']+'_'+p['interaction'] for p in pairs)
    conflicts=[p for p in pairs if names[p['label']+'_'+p['interaction']]>1]
    pairs=[p for p in pairs if names[p['label']+'_'+p['interaction']]==1]
    anomalies.extend(dict(reason='interaction_repeated_across_splits',record=p) for p in conflicts)
    write(OUT/'source_pair_index.json',pairs)
    parent=list(range(len(pairs)));owners={}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,p in enumerate(pairs):
        for role in ('a','b'):
            for key in base.person_keys(p[role]['id']):
                if key in owners: parent[find(i)]=find(owners[key])
                owners[key]=i
    groups=defaultdict(list)
    for i,p in enumerate(pairs):groups[find(i)].append(p)
    rows=[];excluded=[];group_rows=[]
    for group in groups.values():
        sessions=sorted({base.person_keys(p[k]['id'])[0] for p in group for k in ('a','b')})
        people=sorted({base.person_keys(p[k]['id'])[1] for p in group for k in ('a','b')})
        gid='component_'+digest([sessions,people])[:20]
        splits={p['split'] for p in group}
        # Preserve official train; remove its linked eval from this task's evaluation.
        target='train' if 'train' in splits else 'test' if 'test' in splits else 'dev'
        group_rows.append(dict(group_id=gid,sessions=sessions,participants=people,split=target))
        for p in group:
            row=dict(p,pair=p['label']+'_'+p['interaction'],group_id=gid)
            if p['split']==target:rows.append(row)
            else:excluded.append(dict(pair=row['pair'],split=p['split'],group_id=gid,
                                      reason='evaluation_related_to_'+target))
    rows.sort(key=lambda r:digest(r['pair']))
    for a in ('train','dev','test'):
        for b in ('train','dev','test'):
            if a!=b:assert not ({p['group_id'] for p in rows if p['split']==a}&{p['group_id'] for p in rows if p['split']==b})
    counts.update({'source_pairs':len(pairs),'selected_pairs':len(rows),'anomalies':len(anomalies)})
    scope='All original TAR headers inventoried. Official train retained; related eval excluded. Remaining dev/test groups isolated; no test moved to train.'
    write(OUT/'catalog.json',dict(rows=rows,groups=group_rows,scope=scope,
        source_index=str(OUT/'source_pair_index.json'),source_index_sha256=sha(OUT/'source_pair_index.json'),
        original_root=str(SOURCE),excluded_pairs=dict(Counter(r['reason'] for r in excluded))))
    write(OUT/'source_exclusions.json',dict(evaluation_excluded=excluded,anomalies=anomalies))
    summary=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),counts=dict(counts),
        source_splits=dict(Counter(p['split'] for p in pairs)),selected_splits=dict(Counter(p['split'] for p in rows)),
        anomaly_reasons=dict(Counter(p['reason'] for p in anomalies)),
        scope=scope,index_sha256=sha(OUT/'source_pair_index.json'),inventory_complete=True)
    write(OUT/'source_coverage.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=32)
    build(parser.parse_args().workers)
