#!/usr/bin/env python3
"""Full J caption type/count statistics, explicit v3 and 32 single-GPU workers."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from listen_benchmark.j_annotation import read,write
from listen_richness.production import contract,tokenized


def initialize(work,upstream):
    work.mkdir(parents=True,exist_ok=True);old=read(upstream/'config.json');hardware=read(work/'hardware_preflight.json')
    nodes={}
    for name,n in old['nodes'].items():
        uuids={line.split(',')[0].strip():line.split(',')[1].strip() for line in hardware[name]['gpus'].strip().splitlines()}
        nodes[name]=dict(host=n['host'],hostname=n['hostname'],gpu_ids=n['gpu_ids'],gpu_uuids=uuids,
                         ssh_key=n.get('ssh_key','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'))
    paths=[str(p.relative_to(ROOT)) for p in (ROOT/'src/listen_richness').glob('*.py')]
    paths+=['scripts/listen_richness_cluster.py','scripts/listen_richness_cluster_control.py',
            'scripts/analyze_listen_richness_pilot.py','src/listen_benchmark/j_annotation.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    profiles={name:dict(max_num_seqs=seq,max_num_batched_tokens=tok,reservoir_chunks=2)
              for name,seq,tok in [('A128',128,32768),('B256',256,32768),('C256',256,65536),('D512',512,65536)]}
    c=dict(schema='richness-cluster-v1',project_root=str(ROOT),upstream=str(upstream),nodes=nodes,
           contract=contract(),code_hashes=hashes,profiles=profiles,chunk_size=512,prepare_workers=4,
           selected_profile=None,expected_upstream_clips=1151805,expected_upstream_sources=60869)
    p=work/'config.json'
    if p.exists():
        prior=read(p)
        if any(prior[k]!=c[k] for k in ('contract','upstream','code_hashes')):raise ValueError('Existing production contract differs')
    else:write(p,c)
    print(json.dumps(dict(work=str(work),contract=contract(),profiles=profiles)))


def benchmark_inputs(work,count=1024):
    from transformers import AutoTokenizer
    from listen_richness.prepare_cluster import source_records
    from listen_richness.worker_cluster import atomic_jsonl
    c=read(work/'config.json');upstream=Path(c['upstream']);tokenizer=AutoTokenizer.from_pretrained(contract()['model'],local_files_only=True)
    ids=[]
    with (upstream/'ready.jsonl').open() as f:
        for line in f:ids.append(json.loads(line)['job_id'])
    ids=list(dict.fromkeys(ids));random.Random(20261031).shuffle(ids)
    selected=[];coverage={};sources=0
    with ThreadPoolExecutor(4) as pool:
        for start in range(0,len(ids),16):
            ready=[jid for jid in ids[start:start+16] if (upstream/'caption/completed'/f'{jid}.json').exists()]
            for records in pool.map(lambda j:source_records(upstream,j),ready):
                sources+=1
                for r in records:
                    coverage[r['status']]=coverage.get(r['status'],0)+1
                    if r['status']=='caption_ready':selected.append(r['snapshot'])
            write(work/'benchmark_preparation.json',dict(at=time.time(),sources=sources,coverage=coverage,ready=len(selected)))
            if len(selected)>=count*2:break
    if len(selected)<count:raise ValueError('Insufficient benchmark inputs')
    random.Random(20261032).shuffle(selected);selected=selected[:count]
    requests=[tokenized(s,tokenizer) for s in selected];requests.sort(key=lambda r:(-r['prompt_tokens'],r['sample_id']))
    path=work/'benchmark_inputs.jsonl';sha=atomic_jsonl(path,requests)
    lengths=sorted(r['prompt_tokens'] for r in requests)
    write(work/'benchmark_input_manifest.json',dict(at=time.time(),path=str(path),sha256=sha,count=count,
        unique_ids=len({r['sample_id'] for r in requests}),source_pool=sources,coverage=coverage,seed=20261031,
        prompt_tokens=dict(min=min(lengths),median=lengths[len(lengths)//2],max=max(lengths),mean=sum(lengths)/count),
        contract=contract(),real_caption_only=True,same_inputs_for_all_profiles=True))
    print(json.dumps(dict(count=count,path=str(path),sha256=sha)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['init','benchmark-inputs','prepare','worker','merge'])
    p.add_argument('--work',type=Path,required=True);p.add_argument('--upstream',type=Path)
    p.add_argument('--node');p.add_argument('--gpu',type=int);p.add_argument('--profile');p.add_argument('--benchmark-input');p.add_argument('--count',type=int,default=1024);p.add_argument('--once',action='store_true')
    a=p.parse_args();work=a.work.resolve()
    if a.command=='init':initialize(work,a.upstream.resolve())
    elif a.command=='benchmark-inputs':benchmark_inputs(work,a.count)
    elif a.command=='prepare':
        from listen_richness.prepare_cluster import prepare
        prepare(work,a.once)
    elif a.command=='merge':
        from listen_richness.merge_cluster import merge
        print(json.dumps(merge(work)))
    else:
        from listen_richness.worker_cluster import worker
        c=read(work/'config.json');name=a.profile or c['selected_profile']
        worker(work,a.node,a.gpu,c['profiles'][name],a.benchmark_input)


if __name__=='__main__':main()
