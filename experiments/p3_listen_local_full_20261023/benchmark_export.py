import collections
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import listen_local_export as exporter


def main():
    source=ROOT/'data_corpora/seamless_listen_local_full';target=Path(__file__).parent/'export_benchmark';target.mkdir(exist_ok=True)
    for folder in ['results','summaries']:(target/folder).mkdir(exist_ok=True)
    names=list((source/'summaries').glob('*.json'));random.Random(23).shuffle(names);names=names[:2048]
    counts=collections.Counter();input_bytes=0
    for p in names:
        value=json.loads(p.read_text());counts.update({k:v['clips'] for k,v in value['pools'].items()})
        for folder in ['results','summaries']:
            q=target/folder/p.name
            if not q.exists():q.symlink_to(source/folder/p.name)
        input_bytes+=(source/'results'/p.name).stat().st_size
    result=exporter.export(target,dict(complete_pairs=len(names),counts=dict(counts),final=False),workers=16,shards=32)
    record=dict(sample_pairs=len(names),input_bytes=input_bytes,export=result['export'],output_bytes=sum(p.stat().st_size for p in (target/'manifests').glob('*.jsonl')))
    (Path(__file__).parent/'real_export_benchmark.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
