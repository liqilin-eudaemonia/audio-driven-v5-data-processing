"""Verify exactly-once queue claims across the three machines' shared filesystem."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
import run_listen_v7_fast as fast

ROOT=fast.bulk.ROOT/'experiments/p3_listen_v7_performance/shared_queue_check'


def main(stage,node):
    fast.SCHED=ROOT
    queue=[f'probe_{i:04d}' for i in range(600)]
    if stage=='init':
        assert not (ROOT/'state.json').exists()
        fast.bulk.write(ROOT/'state.json',dict(cursor=0,total=len(queue)))
    elif stage=='claim':
        def consume(index):
            values=[]
            while True:
                pair=fast.claim(queue,node*8+index)
                if pair is None:return values
                values.append(pair);time.sleep(.01)
        with ThreadPoolExecutor(max_workers=8) as pool:groups=list(pool.map(consume,range(8)))
        fast.bulk.write(ROOT/f'node_{node}.json',dict(claims=[p for g in groups for p in g]))
    else:
        values=[pair for n in range(3) for pair in fast.bulk.read(ROOT/f'node_{n}.json')['claims']]
        assert len(values)==len(set(values))==len(queue)
        assert set(values)==set(queue)
        assert fast.bulk.read(ROOT/'state.json')['cursor']==len(queue)
        fast.bulk.write(ROOT/'validation.json',dict(exactly_once=True,machines=3,concurrent_threads=24,claims=len(values)))
        print('600 claims, no duplicates or omissions across three machines',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','claim','verify']);p.add_argument('--node',type=int,default=0)
    a=p.parse_args();main(a.stage,a.node)
