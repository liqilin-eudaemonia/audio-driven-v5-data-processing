"""Official LM Listener metrics with lazy, overlap-correct GT frame indexing."""
import ast
from bisect import bisect_right
from collections import defaultdict
from functools import lru_cache
import hashlib
import heapq
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'evaluation/lm-listener/evaluate_listener.py'
SOURCE_SHA256='75b76a848ecb0ea18b0fc2ef17c2d90374ce192f0fce8d65433c149f07e1fddc'


class FrameLookup:
    """Read-only scorer lookup; later segment rows win, exactly as dict assignment."""
    def __init__(self, segments, role):
        self.segments=segments;self.role=role;self.index={}
        groups=defaultdict(list)
        for row,s in enumerate(segments):groups[s['fname']].append(row)
        for fname,rows in groups.items():
            events=defaultdict(lambda:[[],[]])
            for row in rows:
                s=segments[row];start=int(s['split_start_frame']);end=int(s['split_end_frame'])
                if end<=start:continue
                events[start][0].append(row);events[end][1].append(row)
            points=sorted(events);active=set();heap=[];spans=[]
            for i,point in enumerate(points[:-1]):
                added,removed=events[point]
                active.difference_update(removed)
                for row in added:active.add(row);heapq.heappush(heap,-row)
                while heap and -heap[0] not in active:heapq.heappop(heap)
                if heap:spans.append((point,points[i+1],-heap[0]))
            self.index[fname]=([x[0] for x in spans],spans)
        self.block=lru_cache(maxsize=8)(self._block)

    def _locate(self,key):
        try:
            fname,frame=key.rsplit('_',1);frame=int(frame)
            starts,spans=self.index[fname];i=bisect_right(starts,frame)-1
            if i<0 or frame>=spans[i][1]:raise KeyError(key)
            return spans[i][2],frame
        except (ValueError,TypeError,KeyError):raise KeyError(key) from None

    def __contains__(self,key):
        try:self._locate(key);return True
        except KeyError:return False

    def _block(self,row):
        import numpy as np
        s=self.segments[row]
        return np.concatenate([s[self.role+'_'+name] for name in ('exp','pose','shape')],axis=1)

    def __getitem__(self,key):
        row,frame=self._locate(key)
        return self.block(row)[frame-int(self.segments[row]['split_start_frame'])]


def transformed_tree():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256, 'Official scorer changed; review adapter'
    tree=ast.parse(SOURCE.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    new=[];replaced=[];skip=False
    for node in main.body:
        if skip:
            assert isinstance(node,ast.For)
            skip=False;continue
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('frame_map','speaker_map'):
                assert isinstance(node.value,ast.Dict) and not node.value.keys
                role='p0' if name=='frame_map' else 'p1'
                node.value=ast.parse(f"FrameLookup(segments, '{role}')",mode='eval').body
                replaced.append(name);skip=True
        new.append(node)
    assert replaced==['frame_map','speaker_map'] and not skip
    main.body=new
    # This unused scan used to traverse every frame key once per prediction.
    class RemoveUnused(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            if isinstance(node.targets[0],ast.Name) and node.targets[0].id=='valid_keys':
                self.count+=1;return None
            return node
    transform=RemoveUnused();tree=transform.visit(tree);assert transform.count==1
    return ast.fix_missing_locations(tree)


def main():
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
    base=SOURCE.parent
    os.environ.setdefault('DECA_PATH',str(base/'third_party/DECA'))
    sys.path[:0]=[str(base),str(base/'third_party/emoca')]
    env=dict(__name__='__main__',__file__=str(SOURCE),FrameLookup=FrameLookup)
    exec(compile(transformed_tree(),str(SOURCE)+':lazy-index','exec'),env)


if __name__=='__main__':main()
