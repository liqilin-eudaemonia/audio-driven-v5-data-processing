"""Create display-only Seamless GT MP4s from the verified scoring RGB cache."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/common_method_comparison_20261106'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    protocol=json.loads((ROOT/'experiments/p2_compare_seamless200_20261106/dystream/protocol.json').read_text())
    cache=Path(protocol['metrics_decoded_gt_cache']);index={}
    for path in cache.glob('*/*/metadata.json'):
        meta=json.loads(path.read_text());reference=meta['identity']['reference']
        index.setdefault(reference['sample_id'],[]).append((path,meta))
    def one(item):
        uid=item['id'];dest=WORK/'gt_preview/seamless'/(uid+'.mp4');receipt=dest.with_suffix('.json')
        if receipt.exists():
            r=json.loads(receipt.read_text());assert sha(dest)==r['preview_sha256'];return r
        matches=[(path,m) for path,m in index[uid] if m['identity']['reference']==item['reference']['gt_reference']]
        assert matches,uid
        path,meta=matches[0];frames=np.load(path.parent/'frames.npy',mmap_mode='r',allow_pickle=False)
        assert len(frames)==item['frames'] and frames.shape[1:]==(512,512,3) and frames.dtype==np.uint8
        rgb_sha=hashlib.sha256(memoryview(frames)).hexdigest();assert rgb_sha==meta['audit']['rgb_sha256']
        dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.stem+'.tmp.mp4')
        cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','512x512','-r',str(meta['fps']),
             '-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',
             '-threads','1','-movflags','+faststart',str(temp)]
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        _,error=p.communicate(input=memoryview(frames).cast('B'));assert p.returncode==0,error.decode()
        temp.replace(dest);r=dict(id=uid,frames=len(frames),fps=meta['fps'],source_cache_metadata=str(path),
            source_rgb_sha256=rgb_sha,preview_sha256=sha(dest),display_only=True,scoring_inputs_unchanged=True)
        receipt.write_text(json.dumps(r,indent=2)+'\n');return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(one,protocol['samples']))
    (WORK/'GT_PREVIEW_COMPLETE.json').write_text(json.dumps(dict(pairs=200,all_rgb_hashes_verified=True,records=results),indent=2)+'\n')
    print('PASS: 200 display-only GT videos, exact frozen scoring RGB verified')


if __name__=='__main__':main()
