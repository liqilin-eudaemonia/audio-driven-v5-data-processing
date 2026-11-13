"""CPU per-clip PSNR/SSIM diagnostics with the frozen official implementation."""
import csv
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_avatarforcing import read,save,sha

WORK=ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'
BASE=ROOT/'experiments/p2_vico256_combined8000_20261106'


def run():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    path=ROOT/'evaluation/vico_challenge_baseline/evaluations/compute_base_metrics.py'
    spec=importlib.util.spec_from_file_location('official_clip_images',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=read(WORK/'protocol.json');items=read(WORK/'shards/00.json')
    oldname=read(BASE/'protocol.json')['arms'][0]['name']
    name=p['arms'][0]['name'];folder=WORK/'paired_image';folder.mkdir(exist_ok=True)
    save(WORK/'PAIRED_IMAGE_RUNTIME.json',dict(pid=os.getpid(),at=time.time()))
    deadline=time.monotonic()+4*3600
    while True:
        for item in items:
            uid=item['id'];target=folder/(uid+'.json')
            if target.exists() or not (WORK/'conversion'/(uid+'.verified.json')).exists():continue
            paths=dict(gt=Path(p['datasets']['vico']['gt_dir'])/(uid+'.listener.mp4'),
                base=BASE/'scoring'/oldname/(uid+'.listener.mp4'),sft=WORK/'scoring'/name/(uid+'.listener.mp4'))
            frames={k:torch.from_numpy(module.read_mp4(str(q),True,False,True)).float()/255. for k,q in paths.items()}
            assert all(x.shape==frames['gt'].shape for x in frames.values())
            values={}
            with torch.no_grad():
                for method in ['base','sft']:
                    values[method]=dict(psnr=float(module.psnr(frames[method],frames['gt'],reduction='none').double().mean()),
                        ssim=float(module.ssim(frames[method],frames['gt'],data_range=1.,reduction='none').double().mean()))
            save(target,dict(id=uid,frames=len(frames['gt']),values=values,
                delta={k:values['sft'][k]-values['base'][k] for k in ['psnr','ssim']},
                source_sha256={k:sha(q) for k,q in paths.items()},metric_code_sha256=sha(path)))
            del frames
        rows=[read(folder/(x['id']+'.json')) for x in items if (folder/(x['id']+'.json')).exists()]
        frames=sum(x['frames'] for x in rows)
        means={method:{metric:sum(x['values'][method][metric]*x['frames'] for x in rows)/frames for metric in ['psnr','ssim']} for method in ['base','sft']} if rows else {}
        save(WORK/'PAIRED_IMAGE_SCORES.json',dict(complete=len(rows)==43,clips=len(rows),frames=frames,
            frame_weighted=means,per_clip=rows,at=time.time(),
            scope='When incomplete, longest-first completed subset; not a representative final ranking'))
        with (WORK/'paired_image_per_clip.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['id','frames','base_psnr','sft_psnr','delta_psnr','base_ssim','sft_ssim','delta_ssim'])
            for row in rows:writer.writerow([row['id'],row['frames'],*[row['values'][m][k] if m!='delta' else row['delta'][k] for k in ['psnr','ssim'] for m in ['base','sft','delta']]])
        if len(rows)==43:break
        if time.monotonic()>deadline:raise TimeoutError('Waiting for remaining videos; no GPU process restarted')
        time.sleep(15)
    reference=read(BASE/'RESULTS.json')['metrics']
    assert all(abs(means['base'][k]-reference[k])<1e-10 for k in ['psnr','ssim'])
    save(WORK/'PAIRED_IMAGE_AUDIT.json',dict(passed=True,clips=43,
        baseline_global_max_error=max(abs(means['base'][k]-reference[k]) for k in ['psnr','ssim'])))


if __name__=='__main__':run()
