"""Diagnose first-frame detector coverage; never reads future listener frames."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import p2_l2l_gpavatar as gp
import torch,torchvision
gp.guard();engine=gp.tracker()
p=gp.read(gp.WORK/'protocol.json')
sample=min(p['samples'],key=lambda s:s['frames'])
image=torchvision.io.read_image(sample['image']).float()/255.
image=torchvision.transforms.functional.resize(image,[512,512],antialias=True)
records=[]
folder=gp.WORK/'reference_probe';folder.mkdir(exist_ok=True)
for padding in [0,64,128]:
    a=image
    if padding:
        a=torch.nn.functional.pad(a,[padding]*4)
        a=torchvision.transforms.functional.resize(a,[512,512],antialias=True)
    for matte in [False,True]:
        b=engine.matting_engine(a,return_type='matting',background_rgb=0.).cpu() if matte else a
        mica=engine.mica_engine.process_frame(b*255.)
        emoca=engine.emoca_engine.process_frame(b*255.)
        r=dict(padding=padding,matting=matte,mica_detected=mica is not None,emoca_detected=emoca is not None)
        records.append(r);print(r,flush=True)
        torchvision.utils.save_image(b,folder/f'padding{padding}_matte{int(matte)}.png')
gp.save(gp.WORK/'REFERENCE_PROBE.json',dict(id=sample['id'],source_frames=1,records=records))
