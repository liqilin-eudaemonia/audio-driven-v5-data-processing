"""Regenerate only the four affected diagnostic clips with the original6000/SFT2752 system."""
import sys,os,time,json,copy
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha
BASE=ROOT/'experiments/seamless_upright_eval_fix_20261112'
WORK=BASE/'generation6000';PARENT=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'

def prepare():
    from p2_combined_whisper_eval import link
    migration=read(BASE/'prepared_200/MIGRATION.json');ids={x['id'] for x in migration['changed']}
    refs={x['id']:x for x in map(json.loads,(BASE/'prepared_200/references.jsonl').open())}
    items=[dict(x,reference=refs[x['id']]) for x in read(PARENT/'shared_items.json') if x['dataset']=='seamless' and x['id'] in ids]
    assert len(items)==len(ids)==4
    WORK.mkdir(exist_ok=True);save(WORK/'shared_items.json',items)
    p=copy.deepcopy(read(PARENT/'protocol.json'));p.update(shared_items_manifest=str(WORK/'shared_items.json'),shared_items_sha256=sha(WORK/'shared_items.json'),total_samples=4,total_outputs=4,workers=1,metrics_gt_cache=str(WORK/'gt_cache'),metrics_decoded_gt_cache=str(WORK/'decoded_gt'),repair_parent=str(PARENT),repair_source=str(BASE/'prepared_200'),repair_source_sha256=sha(BASE/'prepared_200/references.jsonl'))
    p['datasets']={'seamless':dict(prepared=str(BASE/'prepared_200'),references_sha256=p['repair_source_sha256'],count=200,face_padding=0)}
    p['repair_code']={str(f):sha(f) for f in [Path(__file__),ROOT.parent/'OmniResponse/evaluation/seamless/reader.py',ROOT/'scripts/j_upright_response_sft_data.py']}
    save(WORK/'protocol.json',p)
    link(PARENT/'negative/00',WORK/'negative/00')
    return p,items

def run():
    import torch
    from p2_testsets_eval import runtime as rt
    from p2_testsets_eval import whisper_runtime as wr
    import p2_ltx6000_latest_sft_dual200_20261107 as old
    p=read(WORK/'protocol.json');items=read(WORK/'shared_items.json')
    for f,h in p['repair_code'].items():assert sha(f)==h
    assert sha(BASE/'prepared_200/references.jsonl')==p['repair_source_sha256']
    torch.set_num_threads(4);torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info();assert free>90*2**30,'Insufficient free memory for original generator'
    torch.cuda.set_per_process_memory_fraction(min(120*2**30,free-6*2**30)/total,0)
    original_conditioning=rt.conditioning
    # Existing runtime retains the checkpoint-pinned network and SFT loader;
    # replace only its historical-media cache adapter with the actual dataset reader.
    with old.runtime(WORK),patch('p2_testsets_eval.runtime.conditioning',original_conditioning):
        save(WORK/'status.json',dict(stage='source_readback',at=time.time(),pid=os.getpid(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        from evaluation.seamless.reader import decode_reference,load_listener_audio
        from PIL import Image
        import numpy as np, soundfile as sf
        audits=[]
        for x in items:
            folder=WORK/'conditions'/x['key'];raw=rt.conditioning(x,folder)
            frames,fps,audit=decode_reference(x['reference']['gt_reference'])
            assert np.array_equal(np.asarray(raw['image']),frames[0])
            gt=WORK/'gt'/x['id'];gt.mkdir(parents=True,exist_ok=True);Image.fromarray(frames[0]).save(gt/'first_frame.png')
            wave,sr,aa=load_listener_audio(x['reference']['gt_reference']);sf.write(gt/'listener.wav',wave,sr,subtype='FLOAT')
            # Lossless GT export, same helper as event preparation.
            import subprocess
            dest=gt/'listener.mkv'
            if not dest.exists():
                cmd=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','512x512','-r','25','-i','pipe:0','-an','-c:v','ffv1','-threads','2',str(dest)]
                subprocess.run(cmd,input=frames.tobytes(),check=True)
            save(gt/'audit.json',dict(video=audit,audio=aa));audits.append(dict(id=x['id'],first_frame_equals_gt=True,frames=len(frames),source=audit['video_transform']))
        save(WORK/'SOURCE_READBACK.json',dict(success=True,samples=audits))
        for stage,fn in [('whisper',lambda:wr.cache(WORK,items,0,p)),('qwen_gemma',lambda:rt.cache(WORK,items,0,p)),('render',lambda:old.render(WORK,items,0,p)),('decode',lambda:rt.decode(WORK,items,0,p,p['arms']))]:
            save(WORK/'status.json',dict(stage=stage,at=time.time(),pid=os.getpid()));fn()
        assert all(rt.ready(WORK/'generated'/p['arms'][0]['name']/x['key']) for x in items)
        save(WORK/'COMPLETE.json',dict(success=True,regenerated=4,at=time.time(),original_checkpoint=p['arms'][0],generation=p['generation'],seed=p['seed'],conditions_recomputed=True,old_outputs_rotated=False))
        save(WORK/'status.json',dict(stage='complete',at=time.time()))

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        try:run()
        except BaseException as exc:save(WORK/'status.json',dict(stage='failed',error=repr(exc),at=time.time()));raise
