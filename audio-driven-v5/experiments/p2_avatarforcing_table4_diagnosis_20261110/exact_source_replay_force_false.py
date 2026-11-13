"""Execute the sealed generator unchanged for one flagged case, tracing export."""
import hashlib
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts')]
import p2_avatarforcing_table4_generate as gen


def main():
    import torch
    original=gen.verify();source=Path(gen.__file__);run=gen.WORK
    uid='UCA1A5GqCdQ_001213_001223.1';sample=next(s for s in original['samples'] if s['id']==uid)
    job=WORK/'exact_source_replay_force_false';job.mkdir(exist_ok=True)
    shutil.copy2(run/'protocol.json',job/'protocol.json')
    if not (job/'inputs').exists():(job/'inputs').symlink_to(run/'inputs',target_is_directory=True)
    gen.WORK=job;gen.verify=lambda:dict(original,samples=[sample])
    lines=source.read_text().splitlines()
    export_line=next(i+1 for i,l in enumerate(lines) if "old.export_rgb(output/'generated.mp4'" in l)
    seed_line=next(i+1 for i,l in enumerate(lines) if "torch.manual_seed(p['seed'])" in l)
    after_line=next(i+1 for i,l in enumerate(lines) if 'assert prediction.shape==' in l)
    state=dict(collect=False,noise=[]);original_randn=torch.randn
    def captured(*args,**kwargs):
        a=original_randn(*args,**kwargs)
        if state['collect']:state['noise'].append(hashlib.sha256(a.float().cpu().numpy().tobytes()).hexdigest())
        return a
    def tracer(frame,event,arg):
        if frame.f_code.co_filename!=str(source):return None
        if event=='line' and frame.f_lineno==seed_line:
            torch.backends.cudnn.benchmark=False
            state['collect']=True
        if event=='line' and frame.f_lineno==after_line:state['collect']=False
        if event=='line' and frame.f_lineno==export_line:
            d=frame.f_locals
            gen.save(WORK/'EXACT_SOURCE_EXPORT_TRACE_FORCE_FALSE.json',dict(id=uid,minimum=float(d['lo']),maximum=float(d['hi']),
                raw_mean=float(d['prediction'].mean()),benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,noise_sha256=state['noise'],
                source_sha256=gen.sha(source),wrapper_sha256=gen.sha(__file__)))
        return tracer
    torch.randn=captured;sys.settrace(tracer)
    try:gen.generate()
    finally:sys.settrace(None);torch.randn=original_randn
    gen.save(WORK/'EXACT_SOURCE_REPLAY_FORCE_FALSE_COMPLETE.json',dict(passed_execution=True,source_unchanged=True,
        selected_one_clip=uid,output=str(job/'generated'/uid/'generated.mp4'),trace=str(WORK/'EXACT_SOURCE_EXPORT_TRACE_FORCE_FALSE.json')))


if __name__=='__main__':main()
