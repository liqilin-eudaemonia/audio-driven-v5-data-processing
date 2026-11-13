"""Produce minimal, explicit derivatives of the frozen native training loops."""
from pathlib import Path
import ast, hashlib, json

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]

def build():
    changes={}
    for arm, package, world in [('full','p2_ltx_all_lora',8),('noevent','p2_ltx_all_lora_noevent',24)]:
        source=ROOT/'src'/package/'training.py'
        original=source.read_text(); text=original; edits=[]
        def replace(old,new,count=1):
            nonlocal text
            if text.count(old)!=count:raise ValueError('Native code drift: '+repr(old))
            text=text.replace(old,new); edits.append(dict(old=old,new=new,count=count))
        replace("    from .schedule import Schedule\n", "    # Schedule is injected by the separately hashed split runtime.\n")
        replace("    if t['global_batch']%world:raise ValueError('World must divide fixed global batch')",
                f"    if t['global_batch']!=31 or world!={world}:raise ValueError('Require approved batch31/world{world}')")
        replace("loss,event_parts,event=objective(values,c,counts,len(chosen),world)",
                "loss,event_parts,event=objective(values,c,counts,31,1)")
        replace("/len(chosen)", "/t['global_batch']",3)
        replace("record[k]=v if k.startswith('event') else v/world", "record[k]=v")
        replace("        if execution['optimizer_state_cpu_offload']:move_adam_moments(optimizer)\n        def save(step):",
                "        if execution['optimizer_state_cpu_offload']:raise ValueError('Split requires existing GPU Adam')\n"
                "        split_canary(c=c,model=model,optimizer=optimizer,dataset=dataset,strategy=strategy,sampler=sampler,\n"
                "            pos=pos,effective=effective,names=names,device=device,local=local,rank=rank,world=world,\n"
                "            resume_path=resume_path,identity=identity,start=start)\n"
                "        def save(step):")
        ast.parse(text)
        destination=WORK/f'training_{arm}.py';destination.write_text(text)
        changes[arm]=dict(native=str(source),native_sha256=hashlib.sha256(original.encode()).hexdigest(),
                          output=str(destination),output_sha256=hashlib.sha256(text.encode()).hexdigest(),edits=edits)
    (WORK/'TRAINING_DIFF.json').write_text(json.dumps(changes,indent=2)+'\n')
    return changes

if __name__=='__main__':build()
