"""Opt-in event-presence stage; existing running scorers are never changed."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from .contracts import CLASSES, read, rows, save, sha
from .media import ROOT, bind, verify


def candidates(work):
    work=Path(work);contract,_=verify(work);p=contract['policy']
    unique={m['media_id']:m for row in rows(work/'prepared_pairs.jsonl') for m in row['prepared_media'].values() if m.get('visual25')}
    for media in unique.values():
        folder=Path(media['folder']);output=folder/'candidates.json'
        if output.exists():continue
        command=[p['cue_python'],str(ROOT/'scripts/p2_event_presence.py'),'_candidate','--media',str(folder/'media.json'),
                 '--fer-python',p['fer_python']]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (folder/'candidates.log').open('w') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode or not output.exists():
            save(output,dict(media_id=media['media_id'],frame_coverage=0.,cues=[],candidate_only=True,
                             error='annotation_candidate_subprocess_failed',exit_code=result.returncode))
    save(work/'CANDIDATES_COMPLETE.json',dict(media=len(unique),failed=sum('error' in read(Path(m['folder'])/'candidates.json') for m in unique.values())))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('bind');b.add_argument('--manifest',type=Path,action='append',required=True);b.add_argument('--work',type=Path,required=True)
    b.add_argument('--config',type=Path,default=ROOT/'configs/p2_event_presence_v1.json');b.add_argument('--method',default='model');b.add_argument('--draw',default='0')
    b.add_argument('--visual-only',action='store_true');b.add_argument('--self-check',action='store_true')
    for cmd in ['prepare','candidates','infer','score','compare','review','run']:
        s=sub.add_parser(cmd);s.add_argument('--work',type=Path,required=True)
        if cmd in ['score','compare']:s.add_argument('--human-labels',type=Path)
    c=sub.add_parser('_candidate');c.add_argument('--media',type=Path,required=True);c.add_argument('--fer-python',required=True)
    args=p.parse_args()
    if args.command=='bind':
        bind(args.manifest,args.work,args.config,args.method,args.draw,not args.visual_only,args.self_check)
    elif args.command=='prepare':
        from .media import prepare
        prepare(args.work)
    elif args.command=='candidates':candidates(args.work)
    elif args.command=='infer':
        from .inference import infer
        infer(args.work)
    elif args.command in ['score','compare']:
        from .reporting import report
        result=report(args.work,args.human_labels)
        if result['status']=='partial':raise SystemExit(2)
    elif args.command=='review':
        from .reporting import review
        review(args.work)
    elif args.command=='_candidate':
        from .candidates import measure
        measure(read(args.media),args.fer_python)
    elif args.command=='run':
        contract,_=verify(args.work);cfg=contract['policy'];entry=ROOT/'scripts/p2_event_presence.py';args.work=args.work.resolve()
        for stage,python in [('prepare',cfg['media_python']),('candidates',sys.executable),('infer',cfg['inference_python']),('score',sys.executable)]:
            save(args.work/'RUN_STATUS.json',dict(stage=stage,status='running',pid=os.getpid()))
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            runtime_lib=Path(python).parent.parent/'lib'
            if runtime_lib.is_dir():env['LD_LIBRARY_PATH']=str(runtime_lib)+':'+env.get('LD_LIBRARY_PATH','')
            if stage=='infer':
                env['CUDA_VISIBLE_DEVICES']=str(cfg['physical_gpu'])
                env['LD_LIBRARY_PATH']=str(Path(python).parent.parent/'lib')+':'+env.get('LD_LIBRARY_PATH','')
            else:env['CUDA_VISIBLE_DEVICES']=''
            with (args.work/(stage+'.log')).open('a') as log:
                code=subprocess.run([python,str(entry),stage,'--work',str(args.work)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:
                if stage=='score' and code==2 and (args.work/'summary.json').exists():
                    save(args.work/'RUN_STATUS.json',dict(stage='finished',status='execution_complete_partial',
                        summary_sha256=sha(args.work/'summary.json'),pilot_only=True,human_validation_pending=True,formal_benchmark_ready=False))
                    raise SystemExit(2)
                save(args.work/'RUN_STATUS.json',dict(stage=stage,status='failed',exit_code=code));raise SystemExit(code)
        save(args.work/'RUN_STATUS.json',dict(stage='finished',status='execution_complete',
            summary_sha256=sha(args.work/'summary.json'),pilot_only=True,human_validation_pending=True,formal_benchmark_ready=False))


if __name__=='__main__':main()
