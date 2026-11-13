"""Pinned original RGB8 scoring; full586 feature pooling, resumable25-clip blocks."""
import argparse,time,subprocess
from runtime import *
sys.path.insert(0,'/private/liqi_workspace/.cache/paired_video_metrics/python')
os.environ['PYTHONPATH']='/private/liqi_workspace/.cache/paired_video_metrics/python'

def destination(method,index):return WORK/'metrics'/method/'responsenet'/f'{index:02d}'
def receipt_paths(base,receipt):
    return {n:Path(receipt.get('paths',{}).get(n,base/n)) for n in ['summary.json','contract.json','pairs.jsonl','features.npz','UNAVAILABLE.json']}

@lru_cache(maxsize=1)
def scoring_device():return device()

def score(method,index,in_process=False):
    ss=groups()[index];assert 1<=len(ss)<=25 and all(valid(method,s) for s in ss)
    base=destination(method,index);base.mkdir(parents=True,exist_ok=True)
    with (base/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (base/'SUCCESS.json').exists():return
        out=base/f'attempt_{len(list(base.glob("attempt_*"))):04d}';out.mkdir()
        manifest=[];unavailable=[]
        for s in ss:
            r=read(folder(method,s)/'complete.json');video=r['score_path']
            if video is None:
                assert r.get('official_tracker_no_identity')
                unavailable.append(dict(id=s['id'],identity='listener',metrics={},errors={'generation':r['reason']}));continue
            ref=s['evaluation_reference'];manifest.append(dict(id=s['id'],dataset='responsenet-j-paired-evaluation-v1',identity='listener',gt=ref['gt'],generated=video,gt_resize=[512,512],gt_resize_method='pil_bicubic',source_start=ref['source_start'],source_end=ref['source_end'],listening_intervals=ref['listening_intervals']))
        save(out/'UNAVAILABLE.json',unavailable)
        names=['UNAVAILABLE.json'];code=0
        if manifest:
            scoring_device();spec=read(WORK/'protocol.json')['sources']['responsenet']
            path=out/'manifest.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in manifest))
            cmd=[sys.executable,str(WORK/'metric_launcher.py'),'--manifest',str(path),'--assets',spec['metric_assets'],'--output',str(out),'--device','cuda:0','--threads','2','--batch-size','64','--face-padding','64','--feature-output',str(out/'features.npz'),'--allow-partial','--face-device','cuda:0','--emotion-device','cuda:0','--pixel-workers','4','--gt-cache',spec['metrics_gt_cache']]
            with (out/'video.log').open('w') as log:
                if in_process:
                    from contextlib import redirect_stdout,redirect_stderr
                    from metric_launcher import run_pipeline
                    with redirect_stdout(log),redirect_stderr(log):code=run_pipeline(cmd[2:],resident=True)
                else:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            assert code==0,('Scoring execution failed',code,str(out/'video.log'))
            summary=read(out/'summary.json');assert not summary['initialization_errors'],summary
            actual=[json.loads(x) for x in (out/'pairs.jsonl').read_text().splitlines()]
            assert {r['id'] for r in actual}=={r['id'] for r in manifest}
            known={'emotion':lambda e:e=='No common valid emotion frames','delta_resp':lambda e:e=='ValueError: No measurable listening phases on common valid adjacent frames','fvd':lambda e:e.startswith('ValueError: Short clip: ') and e.endswith('; no padding')}
            errors=[(r['id'],k,e) for r in actual for k,e in r['errors'].items() if k not in known or not known[k](e)]
            assert not errors,('Technical errors; no false partial-success receipt',errors)
            contract=read(out/'contract.json');assert contract['code']==read(WORK/'METRIC_CODE.json')['expected']
            if in_process:
                contract['execution']['resident_backend_cache']=dict(enabled=True,process_pid=os.getpid(),adapter_sha256=sha(WORK/'metric_launcher.py'))
                save(out/'contract.json',contract)
            names+=['summary.json','contract.json','pairs.jsonl','features.npz']
        save(base/'SUCCESS.json',dict(clips=len(ss),generated=len(manifest),unavailable=len(unavailable),returncode=code,paths={n:str(out/n) for n in names},files={str(out/n):sha(out/n) for n in names},at=time.time()))
        print(method,index,'scored',len(manifest),'unavailable',len(unavailable),flush=True)

def merge(method):
    import numpy as np
    activate_readers()
    from evaluation.video.paired_metrics.run import summarize,METRICS
    rows=[];contracts=[];dist={k:[[],[]] for k in ['fid','fvd']};coverage={k:0 for k in dist}
    for i in range(len(groups())):
        base=destination(method,i);r=read(base/'SUCCESS.json')
        for p,h in r['files'].items():assert verified_file(p,h)
        pp=receipt_paths(base,r)
        if pp['UNAVAILABLE.json'].exists():rows.extend(read(pp['UNAVAILABLE.json']))
        if r.get('generated',25)==0:continue
        summary=read(pp['summary.json']);contract=read(pp['contract.json']);assert contract['code']==read(WORK/'METRIC_CODE.json')['expected'];contracts.append(contract)
        rows.extend(json.loads(x) for x in pp['pairs.jsonl'].read_text().splitlines())
        with np.load(pp['features.npz']) as a:
            for k in dist:
                if a[k+'_gt'].size:dist[k][0].append(a[k+'_gt']);dist[k][1].append(a[k+'_generated'])
                coverage[k]+=summary['coverage'][k]['pairs']
    assert len(rows)==len({r['id'] for r in rows})==TOTAL
    assert {r['id'] for r in rows}=={s['id'] for s in selected()}
    for c in contracts:assert c['code']==contracts[0]['code'] and c['assets']==contracts[0]['assets']
    r=summarize(rows,dist,set(METRICS),'omniresponse-paired-video-v1',['responsenet-j-paired-evaluation-v1'],TOTAL,{})
    for k,n in coverage.items():r['coverage'][k]['pairs']=n
    r.update(model=method,dataset='responsenet full586',protocol_sha256=protocol_sha(),aggregation='Original scorer; pool full586 feature support; never average block FID/FVD',audio_metrics='N/A: no generated listener waveform',generation_coverage=dict(requested=TOTAL,generated=sum(not r['errors'].get('generation') for r in rows),unavailable_ids=[r['id'] for r in rows if r['errors'].get('generation')]),metric_code_sha256=sha(WORK/'METRIC_CODE.json'))
    save(WORK/'results'/f'{method}_responsenet.json',r);print(method,r['metrics'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['score','merge']);p.add_argument('method');p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='score':score(a.method,a.index)
    else:merge(a.method)
