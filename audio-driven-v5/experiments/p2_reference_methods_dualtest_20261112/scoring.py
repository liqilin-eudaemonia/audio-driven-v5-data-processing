"""Original paired video scorer; fixed200 global FID/FVD merge."""
import argparse,subprocess,os,time,fcntl
from runtime import *
sys.path.insert(0,"/private/liqi_workspace/.cache/paired_video_metrics/python")
os.environ["PYTHONPATH"]="/private/liqi_workspace/.cache/paired_video_metrics/python"

def samples(dataset,index):return [s for s in selected() if s['dataset']==dataset][index*25:(index+1)*25]
def destination(method,dataset,index):return WORK/'metrics'/method/dataset/f'{index:02d}'
def ready(method,dataset,index):return all(valid(method,s) for s in samples(dataset,index))

def score(method,dataset,index):
    p=read(WORK/'protocol.json');spec=p['sources'][dataset];ss=samples(dataset,index);assert len(ss)==25 and ready(method,dataset,index)
    out=destination(method,dataset,index);out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'SUCCESS.json').exists():return
        manifest=[];unavailable=[]
        for s in ss:
            ref=s['evaluation_reference'];receipt=read(folder(method,s)/'complete.json');video=receipt['score_path']
            if video is None:
                assert receipt.get('official_tracker_no_identity') is True
                unavailable.append(dict(id=s['id'],identity='listener',metrics={},errors={'generation':receipt['reason']},generation_receipt=str(folder(method,s)/'complete.json')))
                continue
            if dataset=='seamless':r=dict(id=s['id'],dataset=dataset,identity='listener',gt_reference=ref['gt_reference'],generated=video,listening_intervals=ref['listening_intervals'])
            else:r=dict(id=s['id'],dataset='responsenet-j-paired-evaluation-v1',identity='listener',gt=ref['gt'],generated=video,gt_resize=[512,512],gt_resize_method='pil_bicubic',source_start=ref['source_start'],source_end=ref['source_end'],listening_intervals=ref['listening_intervals'])
            manifest.append(r)
        save(out/'UNAVAILABLE.json',unavailable)
        if not manifest:
            save(out/'SUCCESS.json',dict(clips=25,generated=0,unavailable=25,files={str(out/'UNAVAILABLE.json'):sha(out/'UNAVAILABLE.json')},at=time.time()));return
        device()
        f=out/'manifest.jsonl';f.write_text(''.join(json.dumps(r)+'\n' for r in manifest))
        cmd=[sys.executable,str(ROOT/'scripts/p2_pipelined_video_metrics.py'),'--manifest',str(f),'--assets',spec['metric_assets'],'--output',str(out),'--device','cuda:0','--threads','2','--batch-size','64','--face-padding',str(spec['face_padding']),'--feature-output',str(out/'features.npz'),'--allow-partial','--face-device','cuda:0','--emotion-device','cuda:0','--pixel-workers','4','--gt-cache',spec['metrics_gt_cache']]
        # Existing decoded-GT cache is optional; source readers and formulas stay identical.
        with (out/'video.log').open('w') as log:code=subprocess.call(cmd,cwd=ROOT.parent/'OmniResponse',stdout=log,stderr=subprocess.STDOUT)
        assert code==0, ('Scorer process failed',code,str(out/'video.log'))
        summary=read(out/'summary.json');assert not summary['initialization_errors'],summary
        actual=[json.loads(x) for x in (out/'pairs.jsonl').read_text().splitlines()]
        assert {r['id'] for r in actual}=={s['id'] for s in manifest}
        allowed_missing={
            'emotion':lambda e:e=='No common valid emotion frames',
            'delta_resp':lambda e:e=='ValueError: No measurable listening phases on common valid adjacent frames',
            'fvd':lambda e:e.startswith('ValueError: Short clip: ') and e.endswith('; no padding'),
        }
        technical=[(r['id'],k,e) for r in actual for k,e in r['errors'].items() if k not in allowed_missing or not allowed_missing[k](e)]
        assert not technical, ('Scoring execution errors; retry rather than publish partial',technical)
        assert (out/'features.npz').exists()
        save(out/'SUCCESS.json',dict(clips=25,generated=len(manifest),unavailable=len(unavailable),returncode=code,pairs_with_errors=summary['pairs_with_errors'],files={str(out/n):sha(out/n) for n in ['summary.json','contract.json','pairs.jsonl','features.npz','UNAVAILABLE.json']},at=time.time()))

def merge(method,dataset):
    import numpy as np
    from evaluation.video.paired_metrics.run import summarize,METRICS
    actual=[];contracts=[];dist={k:[[],[]] for k in ['fid','fvd']};coverage={k:0 for k in dist}
    for i in range(8):
        f=destination(method,dataset,i);receipt=read(f/'SUCCESS.json')
        for path,h in receipt['files'].items():assert sha(path)==h
        if (f/'UNAVAILABLE.json').exists():actual.extend(read(f/'UNAVAILABLE.json'))
        if receipt.get('generated',25)==0:continue
        s=read(f/'summary.json');contracts.append(read(f/'contract.json'));actual.extend(json.loads(x) for x in (f/'pairs.jsonl').read_text().splitlines())
        with np.load(f/'features.npz') as a:
            for k in dist:
                if a[k+'_gt'].size:dist[k][0].append(a[k+'_gt']);dist[k][1].append(a[k+'_generated'])
                coverage[k]+=s['coverage'][k]['pairs']
    assert len(actual)==len({r['id'] for r in actual})==200
    for c in contracts:assert c['code']==contracts[0]['code'] and c['assets']==contracts[0]['assets']
    r=summarize(actual,dist,set(METRICS),contracts[0]['schema'] if contracts else 'omniresponse-paired-video-v1',contracts[0]['datasets'] if contracts else [dataset],200,{})
    for k,n in coverage.items():r['coverage'][k]['pairs']=n
    r.update(generation_coverage=dict(requested=200,generated=sum(not r['errors'].get('generation') for r in actual),unavailable_ids=[r['id'] for r in actual if r['errors'].get('generation')]),model=method,dataset=dataset+' fixed200',protocol_sha256=sha(WORK/'protocol.json'),audio_metrics='N/A: no listener waveform',aggregation='original scorer, all200 pooled features; no mean of shard FID/FVD')
    save(WORK/'results'/f'{method}_{dataset}.json',r)
    print(json.dumps(r['metrics']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['score','merge']);p.add_argument('method');p.add_argument('dataset');p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='score':score(a.method,a.dataset,a.index)
    else:merge(a.method,a.dataset)
