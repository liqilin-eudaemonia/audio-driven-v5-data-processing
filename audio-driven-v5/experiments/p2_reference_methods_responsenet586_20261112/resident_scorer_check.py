"""Check cached official metric instances against one previously scored real clip."""
from runtime import *

def check():
    import numpy as np
    from metric_launcher import run_pipeline
    from scoring import destination,receipt_paths,scoring_device
    marker=WORK/'SCORER_CACHE_PARITY.json'
    base=destination('listenformer',0);receipt=read(base/'SUCCESS.json');pp=receipt_paths(base,receipt)
    original=json.loads(pp['pairs.jsonl'].read_text().splitlines()[0]);spec=read(WORK/'protocol.json')['sources'][selected()[0]['dataset']]
    scoring_device();checks=[];previous=None
    for index in range(2):
        d=WORK/'checks/scorer_resident'/f"run_{os.getpid()}_{index}";d.mkdir(parents=True,exist_ok=True)
        manifest=d/'manifest.jsonl';manifest.write_text(json.dumps(original['input'])+'\n')
        args=['--manifest',str(manifest),'--assets',spec['metric_assets'],'--output',str(d),'--device','cuda:0','--threads','2','--batch-size','64','--face-padding',str(spec['face_padding']),'--feature-output',str(d/'features.npz'),'--allow-partial','--face-device','cuda:0','--emotion-device','cuda:0','--pixel-workers','4','--gt-cache',spec['metrics_gt_cache']]
        code=run_pipeline(args,resident=True);assert code==0
        actual=json.loads((d/'pairs.jsonl').read_text().splitlines()[0]);assert actual['errors']==original['errors']
        for k,v in original['metrics'].items():np.testing.assert_allclose(actual['metrics'][k],v,rtol=1e-5,atol=1e-5)
        with np.load(pp['features.npz']) as old,np.load(d/'features.npz') as new:
            for k in ['fid_gt','fid_generated','fvd_gt','fvd_generated']:
                np.testing.assert_allclose(new[k],old[k][:len(new[k])],rtol=1e-5,atol=1e-5)
                if previous is not None:np.testing.assert_allclose(new[k],previous[k],rtol=1e-6,atol=1e-6)
            previous={k:new[k].copy() for k in new.files}
        checks.append(dict(run=index,pair=actual['id'],metrics_match_original=True,features_match_original=True))
    save(marker,dict(passed=True,checks=checks,adapter_sha256=sha(WORK/'metric_launcher.py'),at=time.time()))
    print('Original/cached metric parity passed; instances stay loaded for production blocks',flush=True)
