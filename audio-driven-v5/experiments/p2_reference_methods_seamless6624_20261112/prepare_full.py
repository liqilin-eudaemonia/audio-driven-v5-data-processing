"""Freeze Seamless6624 and verify/read-only reuse of the sealed fixed200 outputs."""
from pathlib import Path
import sys,json,hashlib,time,shutil,math
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_more_common import sha,read,save
OLD=ROOT/'experiments/p2_reference_methods_dualtest_20261112'

def link(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() and not target.is_symlink():target.symlink_to(source,target_is_directory=source.is_dir())
    assert target.resolve()==source.resolve()

def main():
    # The old result itself remains sealed; verify its complete manifest before reuse.
    c=read(OLD/'COMPLETE.json');assert c['passed']
    for p,h in c['files'].items():assert sha(p)==h,p
    oldp=read(OLD/'protocol.json');spec=oldp['sources']['seamless'];assert sha(spec['input_protocol'])==spec['input_protocol_sha256']
    ss=[s for s in oldp['samples'] if s['dataset']=='seamless'];assert len(ss)==len({s['id'] for s in ss})==6624
    assert len([s for s in ss if s['quick200']])==200
    ss.sort(key=lambda s:(not s['quick200'],s['id']))
    protocol=dict(schema='reference-methods-seamless-full6624-v1',seed=137,scope='seamless6624',methods=oldp['methods'],sources={'seamless':spec},samples=ss,prior_protocol=str(OLD/'protocol.json'),prior_protocol_sha256=sha(OLD/'protocol.json'),reuse_scope='Fixed200 input-identical outputs; remainder6424 new',audio_metrics='N/A: no generated listener waveform')
    save(WORK/'protocol.json',protocol);ph=sha(WORK/'protocol.json')
    save(WORK/'RUN_SCOPE.json',dict(scope='seamless6624',user_confirmed=True,total=6624,keys=[s['key'] for s in ss],datasets=['seamless'],gpu_ids=[4,7]))
    save(WORK/'BATCHES.json',dict(size=25,groups=[[s['key'] for s in ss[i:i+25]] for i in range(0,len(ss),25)],fixed200_preserved=True,final_batch=24))
    # Freeze Python evaluator code. Runtime points its data root back at the original sealed sources.
    base=ROOT.parent/'OmniResponse/evaluation';pin=WORK/'pinned_metrics/evaluation'
    for sub in ['video/paired_metrics','video/fvd','seamless']:
        for p in (base/sub).glob('*.py'):
            dest=pin/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for sub in ['', 'video']:
        p=pin/sub/'__init__.py'
        if not p.exists():p.write_text('')
    code={p.name:sha(p) for p in (pin/'video/paired_metrics').glob('*.py')}
    code['../fvd/frechet_video_distance.py']=sha(pin/'video/fvd/frechet_video_distance.py');code['../../seamless/reader.py']=sha(pin/'seamless/reader.py')
    shutil.copyfile(ROOT/'scripts/p2_pipelined_video_metrics.py',WORK/'pinned_metrics/p2_pipelined_video_metrics.py')
    save(WORK/'METRIC_CODE.json',dict(expected=code,files={str(p):sha(p) for p in (WORK/'pinned_metrics').rglob('*.py')},data_root=str(ROOT),method='Original evaluator and pipeline copied byte-for-byte; runtime DATA_ROOT points at original project'))
    shutil.copyfile(OLD/'MODEL_MANIFEST.json',WORK/'MODEL_MANIFEST.json')
    reused={m:0 for m in protocol['methods']};reused_aux={};score_reused=[]
    for s in ss[:200]:
        assert s['quick200'];dataset=s['dataset'];uid=s['id']
        oldsample=next(x for x in oldp['samples'] if x['key']==s['key'])
        for k in ['reference_sha256','audio_sha256','frames','fps']:assert s[k]==oldsample[k]
        for area in ['features','evaluation_gt','gpavatar_references']:
            source=OLD/area/dataset/uid
            marker=source/('TRACK_UNAVAILABLE.json' if (source/'TRACK_UNAVAILABLE.json').exists() else 'complete.json');r=read(marker)
            if area=='features':
                assert r['input_image_sha256']==s['reference_sha256'] and r['input_audio_sha256']==s['audio_sha256']
            elif area=='gpavatar_references':assert r['source_image_sha256']==s['reference_sha256']
            for p,h in r.get('files',{}).items():assert sha(p)==h
            if area=='evaluation_gt':assert sha(r['path'])==r['sha256']
            link(source,WORK/area/dataset/uid);reused_aux[area]=reused_aux.get(area,0)+1
        for m in protocol['methods']:
            source=OLD/m/dataset/uid;dest=WORK/m/dataset/uid;dest.mkdir(parents=True,exist_ok=True)
            for name in ['MOTION_COMPLETE.json','complete.json']:
                receipt=source/name
                if not receipt.exists():continue
                r=read(receipt)
                if name=='complete.json':assert r['input_image_sha256']==s['reference_sha256'] and r['input_audio_sha256']==s['audio_sha256']
                files={}
                for p,h in r['files'].items():
                    assert sha(p)==h,p
                    target=dest/Path(p).name;link(Path(p),target);files[str(target)]=h
                r.update(protocol_sha256=ph,files=files,reused_from=str(receipt),reused_receipt_sha256=sha(receipt),at=time.time())
                if r.get('score_path'):r['score_path']=str(dest/'scoring.mp4')
                save(dest/name,r)
            reused[m]+=1
    # Preserve the original 8 stochastic L2L batches exactly.
    for name in ['conan','fallon','stephen','trevor']:
        for i in range(8):
            p=OLD/'l2l_batches'/name/'seamless'/f'{i:02d}.json';r=read(p)
            assert r['ids']==[s['id'] for s in ss[25*i:25*(i+1)]]
            for path,h in r['files'].items():assert sha(path)==h
            dest=WORK/'l2l_batches'/name/'seamless'/p.name;save(dest,dict(r,reused_from=str(p),reused_receipt_sha256=sha(p)))
    # Reuse scoring primitives only when frozen evaluator fingerprints already match.
    for m in protocol['methods']:
        for i in range(8):
            folder=OLD/'metrics'/m/'seamless'/f'{i:02d}'
            if not (folder/'SUCCESS.json').exists() or not (folder/'contract.json').exists():continue
            receipt=read(folder/'SUCCESS.json');contract=read(folder/'contract.json')
            if contract['code']!=code:continue
            for p,h in receipt['files'].items():assert sha(p)==h
            ids={json.loads(x)['id'] for x in (folder/'pairs.jsonl').read_text().splitlines()}
            if (folder/'UNAVAILABLE.json').exists():ids|={r['id'] for r in read(folder/'UNAVAILABLE.json')}
            assert ids=={s['id'] for s in ss[25*i:25*(i+1)]}
            link(folder,WORK/'metrics'/m/'seamless'/f'{i:02d}');score_reused.append(dict(method=m,index=i,source=str(folder),receipt_sha256=sha(folder/'SUCCESS.json')))
    save(WORK/'REUSE_COMPLETE.json',dict(passed=True,inputs=200,methods=reused,auxiliary=reused_aux,score_shards=score_reused,prior_complete_sha256=sha(OLD/'COMPLETE.json'),prior_input_audit_sha256=sha(OLD/'INPUT_COMPLETE.json'),protocol_sha256=ph,at=time.time()))
    print('Prepared Seamless6624, reused',reused_aux,'and',len(score_reused),'compatible scoring shards',flush=True)
if __name__=='__main__':main()
